#!/usr/bin/env python
# coding: utf-8

# ## About this kernel
# 
# Most of this notebook is blatantly stolen from https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus
# 
# I've uploaded some tfrecord datasets. All permutations of the same cover image should be separated into each tfrecord file. So two tfrecord files should never have any image that matches the cover. This should make validation better, because you don't want the model to see the cover in train and then the stega image in validation.
# 
# The main contribution here is the 6 alaska datasets that have been turned into TFRecords and a cache trick. 
# 
# Because TPUs have limited ram, if you decode the JPEG and then cache it, you will run out of ram because you are caching an uncompressed JPEG, which is massive. Instead, cache the JPEG binary and decode it. TPUs have alot of CPU cores for that job and your main bottleneck is usually file access. 
# 
# Also, the tfrecords stream better than 300k images as separate files. 
# 
# Tweak it yourself and experiment!
# 

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import math, re, os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
import efficientnet.tfkeras as efn
from sklearn import metrics
from sklearn.model_selection import train_test_split
import cv2


# ## TPU Strategy and other configs 

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is
    # set: this is always the case on Kaggle.
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access

GCS_DS_PATH = KaggleDatasets().get_gcs_path('alaska2-image-steganalysis')


# Configuration
EPOCHS = 30
BATCH_SIZE = 32 * strategy.num_replicas_in_sync


# In[ ]:


import glob
from tqdm.notebook import tqdm 

all_records = []
for i in tqdm(range(7)):
    
    records = glob.glob('/kaggle/input/alaska0%i/*' % i )
    TF_REC_DS_PATH = KaggleDatasets().get_gcs_path('alaska0%i' % i )
    records = [os.path.join(TF_REC_DS_PATH, record[-14:]) for record in records]
    all_records += records 
    
all_records = [record for record in all_records if int(record.split('/')[-1].split('.')[0]) < 200]
    
train_records, valid_records = train_test_split(all_records, test_size = 0.1)

len(all_records)


# In[ ]:





# ## Create Dataset objects
# 
# A `tf.data.Dataset` object is needed in order to run the model smoothly on the TPUs. Here, I heavily trim down [my previous kernel](https://www.kaggle.com/xhlulu/flowers-tpu-concise-efficientnet-b7), which was inspired by [Martin's kernel](https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu).

# In[ ]:


# consider random crop of the data rather than resize

feature_description = {
    'bits': tf.io.FixedLenFeature([], tf.string, default_value=''),
    'label' : tf.io.FixedLenFeature([], tf.int64, default_value=0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
  return tf.io.parse_single_example(example_proto, feature_description)

def split(features):
    return features['bits'], features['label']


def read_image(filename, label=None):
    bits = tf.io.read_file(filename)
    if label is None:
        return bits
    else:
        return bits, label

binary = False
def convert_label(label):
    if binary:
        return tf.cast(label != 0, tf.int64)
    else:
        return label
    
    
def decode_image(bits, label=None, image_size=(512, 512)):
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, convert_label(label)
    


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label


# In[ ]:



train_dataset = (
    tf.data.TFRecordDataset(train_records, num_parallel_reads=AUTO)
    .map(_parse_function, num_parallel_calls=AUTO)
    .map(split, num_parallel_calls=AUTO)
    .cache()
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls = AUTO)
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)


valid_dataset = (
    tf.data.TFRecordDataset(valid_records, num_parallel_reads=AUTO)
    .map(_parse_function, num_parallel_calls=AUTO)
    .map(split, num_parallel_calls=AUTO)
    .cache()
    .map(decode_image, num_parallel_calls=AUTO)
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset.take(1)


# In[ ]:


sub = pd.read_csv('/kaggle/input/alaska2-image-steganalysis/sample_submission.csv')
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))
test_paths = append_path('Test')(sub.Id.values)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(read_image, num_parallel_calls=AUTO)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


# ## Modelling

# ### Helper Functions

# In[ ]:


def build_lrfn(lr_start=0.00001, lr_max=1e-3, 
               lr_min=0.000001, lr_rampup_epochs=2, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn


# ### Load Model into TPU

# In[ ]:


if binary:
    out = 1
    loss_fn = tf.keras.losses.BinaryCrossentropy
    metrics = ['accuracy']
else:
    out = 4
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy
    metrics = ['sparse_categorical_accuracy']

with strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB2(
            input_shape=(512, 512, 3),
            weights='imagenet',
            include_top=False
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(out, activation='sigmoid')
    ])

#     optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3 * 8)
    
    model.compile(
        optimizer='adam',
        loss = loss_fn(),
        metrics=metrics 
    )
    model.summary()


# ### Start training

# In[ ]:


lrfn = build_lrfn(lr_start = 1e-5 ,lr_max = 1e-3 * 4, lr_rampup_epochs=4, 
               lr_sustain_epochs=8, lr_exp_decay=.9)

lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)

# STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE

STEPS_PER_EPOCH = 100

history = model.fit(
    train_dataset, 
    epochs=EPOCHS, 
    callbacks=[lr_schedule],
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_data=valid_dataset,
#     validation_steps = 120
)


# In[ ]:


model.save("model.h5")


# ## Evaluation

# Unhide below to see helper function `display_training_curves`:

# ## Submission

# In[ ]:



if not binary: 

    preds = model.predict(test_dataset, verbose=1)
    y_pred = 1 - tf.nn.softmax(preds)[:, 0]
    sub.Label = y_pred


sub.to_csv('submission.csv', index=False)
sub.head()


# In[ ]:


# np.mean(y_pred)

