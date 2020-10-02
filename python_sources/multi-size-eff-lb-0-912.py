#!/usr/bin/env python
# coding: utf-8

# # Thank to https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/160147
# 
# I tried to ensemble use https://www.kaggle.com/truonghoang/stacking-ensemble-on-my-submissions?scriptVersionId=37760143 and got **LB 0.914**
# 
# Hope that I get some improves from this notebook
# 
# Good luck to all !!!

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import os, random
import re
import numpy as np
import pandas as pd
import math

import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras import backend as K

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets


# In[ ]:


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
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# Configuration
EPOCHS = 12
SEED = 2048
BATCH_SIZE = 16 * strategy.num_replicas_in_sync


# In[ ]:


def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    random.seed(SEED)

seed_everything(SEED)


# In[ ]:


sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')


# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [IMAGE_SIZE, IMAGE_SIZE, 3]) # explicit size needed for TPU
    return image

def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "target": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    return image, label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
        "image_name": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['image_name']
    return image, idnum # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):
    # Read from TFRecords. For optimal performance, reading from multiple files at once and
    # disregarding data order. Order does not matter since we will be shuffling the data anyway.

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    if labeled: 
        dataset = dataset.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        dataset = dataset.map(read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
    return dataset

def data_augment(image, label):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return image, label   

def get_training_dataset(files):
    dataset = load_dataset(files, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(files, ordered=False):
    dataset = load_dataset(files, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(filenames, labeled = False, aug=False, ordered = True):
    dataset = load_dataset(filenames, labeled = labeled, ordered = ordered)
    if aug:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    # prefetch next batch while training (autotune prefetch buffer size)
    dataset = dataset.prefetch(AUTO) 
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


# In[ ]:


def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
               lr_min=0.000001, lr_rampup_epochs=20, 
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


# In[ ]:


GCS_PATH_SELECT = {
    '256': KaggleDatasets().get_gcs_path('melanoma-256x256'),
    '384': KaggleDatasets().get_gcs_path('melanoma-384x384'),
    '512': KaggleDatasets().get_gcs_path('melanoma-512x512'),
    '768': KaggleDatasets().get_gcs_path('melanoma-768x768'),
    '512x': KaggleDatasets().get_gcs_path('512x512-melanoma-tfrecords-70k-images'),
}

lr_schedule = tf.keras.callbacks.LearningRateScheduler(build_lrfn(), verbose=1)


# In[ ]:


probs = []
probs_tta = []

for size, path in GCS_PATH_SELECT.items():
    if size == '512x':
        IMAGE_SIZE = 512
    else:
        IMAGE_SIZE = int(size)
    TRAINING_FILENAMES = tf.io.gfile.glob(path + '/train*.tfrec')
    STEPS_PER_EPOCH = count_data_items(TRAINING_FILENAMES) / BATCH_SIZE

    with strategy.scope():
        base_model = efn.EfficientNetB3(weights='imagenet', include_top=False, input_shape=(None, None, 3))
        x = base_model.output
        x = L.GlobalAveragePooling2D()(x)
        x = L.Dense(1, activation='sigmoid')(x)

        model = Model(inputs=base_model.input, outputs=x)

        model.compile(
            optimizer='adam',
            loss = 'binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC()]
        )
    
    model.fit(
        get_training_dataset(TRAINING_FILENAMES), 
        epochs=EPOCHS, 
        callbacks=[lr_schedule],
        steps_per_epoch=STEPS_PER_EPOCH
    )

    model.save("model_" + size + ".h5")

    print('Generating submission.csv file...')
    TEST_FILENAMES = tf.io.gfile.glob(path + '/test*.tfrec')
    test_ds = get_test_dataset(TEST_FILENAMES, ordered=True)
    test_images_ds = test_ds.map(lambda image, image_name: image)
    prob = model.predict(test_images_ds)

    test_ids_ds = test_ds.map(lambda image, image_name: image_name).unbatch()
    test_ids = next(iter(test_ids_ds.batch(count_data_items(TEST_FILENAMES)))).numpy().astype('U') # all in one batch
    pred_df = pd.DataFrame({'image_name': test_ids, 'target': prob[:, 0]})
    sub.drop('target', inplace = True, axis = 1)
    sub = sub.merge(pred_df, on='image_name')
    sub.to_csv('submission_' + size + '.csv', index = False)
    probs.append(sub['target'])
    
    K.clear_session()


# In[ ]:


sub['target'] = np.mean(probs, 0)
sub.to_csv('submission.csv', index=False)

