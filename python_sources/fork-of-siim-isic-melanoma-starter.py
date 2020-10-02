#!/usr/bin/env python
# coding: utf-8

#    This notebook implement the model use pretrain network, easy to undertand and custom.
# 
#     If it is useful or you want to fork it :) please vote for this kernel.
#     
#     I am very happy if get some idea or suggest from you.
#     
#     Good look!!!

# # Import Libraries

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import math, os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
import efficientnet.tfkeras as efn
import tensorflow.keras.backend as K
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from albumentations import *


# # Configurations

# In[ ]:


# Detect hardware, return appropriate distribution strategy
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 2048
seed_everything(seed)
print("REPLICAS: ", strategy.num_replicas_in_sync)

# Data access
from kaggle_datasets import KaggleDatasets
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
# GCS_DS_PATH = './'

AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


# Configuration
IMAGE_SIZE = [256, 256]
EPOCHS = 15
SEED = 777
BATCH_SIZE = 32 * strategy.num_replicas_in_sync
SPLIT_VALIDATION = True
SPLIT_FOLD = False


# # Mixed Precision and/or XLA
# The following booleans can enable mixed precision and/or XLA on GPU/TPU. By default TPU already uses some mixed precision but we can add more. These allow the GPU/TPU memory to handle larger batch sizes and can speed up the training process. The Nvidia V100 GPU has special Tensor Cores which get utilized when mixed precision is enabled. Unfortunately Kaggle's Nvidia P100 GPU does not have Tensor Cores to receive speed up.

# In[ ]:


MIXED_PRECISION = False
XLA_ACCELERATE = True

if MIXED_PRECISION:
    from tensorflow.keras.mixed_precision import experimental as mixed_precision
    if tpu: policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
    else: policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
    mixed_precision.set_policy(policy)
    print('Mixed precision enabled')

if XLA_ACCELERATE:
    tf.config.optimizer.set_jit(True)
    print('Accelerated Linear Algebra enabled')


# # Import Data

# In[ ]:


train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')
test = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/test.csv')
sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')

train_paths = train.image_name.apply(lambda x: GCS_DS_PATH + '/jpeg/train/' + x + '.jpg').values
test_paths = test.image_name.apply(lambda x: GCS_DS_PATH + '/jpeg/test/' + x + '.jpg').values

train_labels = train['target'].values


# # Dataset Functions

# In[ ]:


def decode_image(filename, label=None):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size needed for TPU
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
    # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
    # of the TPU while the TPU itself is computing gradients.
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_hue(image, 0.01)
    image = tf.image.random_saturation(image, 0.7, 1.3)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_brightness(image, 0.1)
    
    if label is None:
        return image
    else:
        return image, label

def get_training_dataset(paths, labels, do_aug=True):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(decode_image, num_parallel_calls=AUTO)
    if do_aug:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.cache()
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(paths, labels, ordered=False):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels)).map(decode_image, num_parallel_calls=AUTO)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO)
    return dataset

def get_test_dataset(paths, ordered=False):
    dataset = tf.data.Dataset.from_tensor_slices(paths).map(decode_image, num_parallel_calls=AUTO).batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset


# # Custom Callbacks

# In[ ]:


def build_lrfn(lr_start=0.000005, lr_max=0.000020, 
               lr_min=0.000001, lr_rampup_epochs=5, 
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

lr_schedule = tf.keras.callbacks.LearningRateScheduler(build_lrfn(), verbose=1)
callbacks = [lr_schedule]


# # Ensemble model, Train, Inference

# In[ ]:


def get_model(name='EfficientNetB3', weights="imagenet"):
    with strategy.scope():
        if name == 'InceptionResNetV2':
            model = tf.keras.Sequential([
                InceptionResNetV2(
                    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                    weights=weights,
                    include_top=False
                ),
                GlobalMaxPooling2D(),
                Dense(1, activation='sigmoid')
            ])
        else:
            model = tf.keras.Sequential([
                efn.EfficientNetB3(
                    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
                    weights=weights,
                    include_top=False
                ),
                GlobalAveragePooling2D(),
                Dense(1, activation='sigmoid')
            ])

        model.compile(optimizer='adam',
                      loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
                      metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def train(model_obj, training_data, validation_data):
    model_obj.summary()
    model_obj.fit(training_data,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
                    validation_data=validation_data)

    return model_obj


# In[ ]:


if SPLIT_VALIDATION:
    trn_paths, val_paths, trn_labels, val_labels = train_test_split(train_paths, train_labels, test_size=0.2, random_state=seed)
else:
    trn_paths, val_paths = train_paths, train_labels

models = [
    'EfficientNetB3_noisy-student',
    'EfficientNetB3'
]

probs = []
for mid, model_name in enumerate(models):
    if model_name == 'EfficientNetB3_noisy-student':
        model = get_model(name='EfficientNetB3', weights='noisy-student')
    else:
        model = get_model(name=model_name)

    model = train(
        model,
        get_training_dataset(trn_paths, trn_labels, do_aug=True),
        get_validation_dataset(val_paths, val_labels) if SPLIT_VALIDATION else None
    )
    model.save_weights('model_' + str(mid) + '.h5')

    prob = model.predict(get_test_dataset(test_paths), verbose=1)
    sub['target'] = prob
    sub.to_csv('submission' + str(mid) + '.csv', index=False)
    probs.append(prob)


# In[ ]:


sub['target'] = np.mean(probs, 0)
sub.to_csv('submission.csv', index=False)
sub.head()


# In[ ]:


# !kaggle competitions submit -c siim-isic-melanoma-classification -f submission.csv -m "Message"

