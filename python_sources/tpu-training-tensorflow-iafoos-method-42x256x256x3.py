#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -q ../input/kaggle-efficientnet-repo/efficientnet-1.0.0-py3-none-any.whl')


# In[ ]:


import os
import random
import numpy as np
import pandas as pd
import argparse
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.metrics import categorical_accuracy, top_k_categorical_accuracy
from tensorflow.keras import layers as L
import efficientnet.tfkeras as efn
import tensorflow.keras.backend as K


# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE


# In[ ]:


train_fns = ['gs://kds-bff29645c21bc6643776eb0d7a601dd1112709d8e74064e9a9b228e9/0_train00-2104.tfrec',
 'gs://kds-bff29645c21bc6643776eb0d7a601dd1112709d8e74064e9a9b228e9/1_train00-2103.tfrec',
 'gs://kds-bff29645c21bc6643776eb0d7a601dd1112709d8e74064e9a9b228e9/2_train00-2103.tfrec',
 'gs://kds-bff29645c21bc6643776eb0d7a601dd1112709d8e74064e9a9b228e9/3_train00-2103.tfrec']
val_fns = ['gs://kds-bff29645c21bc6643776eb0d7a601dd1112709d8e74064e9a9b228e9/4_train00-2103.tfrec']


# In[ ]:


def get_strategy():
  # Detect hardware, return appropriate distribution strategy
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection
        print('Running on TPU ', tpu.cluster_spec().as_dict()['worker'])
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()

    print('REPLICAS: ', strategy.num_replicas_in_sync)
    return strategy
strategy = get_strategy()


# In[ ]:


backbone_name = 'efficientnet-b0'
N_TILES = 42
IMG_SIZE = 256


# In[ ]:


# tf.keras.mixed_precision.experimental.set_policy('mixed_float16')


# In[ ]:


class ConvNet(tf.keras.Model):

    def __init__(self, engine, input_shape, weights):
        super(ConvNet, self).__init__()
        
        self.engine = engine(
            include_top=False, input_shape=input_shape, weights=weights)
        
        
        self.avg_pool2d = tf.keras.layers.GlobalAveragePooling2D()
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense_1 = tf.keras.layers.Dense(1024)
        self.dense_2 = tf.keras.layers.Dense(1)

    @tf.function
    def call(self, inputs, **kwargs):
        x = tf.reshape(inputs, (-1, IMG_SIZE, IMG_SIZE, 3))
        x = self.engine(x)
        shape = x.shape
        x = tf.reshape(x, (-1, N_TILES, shape[1], shape[2], shape[3])) 
        x = tf.transpose(x, perm=[0, 2, 1, 3, 4])
        x = tf.reshape(x, (-1, shape[1], N_TILES*shape[2], shape[3])) 
        x = self.avg_pool2d(x)
        x = self.dropout(x, training=kwargs.get('training', False))
        x = self.dense_1(x)
        x = tf.nn.relu(x)
        return self.dense_2(x)


# In[ ]:


if backbone_name.startswith('efficientnet'):
    model_fn = getattr(efn, f'EfficientNetB{backbone_name[-1]}')


# In[ ]:


with strategy.scope():
    model = ConvNet(engine=model_fn, input_shape=(IMG_SIZE, IMG_SIZE, 3), weights='imagenet') 
    model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001), loss="mean_squared_error")    


# In[ ]:


lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.5, patience=3, verbose=1, mode='min',
    min_delta=0.0001, cooldown=0, min_lr=0.000001)
checkpoint = tf.keras.callbacks.ModelCheckpoint(backbone_name+'.h5', monitor='val_loss',
    verbose=1, save_best_only=True, save_weights_only=True, mode='min')


# In[ ]:


def flip(x: tf.Tensor) -> tf.Tensor:
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)
    return x

def decode_image(image_data, rand = True):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.reshape(image, [*(IMG_SIZE, IMG_SIZE), 3]) # explicit size needed for TPU
    if rand:
        image = flip(image)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    return image

def collage_image(example, rand = False):
    images = []
    k = 0
    while k < N_TILES:
        images.append(decode_image(example['image_'+str(k)], rand))
        k += 1
    images = tf.stack(images)

    return images


def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {}
    k = 0
    while k < N_TILES:
        LABELED_TFREC_FORMAT['image_'+str(k)] = tf.io.FixedLenFeature([], tf.string)
        k += 1
    LABELED_TFREC_FORMAT['label'] = tf.io.FixedLenFeature([], tf.int64)
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = collage_image(example, rand=True)
    label = tf.cast(example['label'], tf.int32)
    return image, label

def read_labeled_tfrecord_noshuffle(example):
    LABELED_TFREC_FORMAT = {}
    k = 0
    while k < N_TILES:
        LABELED_TFREC_FORMAT['image_'+str(k)] = tf.io.FixedLenFeature([], tf.string)
        k += 1
    LABELED_TFREC_FORMAT['label'] = tf.io.FixedLenFeature([], tf.int64)
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = collage_image(example)
    label = tf.cast(example['label'], tf.int32)
    return image, label

def load_dataset(filenames, labeled=True, ordered=False, training=True):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    
    if training == False:
        dataset = dataset.map(read_labeled_tfrecord_noshuffle)
    else:
        dataset = dataset.map(read_labeled_tfrecord)
    return dataset

def get_training_dataset(TRAINING_FILENAMES):
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(TEST_FILENAMES, ordered=False):
    dataset = load_dataset(TEST_FILENAMES, labeled=True, ordered=ordered, training=False)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset


# In[ ]:


num_train_samples = sum(int(fn.split('-')[-1].split('.')[0]) for fn in train_fns)
num_val_samples = sum(int(fn.split('-')[-1].split('.')[0]) for fn in val_fns)
num_train_samples, num_val_samples 


# In[ ]:


BATCH_SIZE = 64
STEPS_PER_EPOCH = num_train_samples // BATCH_SIZE
VAL_STEPS_PER_EPOCH = num_val_samples // BATCH_SIZE


# In[ ]:


history = model.fit(get_training_dataset(train_fns), steps_per_epoch=STEPS_PER_EPOCH, verbose=1,
    validation_data = get_validation_dataset(val_fns), validation_steps=VAL_STEPS_PER_EPOCH, epochs=30, callbacks = [lr_callback, checkpoint])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




