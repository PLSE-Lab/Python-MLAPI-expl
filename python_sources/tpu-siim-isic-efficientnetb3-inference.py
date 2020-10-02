#!/usr/bin/env python
# coding: utf-8

# # About this kernel
# - TPU tensorflow EfficientNetB3 starter code
# - 4 folds
# - References are below
# - https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
# - https://www.kaggle.com/xhlulu/alaska2-efficientnet-on-tpus 

# # Library

# In[ ]:


get_ipython().system('pip install -q efficientnet')


# In[ ]:


import os
import re

import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.layers as L

import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets


# # TPU Strategy and other configs 

# In[ ]:


try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

print("REPLICAS: ", strategy.num_replicas_in_sync)


# In[ ]:


# For tf.dataset
AUTO = tf.data.experimental.AUTOTUNE

# Data access
GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

# Configuration
DEBUG = False
N_FOLD = 4
EPOCHS = 1 if DEBUG else 7
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]


# # Prepare Data & Loader

# In[ ]:


sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
test_files = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')


# In[ ]:


def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    idnum = example['image_name']
    return image, idnum

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(ignore_order)
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def get_test_dataset(test_files, ordered=False):
    dataset = load_dataset(test_files, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


# # Model

# In[ ]:


def get_model():
    
    with strategy.scope():
        model = tf.keras.Sequential([
            efn.EfficientNetB3(
                input_shape=(*IMAGE_SIZE, 3),
                weights=None,
                include_top=False
            ),
            L.GlobalAveragePooling2D(),
            L.Dense(1, activation='sigmoid')
        ])
    
    return model


# # Inference

# In[ ]:


from tqdm import tqdm

pred_df = pd.DataFrame()

tk0 = tqdm(range(N_FOLD), total=N_FOLD)

for fold in tk0:
    num_test = count_data_items(test_files)
    test_ds = get_test_dataset(test_files, ordered=True)
    test_images_ds = test_ds.map(lambda image, idnum: image)
    model = get_model()
    model.load_weights(f"../input/tpu-siim-isic-efficientnetb3-training/fold{fold}_model.h5")
    probabilities = model.predict(test_images_ds)
    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(num_test))).numpy().astype('U')
    _pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
    pred_df = pd.concat([pred_df, _pred_df])


# In[ ]:


mean_pred_df = pred_df.groupby('image_name', as_index=False).mean()
mean_pred_df.columns = ['image_name', 'target']
del sub['target']
sub = sub.merge(mean_pred_df, on='image_name')
sub.to_csv('submission.csv', index=False)
sub.head()

