#!/usr/bin/env python
# coding: utf-8

# #### Training model using TPU and save model

# In[ ]:


import numpy as np
import pandas as pd
import os
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from kaggle_datasets import KaggleDatasets

print(tf.__version__)
print(tf.keras.__version__)


# In[ ]:


get_ipython().system('pip install efficientnet')
import efficientnet.tfkeras as efn


# In[ ]:


AUTO = tf.data.experimental.AUTOTUNE
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

print("REPLICAS: ", strategy.num_replicas_in_sync)


# Data access
GCS_DS_PATH = KaggleDatasets().get_gcs_path()


# In[ ]:


train_df = pd.read_csv("../input/imet-2020-fgvc7/train.csv")
train_df["attribute_ids"]=train_df["attribute_ids"].apply(lambda x:x.split(" "))
train_df["id"]=train_df["id"].apply(lambda x:x+".png")
train_df["id"]=train_df["id"].apply(lambda x:'/train/' + x)

print(train_df.shape)
train_df.head()


# In[ ]:


train_paths = train_df["id"].apply(lambda x: GCS_DS_PATH + x).values


# In[ ]:


from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()

train_df_d = pd.DataFrame(mlb.fit_transform(train_df["attribute_ids"]),columns=mlb.classes_, index=train_df.index)

print(train_df_d.shape)
train_df_d.head()


# In[ ]:


train_df_d[:1][['448','2429','782']]


# In[ ]:


train_labels = train_df_d.astype('int32').values

train_labels


# In[ ]:


import gc

del train_df_d
gc.collect()


# In[ ]:


BATCH_SIZE= 8 * strategy.num_replicas_in_sync
img_size = 32
EPOCHS = 1
nb_classes = 3471


# In[ ]:


def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    if label is None:
        return image
    else:
        return image, label


# In[ ]:


train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
    )


# In[ ]:


gc.collect()


# In[ ]:


def get_model():
    base_model =  efn.EfficientNetB0(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)


# In[ ]:


with strategy.scope():
    model = get_model()
    
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


# In[ ]:


get_ipython().run_cell_magic('time', '', 'model.fit(\n    train_dataset, \n    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,\n    epochs=EPOCHS\n)')


# In[ ]:


model.save('model.h5')

