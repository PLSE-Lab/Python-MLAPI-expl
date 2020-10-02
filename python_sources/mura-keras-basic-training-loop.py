#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import tensorflow as tf
from tensorflow.keras import layers
from skimage.io import imshow
from pathlib import Path
import pandas as pd
from tensorflow.keras.callbacks import ReduceLROnPlateau

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


dataset_root = Path('../input/mura/MURA-v1.1')


# In[ ]:


list(dataset_root.iterdir())


# In[ ]:


df = pd.read_csv(dataset_root/'train_image_paths.csv', header=None, names=['filename'])
df.head()


# In[ ]:


df['class'] = (df.filename
               .str.extract('study.*_(positive|negative)'))
df.head()


# In[ ]:


def generate_df(dataset_root, csv_name):
    df = pd.read_csv(dataset_root/csv_name, header=None, names=['filename'])
    df['class'] = (df.filename
               .str.extract('study.*_(positive|negative)'))
    return df


# In[ ]:


list(dataset_root.parent.iterdir())


# Since the dataset images are all very rectangular and not even close to being a square, we really need to be careful about how the images are resized. What could happen 
# 
# However, for the first draft of the training pipeline, I will just be lazy and use the default resizing method provided by Keras.

# In[ ]:


datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1. / 255)

train_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'train_image_paths.csv'),
                                        directory=dataset_root.parent,
                                        target_size=(512, 512),
                                        class_mode='binary')
valid_gen = datagen.flow_from_dataframe(generate_df(dataset_root, 'valid_image_paths.csv'),
                                        directory=dataset_root.parent,
                                        target_size=(512, 512),
                                        class_mode='binary')


# In[ ]:


densenet = tf.keras.applications.DenseNet169(weights='imagenet', include_top = False, input_shape=(512, 512, 3))


# In[ ]:


densenet.trainable = False


# In[ ]:


model = tf.keras.models.Sequential([
    densenet,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1, activation='sigmoid')
])


# In[ ]:


lr_scheduler = ReduceLROnPlateau(monitor='accuracy', factor=0.1, patience=10, verbose=0, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)


# In[ ]:


model.compile(loss=tf.keras.losses.binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])


# In[ ]:


model.fit_generator(train_gen, 
                    epochs=100, 
                    validation_data=valid_gen, 
                    use_multiprocessing=True, 
                    callbacks=[lr_scheduler])


# In[ ]:




