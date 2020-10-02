#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().system('pip install keras-tuner')


# In[ ]:


import numpy as np


# In[ ]:


import tensorflow as tf
from tensorflow import keras


# In[ ]:


print(tf.__version__)


# In[ ]:


fm=keras.datasets.fashion_mnist


# In[ ]:


(train_images,train_labels),(test_images,test_labels)=fm.load_data()


# In[ ]:


train_images


# In[ ]:


train_images=train_images/255.0


# In[ ]:


test_images=test_images/255.0


# In[ ]:


train_images[0].shape


# # We have to reshape it to (28,28,1) for the input in CNN

# In[ ]:


len(train_images)


# In[ ]:


train_images=train_images.reshape(len(train_images),28,28,1)


# In[ ]:


test_images=test_images.reshape(len(test_images),28,28,1)


# In[ ]:


def build_model(hp):  
  model = keras.Sequential([
    keras.layers.Conv2D(
        filters=hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
        kernel_size=hp.Choice('conv_1_kernel', values = [3,5]),
        activation='relu',
        input_shape=(28,28,1)
    ),
    keras.layers.Conv2D(
        filters=hp.Int('conv_2_filter', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv_2_kernel', values = [3,5]),
        activation='relu'
    ),
    keras.layers.Flatten(),
    keras.layers.Dense(
        units=hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
    ),
    keras.layers.Dense(10, activation='softmax')
  ])
  
  model.compile(optimizer=keras.optimizers.Adam(hp.Choice('learning_rate', values=[1e-2, 1e-3])),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
  
  return model


# In[ ]:


from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters


# In[ ]:


tuner_search=RandomSearch(build_model,objective='val_accuracy',max_trials=5,directory='output',project_name="Mnist Fashion")


# In[ ]:


tuner_search.search(train_images,train_labels,epochs=3,validation_split=0.1)


# In[ ]:


model=tuner_search.get_best_models(num_models=1)[0]


# In[ ]:


model.summary()


# In[ ]:


model.fit(train_images, train_labels, epochs=10, validation_split=0.1, initial_epoch=3)


# # Accuracy reached to 99.39% using Hyperparameter Optimisation..
