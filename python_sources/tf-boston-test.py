#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install tensorflow==2.0.0-alpha0')


# In[ ]:


#%load_ext tensorboard.notebook
#%tensorboard --logdir logs


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import Normalizer


# In[ ]:


#!pip install tensorflow==2.0.0-alpha0


# In[ ]:


print(tf.__version__)


# In[ ]:


#df = pd.read_csv('../input/housing.csv')


# In[ ]:


#df.shape


# In[ ]:


data = tf.keras.datasets.boston_housing


# In[ ]:


(x_train, y_train),(x_test, y_test) = data.load_data()


# In[ ]:


x_train = tf.keras.utils.normalize(x_train)
#y_train = tf.keras.utils.normalize(y_train)
x_test = tf.keras.utils.normalize(x_test)
#y_test = tf.keras.utils.normalize(y_test)


# In[ ]:


x_train.shape, x_test.shape, y_train.shape, y_test.shape


# In[ ]:


type(x_train)


# In[ ]:


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(13, )),
  tf.keras.layers.Dense(23, activation=tf.nn.relu),
  tf.keras.layers.Dense(23, activation=tf.nn.relu),
  tf.keras.layers.Dense(23, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(1)
])


# In[ ]:


type(model.weights), len(model.weights), model.weights[0].shape


# In[ ]:


model.weights[0][0]


# In[ ]:


model.compile(optimizer='adam',
              loss='mean_squared_error',
              metrics=['mean_squared_error'])


# In[ ]:


history = model.fit(x_train, y_train, 
                    epochs=500, 
                    verbose = 0)


# In[ ]:


model.evaluate(x_test, y_test)


# In[ ]:


history.history.keys()


# In[ ]:


fig, axes = plt.subplots(figsize = (12,4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.subplot(1, 2, 2)
plt.plot(history.history['mean_squared_error'])
plt.xlabel('epoch')
plt.ylabel('mean_squared_error')
None


# In[ ]:




