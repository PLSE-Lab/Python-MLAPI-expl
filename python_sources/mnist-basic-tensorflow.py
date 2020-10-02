#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # MNIST - TensorFlow Basics
# 
# The objective of this notebook is to build a basic model for MNIST dataset using TensorFlow. This code is from [pythonprogramming.net](https://pythonprogramming.net/introduction-deep-learning-python-tensorflow-keras/)

# In[ ]:


import tensorflow as tf

tf.__version__


# ## MNIST Data
# Load MNIST dataset. MNIST data is a 28 x 28 images of hand-written digits from 0 to 9.

# In[ ]:


def load_data():
    with np.load("../input/mnist.npz") as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train[0])


# ## Plot Data
# This is to show the shape of the sample set` x_train[0]`.

# In[ ]:


import matplotlib.pyplot as plt

plt.imshow(x_train[0], cmap = plt.cm.binary)
plt.show();


# ## Scale the Data
# Normalizing the MNIST dataset

# In[ ]:


x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)
print(x_train[0])


# ## Build the Model

# In[ ]:


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# ## Model Parameters 

# In[ ]:


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# ## Train the Model

# In[ ]:


model.fit(x_train, y_train, epochs=10)


# ## Model Evaluation 

# In[ ]:


val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Saving Model
# model.save('num_reader.model')

# Loading Model
# model = tf.keras.models.load_models('num_reader.model')


# ## Model Predictions

# In[ ]:


predictions = model.predict([x_test])

print(np.argmax(predictions[35]))
plt.imshow(x_test[35], cmap=plt.cm.binary)
plt.show();


# In[ ]:




