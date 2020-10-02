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


import tensorflow as tf


# In[ ]:


print(tf.__version__)


# In[ ]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()


# In[ ]:


train_images.shape


# In[ ]:


# train_images[0]
train_images[0].ndim


# In[ ]:


train_images[0].shape


# In[ ]:


train_labels[0]


# In[ ]:


train_labels[2]


# In[ ]:


from tensorflow.keras import models
from tensorflow.keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# MultiClassification Problem so "Softmax"
network.add(layers.Dense(10, activation='softmax'))
# probability will be check


# In[ ]:


network.compile(
    optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
# loss function


# In[ ]:


train_images = train_images.reshape((60000, 28 * 28)) # here converting 3d to 2d 
train_images = train_images.astype('float32') / 255 # all values will come under the range of 0 - 1

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255


# In[ ]:


from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# In[ ]:


network.fit(
    train_images,
    train_labels,
    epochs=5, # no. of iteration
    batch_size=128 # chunk of input from total input
)


# In[ ]:


test_loss, test_acc = network.evaluate(test_images, test_labels)
print('Test Accuracy: ', test_acc)


# # Plot Image

# In[ ]:


from tensorflow.keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# In[ ]:


train_labels[4]

