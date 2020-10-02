#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


from keras.datasets import mnist


# In[ ]:


(train_img, train_labels), (test_img, test_labels) = mnist.load_data()


# # Training Data

# In[ ]:


train_img.shape


# In[ ]:


len(train_labels)


# In[ ]:


train_labels


# In[ ]:


digit = train_img[4]

import matplotlib.pyplot as plt
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()


# # Test Data

# In[ ]:


print(test_img.shape)
print(len(test_labels))
print(test_labels)


# # The Network Architecture

# In[ ]:


from keras import models
from keras import layers


# In[ ]:


"""It consists of two Dense layers, which are densly connected neural layers.
second layer is 10-way softmax--> it will return array of 10 probability scores. """

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape = (28*28,)))
network.add(layers.Dense(10,activation='softmax'))


# To make our network ready for training, we need to pick three more things, as part of "compilation" step:
# 1. A loss function: the is how the network will be able to measure how good a job it is doing on its training data, and thus how it will be able to steer itself in the right direction. 
# 2. An optimizer: this is the mechanism through which the network will update itself based on the data it sees and its loss function. 
# 3. Metrics to monitor during training and testing. Here we will only care about accuracy
# 

# # compilation step

# In[ ]:


network.compile(optimizer='rmsprop',
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])


# # Preparing the image data
# Scale image to [0,1] interval from [0,255].

# In[ ]:


train_img = train_img.reshape((60000, 28*28))
train_img = train_img.astype('float32')/255

test_img = test_img.reshape((10000, 28*28))
test_img = test_img.astype('float32')/255


# # Preparing the labels

# In[ ]:


from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)


# # Train the network

# In[ ]:


network.fit(train_img, train_labels, epochs =5, batch_size =128)


# # Evaluate the network

# In[ ]:


test_loss, test_acc = network.evaluate(test_img, test_labels)


# In[ ]:


test_acc


# WOOOHOOOO!!! We got 97% accuracy so far. But it is bit lower than our training accuracy.
# Gap between training and testing acccuracy is "Overfitting" if trainAcc > testAcc.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




