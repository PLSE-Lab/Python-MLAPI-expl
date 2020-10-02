#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import math

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Start with loading the training and testing data. 
# We are not seperating the Training model in training and testing set

# In[ ]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# Let us begin with the visualization of the data given
# The first row has the labels and other values are the flattened values of the pixels of the image

# In[ ]:


train.head()


# The test data is same as train data with just one column left i.e of label

# In[1]:


test.head()


# Let us seperate the training model into labels and pixels values

# In[ ]:


train_label = train.pop('label')


# Normalising the given data as each data has the greyscale value of the image

# In[ ]:


train = train/255
test = test/255


# preparing the 3 layer neural network to solve the problem

# In[ ]:


model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(400, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(100, activation = tf.nn.relu),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation = tf.nn.softmax)
])


# Compiling the model

# In[ ]:


model.compile(optimizer = 'adam',
loss = 'sparse_categorical_crossentropy',
metrics = ['accuracy'])


# In[ ]:


batch_size = 100
training_size = len(train_label)


# starting the learning to find the weights of the layers

# In[ ]:


import math
model.fit(train.values, train_label.values, epochs =5, steps_per_epoch=math.ceil(training_size/batch_size))


# Predicting the model on the given test data

# In[ ]:


test_label = model.predict(test.values)
label = np.argmax(test_label, axis =1)
label.shape


# > **Save the data**

# In[ ]:


sample = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sample.head()


# In[ ]:


index = list(range(1, label.shape[0]+1))


# In[ ]:


df = pd.DataFrame({'ImageId': index, 'Label': label})
df.head()


# In[ ]:


df.to_csv('predict.csv', index=False)

