#!/usr/bin/env python
# coding: utf-8

# In[153]:


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


# In[154]:


# import
import tensorflow as tf
from tensorflow.keras import layers


# **DATA**
# 
# Load data and preprocess it, so it may be feed into the neural network.

# In[155]:


train_raw = pd.read_csv('../input/train.csv')
test_raw = pd.read_csv('../input/test.csv')

train_label = train_raw['label']
train_data = train_raw.drop('label', axis = 1)


# Load the data, which will be seperated into training and testing. Both training and testing will be further divided into labels and data.
# 
# This data will not require cleaning or verification. It comes from a trusted set.

# In[156]:


# normalize the data
train_data = train_data / 255
test_data = test_raw / 255

# seperate training examples for testing
held_data = train_data[-2000:]
held_label = train_label[-2000:]
train_data = train_data[:-2000]
train_label = train_label[:-2000]

# convert data from a data frame to numpy array and reshape
train_label = train_label.values
held_label = held_label.values
train_data = train_data.values.reshape([-1, 28, 28, 1])
test_data = test_data.values.reshape([-1, 28, 28, 1])
held_data = held_data.values.reshape([-1, 28, 28, 1])


# Normalize the data, which means use values of 0 to 1. This is greyscale, so it is a one dimension array with values of 0 to 255. Divide each value by 255 to achive this.
# 
# Seperate a section of the train examples. These will not be used for training. They will become a test example. A test example lets the model be evaulated, without submitting. This is especially helpful to measure if the trained model is overfitted.
# 
# Convert the all the data and labels into a numpy array. Pandas dataframes are not useable with Tensorflow. Use '.values' to get the updated type. The arrays given to the neural network require a shape of [28, 28, 1]. The data is of a 28 by 28 greyscale picture. If it were color, [28, 28, 3] would be used instead.

# **MODEL**

# In[157]:


model = tf.keras.models.Sequential()


# Use a sequential model, so data flows from one layer to the next automatically.

# In[158]:


model.add(
tf.keras.layers.Conv2D(
    filters = 40,
    kernel_size = [5,5],
    padding = 'same',
    activation = 'relu',
    input_shape = (28,28,1)
))

model.add(
tf.keras.layers.MaxPool2D(
    pool_size = [2,2],
    strides = 2
))

model.add(
tf.keras.layers.Conv2D(
    filters = 40,
    kernel_size = [5,5],
    padding = 'same',
    activation = 'relu',
    input_shape = (28,28,1)
))

model.add(
tf.keras.layers.MaxPool2D(
    pool_size = [2,2],
    strides = 2
))

model.add(
tf.keras.layers.Conv2D(
    filters = 80,
    kernel_size = [3,3],
    padding = 'same',
    activation = 'relu',
    input_shape = (28,28,1)
))

model.add(
tf.keras.layers.MaxPool2D(
    pool_size = [2,2],
    strides = 2
))

model.add(
tf.keras.layers.Dropout(0.10)
)


# Use 2D convulational layer. These are great for images, since they 'scan' the whole image. Padding 'same' adds zeros as needed. Activation is 'relu', as it speeds ups learning. Filters and kernel size were chosen somewhat arbitrarily. 3 layers were used in total, to ensure there is enough 'depth'.
# 
# 
# After each convulational layer, include a max pooling 2d layer. These reduce the spatial dimensions, which improves the efficiency of the model. The pool size and strides are equal, preventing overlap.

# In[159]:


model.add(
tf.keras.layers.Flatten()
)

model.add(
tf.keras.layers.Dense(256, activation='relu')
)

model.add(
tf.keras.layers.Dropout(0.10)
)


# The network needs to switch to a fully connected layer, so it can have classified output. The output for a CNN layer must be converted into a linear shape: flatten. The flatten output becomes the input for a dense layer.
# 
# A dropout layer is included to help prevent overfitting.

# In[160]:


model.add(
tf.keras.layers.Dense(10, activation='softmax')
)


# Create an output layer that has 10 nodes. That is one node for each possible outcome. Use 'softmax' so the outputs represent probability.

# **TRAINING**

# In[161]:


model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy']
)
history = model.fit(train_data, train_label, epochs=15)


# Finalize the model by 'compile'. 'adam' was chosen as an optimizer, since it is considered fast and effective.
# 
# Accuracy was chosen as metric, because it is the easiest to understand. Loss would have been an effective chosen.

# In[162]:


import matplotlib.pyplot as plt

plt.plot(history.history['acc'])
plt.title('accuracy over epoch')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.show()


# **EVALUATE THE MODEL**

# In[163]:


model.evaluate(held_data, held_label)


# The original model had an accuracy difference of about 0.6% to 0.7%. This was a measure of the training model being overfitted.
# 
# An additional dropout layer was added, at the end of the convolutional layers. Training epochs was lowered from 20 to 15. The accuracy difference became about 0.4% to 0.5%.

# In[164]:


predict = model.predict_classes(test_data)
submit = pd.DataFrame(
    {"ImageId": list(range(1,len(predict)+1)),"Label": predict})

submit.to_csv("../mnist_output.csv",index=False)


# Make predictions from the test data. Convert the data into a dataframe, include 'ImageId. Save the data frame as a CSV file.

# In[165]:


print(os.listdir("../"))

