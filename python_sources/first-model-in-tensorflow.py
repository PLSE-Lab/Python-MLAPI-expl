#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import math
import cv2
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train = pd.read_json('../input/train.json')
train.keys()


# In[ ]:


train.head()


# In[ ]:


test = pd.read_json('../input/test.json')
test.keys()


# **Let's explore some images.**

# In[ ]:


icebergs = train[train.is_iceberg == 1]
print(type(icebergs))
icebergs = icebergs.sample(n=18, random_state = 456)

ships = train[train.is_iceberg == 0]
ships = ships.sample(n=18, random_state = 456)


# In[ ]:


fig = plt.figure(1, figsize = (15, 15))
for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    arr = np.reshape(np.array(icebergs.iloc[i, 0]), (75, 75)) +           np.reshape(np.array(icebergs.iloc[i, 1]), (75, 75))
    ax.set_title('Incidence Angle: {}, label: {}'.format(icebergs.iloc[i, 3], icebergs.iloc[i, 4]))
    ax.imshow(arr, cmap = 'winter')
    
plt.show()


# In[ ]:


img = icebergs.iloc[0, 0]
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])


#blurred = cv2.medianBlur(x_band1, 5)


# In[ ]:


blurred = cv2.medianBlur(x_band1[0], 5)


# In[ ]:


fig = plt.figure(1,figsize=(15,15))
for i in range(9):
    ax = fig.add_subplot(3, 3, i+1)
    arr = np.reshape(np.array(ships.iloc[i, 0]), (75, 75)) #+ \
          #np.reshape(np.array(ships.iloc[i, 1]), (75, 75))
    ax.set_title('Incidence Angle: {}, label: {}'.format(ships.iloc[i, 3], ships.iloc[i, 4]))
    ax.imshow(arr, cmap='winter')
    
plt.show()


# I follow Miha Skalic guide to process data for tensorflow model. 

# In[ ]:


x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in train["band_2"]])
X_train = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
y_train = np.array(train["is_iceberg"], dtype = np.float32).reshape(train.shape[0],1)
print("Xtrain:", X_train.shape)

# Test data
x_band1 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_1"]])
x_band2 = np.array([np.array(band).astype(np.float32).reshape(75, 75) for band in test["band_2"]])
X_test = np.concatenate([x_band1[:, :, :, np.newaxis], x_band2[:, :, :, np.newaxis]], axis=-1)
print("Xtest:", X_test.shape)


# In[ ]:


y_train.shape


# In[ ]:


nans = lambda df: df[df.isnull().any(axis=1)]

nans(train).shape


# In[ ]:


learning_rate = 0.01
epochs = 100
batch_size = 128


# In[ ]:


x = tf.placeholder(tf.float32, [None, 75, 75, 2])
y = tf.placeholder(tf.float32, [None, 1])


# In[ ]:


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 3, 3, 1], 
                         strides = [1, 2, 2, 1], padding = 'SAME')
def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)
def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# In[ ]:


#x_ = tf.reshape(x, [-1, 75, 75, 2])

conv1 = conv_layer(x, shape = [5, 5, 2, 32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape = [5, 5, 32, 64])
conv2_pool = max_pool_2x2(conv2)


# In[ ]:


print(conv1.shape)
print(conv1_pool.shape)
print(conv2.shape)
print(conv2_pool.shape)


# In[ ]:


conv2_flat = tf.reshape(conv2_pool, shape = [-1, 19*19*64])
full_2 = tf.nn.relu(full_layer(conv2_flat, 1024))


# In[ ]:


keep_prob = tf.placeholder(tf.float32)
full2_drop = tf.nn.dropout(full_2, keep_prob = keep_prob)

y_conv = full_layer(full2_drop, 1)

log_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
    logits = y_conv,
    labels =   y))


# In[ ]:


train_step = tf.train.AdamOptimizer(1e-4).minimize(log_loss)
correct_prediction = tf.equal(tf.argmax(y_conv, axis = 1), 
                              tf.argmax(y, axis = 1))


# In[ ]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[ ]:


y_train[0:10].dtype


# In[ ]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(epochs):
        for j in range(math.ceil(train.shape[0]/batch_size)):
            if j == 12:
                training = X_train[(train.shape[0] - 12*batch_size):train.shape[0]]
                y_training = y_train[(train.shape[0] - 12*batch_size):train.shape[0]]
            else:
                training = X_train[(j*batch_size):((j+1)*batch_size)]
                y_training = y_train[(j*batch_size):((j+1)*batch_size)]
            if i % 5 == 0:
                 #print("step {}, training accuracy {}". format(i, 
                 #                                        train_accuracy))
                print(sess.run(log_loss, feed_dict = {x: training,
                                         y: y_training,
                                             keep_prob: 1.0}))
            sess.run(train_step, feed_dict = {x: training,
                                         y: y_training,
                                             keep_prob: 0.5})


# In[ ]:




