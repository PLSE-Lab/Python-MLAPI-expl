#!/usr/bin/env python
# coding: utf-8

# Copyright 2016 Google Inc. All Rights Reserved.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 
# --------------------------------------

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
import tensorflow as tf
tf.enable_eager_execution()
print(tf.__version__)

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[2]:


get_ipython().system('python -V')


# In[3]:


data_train_file = "../input/fashion-mnist_train.csv"
data_test_file = "../input/fashion-mnist_test.csv"

df_train = pd.read_csv(data_train_file)
df_test = pd.read_csv(data_test_file)


# In[4]:


df_train.head()


# ## Preprocessing the data

# In[5]:


def get_features_labels(df):
    # Select all columns but the first
    features = df.values[:, 1:]/255
    # The first column is the label. Conveniently called 'label'
    labels = df['label'].values
    return features, labels


# In[6]:


train_features, train_labels = get_features_labels(df_train)
test_features, test_labels = get_features_labels(df_test)


# In[7]:


print(train_features.shape)
print(train_labels.shape)


# In[8]:


# take a peak at some values in an image
train_features[20, 300:320]


# In[9]:


example_index = 221
plt.figure()
_ = plt.imshow(np.reshape(train_features[example_index, :], (28,28)), 'gray')


# ## Convert labels to one-hot encoding

# In[10]:


train_labels.shape


# In[11]:


train_labels[example_index]


# In[12]:


train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)


# In[13]:


train_labels.shape


# In[14]:


train_labels[example_index]


# ## Create the model

# In[16]:


model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(30, activation=tf.nn.relu, input_shape=(784,)))
model.add(tf.keras.layers.Dense(20, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


# Create a TensorFlow optimizer, rather than using the Keras version
# This is currently necessary when working in eager mode
optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

# We will now compile and print out a summary of our model
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

model.summary()


# ## Training with Keras

# In[ ]:


EPOCHS=2
BATCH_SIZE=128


# In[ ]:


model.fit(train_features, train_labels, epochs=EPOCHS, batch_size=BATCH_SIZE)


# ## Control Training using datasets

# In[ ]:


def make_generator(features, labels):

    def _generator():
        for feature, label in zip(features, labels):
            yield feature, label

    return _generator


# In[ ]:


dataset = tf.data.Dataset.from_generator(
      make_generator(train_features, train_labels),
      (tf.float32, tf.float32))

dataset = dataset.shuffle(1000)
dataset = dataset.batch(BATCH_SIZE)


# In[ ]:


for epoch in range(EPOCHS):
    step=0
    for features, labels in dataset:
        train_loss, train_accuracy = model.train_on_batch(features, labels)
        if step % 100 == 0:
            print('Step #%3d: Loss: %.6f\tAccuracy: %.6f' % (step + 1, train_loss, train_accuracy))
        step +=1
        
    # Here you can gather any metrics or adjust your training parameters
    print('Epoch #%d\t Loss: %.6f\tAccuracy: %.6f' % (epoch + 1, train_loss, train_accuracy))
  


# In[ ]:


test_loss, test_acc = model.evaluate(test_features, test_labels)


# In[ ]:


print('test_acc:', test_acc)


# In[ ]:




