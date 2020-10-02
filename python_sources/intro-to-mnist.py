#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


# ## Reading the dataset

# In[ ]:


train = pd.read_csv('../input/train.csv')
train.head()


# In[ ]:


test = pd.read_csv('../input/test.csv')
test.head()


# In[ ]:


print("Number of training examples :", len(train))
print("Number of test examples :", len(test))


# ## Learning about the dataset

# In[ ]:


train_labels = train['label']
train_data = train.drop(['label'], axis=1)
test_data = test

train_data.shape, test_data.shape


# In[ ]:


trainDistribution = pd.DataFrame(train_labels.value_counts(sort=False))
trainDistribution.columns = ['MNIST Train Examples Count']
trainDistribution


# In[ ]:


plt.figure()
sample_image = np.array(train_data.iloc[3]).reshape([28,28])
plt.imshow(sample_image)
plt.grid(True)


# In[ ]:


train_data = np.array(train_data) / 255.0
train_labels = np.array(train_labels)
test_data  = np.array(test_data) / 255.0


# In[ ]:


plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(train_data[i].reshape([28,28]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[i])


# ## Implementing a Neural Network
# #### Predicting labels for the test data and testing accuracy 

# ### Shallow Neural Network

# In[ ]:


nn_model1 = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[ ]:


nn_model1.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


nn_model1.fit(train_data, train_labels, epochs=3)


# ### Deeper Neural Networks

# In[ ]:


nn_model2 = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu),
     keras.layers.Dense(64, activation=tf.nn.relu),
     keras.layers.Dense(32, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])


# In[ ]:


nn_model2.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


nn_model2.fit(train_data, train_labels, epochs=5)


# ## Making Predictions 

# ### Model I

# In[ ]:


predictions_on_model1 = nn_model1.predict(test_data)
predictions_on_model1 = np.argmax(predictions_on_model1, axis=1)
predictions_on_model1


# In[ ]:


plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(test_data[i].reshape([28,28]), cmap=plt.cm.binary)
    plt.xlabel(predictions_on_model1[i])


# ### Model II

# In[ ]:


predictions_on_model2 = nn_model2.predict(test_data)
predictions_on_model2 = np.argmax(predictions_on_model2, axis=1)
predictions_on_model2


# In[ ]:


plt.figure(figsize=(8,8))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(True)
    plt.imshow(test_data[i].reshape([28,28]), cmap=plt.cm.binary)
    plt.xlabel(predictions_on_model2[i])


# In[ ]:


predictions = pd.Series(predictions_on_model2)
predictions = predictions.to_frame(name='Label')
predictions.set_index(np.arange(1, len(predictions) + 1), inplace=True)
predictions.to_csv('submission.csv',index_label='ImageId')
predictions

