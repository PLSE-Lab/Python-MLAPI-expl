#!/usr/bin/env python
# coding: utf-8

# # Basic DNN
# 
# We going to use Basic DNN with Tensorflow, And explain every step in the way (:
# Hope it might be useful
# 
# Our task is to recognise handwritten digits images and identify there number (Label)
# 
# Steps:
# 1. Collecting data
# 2. Preparing the data
# 3. Training a model
# 4. Evaluate the model
# 5. Improving the performance

# # 1. Collecting data
# * And needed librarys

# In[43]:


# basic lib
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# tensorflow
import tensorflow as tf

# keras 
from keras.models import Sequential
from keras.layers import Dense,Flatten

# train test split
from sklearn.model_selection import train_test_split

#ignore warning messages 
import warnings
warnings.filterwarnings('ignore') 


# In[44]:


# Get the train and test data 
train = pd.read_csv("../input/train.csv")

# check the columns in the data
print(train.columns)


# In[45]:


# Splite data the X - Our data , and y - the prdict label
X = train.drop('label',axis = 1)
y = train['label']


# In[46]:


# Check the data shape (columns , rows)
print(X.shape)

print("We have 42000 row - images , and 784 columns - pixels")
print("Every image has 784 pixels")
print("\n784 = 28X28")

# let look at the data
X.head()


# Let look at the data images

# In[47]:


# let look at the data 

# print 1 image by row number
row_number = 5
plt.imshow(X.iloc[row_number].values.reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()

# print 4X4 first images
# plt.figure(figsize = (12,10))
# row, colums = 4, 4
# for i in range(16):  
#     plt.subplot(colums, row, i+1)
#     plt.imshow(X.iloc[i].values.reshape(28,28),interpolation='nearest', cmap='Greys')
# plt.show()


# # 2. Preparing the data

# In[48]:


# splite the data
X_train, X_test, y_train, y_test = train_test_split(X,y)

# scale data
X_train = X_train.apply(lambda X: X/255)
X_test = X_test.apply(lambda X: X/255)


# In[49]:


print("Data after scaler")
row_number = 2
plt.imshow(X_train.iloc[row_number].values.reshape(28,28),interpolation='nearest', cmap='Greys')
plt.show()


# # 3. Training a model

# In[50]:


# reshape
X_train = tf.reshape(X_train, [-1, 28, 28,1])
X_test = tf.reshape(X_test, [-1, 28, 28,1])


# In[ ]:


input_shape = (28,28)
output_count = 10

model = Sequential([
    Flatten(input_shape=input_shape),
    Dense(128, activation=tf.nn.relu),
    Dense(output_count, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, steps_per_epoch = 32,epochs=15, validation_data = (X_test, y_test), validation_steps = 10)


# In[ ]:


test_loss, test_acc = model.evaluate(X_test, y_test, steps = 10)
print("loss",test_loss)
print("acc",test_acc)


# How to know if I over fitting?

# In[52]:


input_shape = (28,28)
output_count = 10


model = Sequential([
    Flatten(input_shape=input_shape),
    Dense(128, activation=tf.nn.relu),
    Dense(64, activation=tf.nn.relu),
    Dense(32, activation=tf.nn.relu),
    Dense(16, activation=tf.nn.relu),
    Dense(output_count, activation=tf.nn.softmax)
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, steps_per_epoch = 32,epochs=30, validation_data = (X_test, y_test), validation_steps = 10)


# Train on 31500 samples, validate on 10500 samples
# Epoch 1/30
# 32/32 [==============================] - 43s 1s/step - loss: 1.5606 - acc: 0.4769 - val_loss: 0.8248 - val_acc: 0.7503
# Epoch 2/30
# 32/32 [==============================] - 8s 259ms/step - loss: 0.4741 - acc: 0.8670 - val_loss: 0.3032 - val_acc: 0.9129
# Epoch 3/30
# 32/32 [==============================] - 8s 263ms/step - loss: 0.2204 - acc: 0.9364 - val_loss: 0.2067 - val_acc: 0.9399
# Epoch 4/30
# 32/32 [==============================] - 8s 253ms/step - loss: 0.1407 - acc: 0.9596 - val_loss: 0.1661 - val_acc: 0.9503
# Epoch 5/30
# 32/32 [==============================] - 8s 252ms/step - loss: 0.0961 - acc: 0.9734 - val_loss: 0.1456 - val_acc: 0.9560
# Epoch 6/30
# 32/32 [==============================] - 8s 251ms/step - loss: 0.0676 - acc: 0.9820 - val_loss: 0.1366 - val_acc: 0.9598
# Epoch 7/30
# 32/32 [==============================] - 8s 251ms/step - loss: 0.0478 - acc: 0.9882 - val_loss: 0.1336 - val_acc: 0.9619
# Epoch 8/30
# 32/32 [==============================] - 8s 256ms/step - loss: 0.0336 - acc: 0.9928 - val_loss: 0.1334 - val_acc: 0.9641
# Epoch 9/30
# 32/32 [==============================] - 8s 247ms/step - loss: 0.0233 - acc: 0.9959 - val_loss: 0.1363 - val_acc: 0.9641
# Epoch 10/30
# 32/32 [==============================] - 8s 249ms/step - loss: 0.0161 - acc: 0.9977 - val_loss: 0.1413 - val_acc: 0.9642
# Epoch 11/30
# 32/32 [==============================] - 8s 255ms/step - loss: 0.0111 - acc: 0.9989 - val_loss: 0.1486 - val_acc: 0.9641
# Epoch 12/30
# 32/32 [==============================] - 8s 241ms/step - loss: 0.0075 - acc: 0.9996 - val_loss: 0.1571 - val_acc: 0.9640
# Epoch 13/30
# 32/32 [==============================] - 8s 247ms/step - loss: 0.0051 - acc: 0.9998 - val_loss: 0.1651 - val_acc: 0.9636
# Epoch 14/30
# 32/32 [==============================] - 8s 238ms/step - loss: 0.0037 - acc: 0.9999 - val_loss: 0.1720 - val_acc: 0.9639
# Epoch 15/30
# 32/32 [==============================] - 8s 240ms/step - loss: 0.0027 - acc: 1.0000 - val_loss: 0.1779 - val_acc: 0.9641
# Epoch 16/30
# 32/32 [==============================] - 8s 237ms/step - loss: 0.0021 - acc: 1.0000 - val_loss: 0.1833 - val_acc: 0.9639
# Epoch 17/30
# 32/32 [==============================] - 8s 237ms/step - loss: 0.0017 - acc: 1.0000 - val_loss: 0.1881 - val_acc: 0.9639
# 
#                 
# We can see the val_acc is getting and the acc is still getting up
