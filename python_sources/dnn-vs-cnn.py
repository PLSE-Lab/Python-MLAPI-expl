#!/usr/bin/env python
# coding: utf-8

# # DNN vs CNN

# A simple demostration between DNN and CNN models with basic to better architecture reach a satisfying accuracy.<br>
# Both DNN and CNN will have two types of models:
# 1. Basic build-up model.
# 2. Advance model/tuned with better hyperparameters.

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
import matplotlib.pyplot as plt
import random

from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, optimizers, Sequential


# In[ ]:


# set random seed for tf and numpy
tf.random.set_seed(42)
np.random.seed(42)


# In[ ]:


# load/read data
train_path = '/kaggle/input/fashionmnist/fashion-mnist_train.csv'
test_path = '/kaggle/input/fashionmnist/fashion-mnist_test.csv'
# data = tf.keras.utils.get_file('train.csv',URL) to download file
train = pd.read_csv(train_path)
test  = pd.read_csv(test_path)
train.shape , test.shape


# In[ ]:


train.head(10)


# In[ ]:


train.info()


# In[ ]:


# Normalize the data
train_arr = np.array(train, dtype='float32')
x = train_arr[:,1:]/255
y = train_arr[:,0]

test_arr = np.array(test, dtype='float32')
x_test = test_arr[:,1:]/255
y_test = test_arr[:,0]

train_arr.shape


# In[ ]:


# reshape train data for DNN model
dnn_data = x.reshape(-1, 28, 28)
dnn_data.shape


# In[ ]:


# reshape test data for DNN model
x_test = x_test.reshape(x_test.shape[0], 28, 28)
x_test.shape


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(dnn_data,y,test_size = 0.1)


# # DNN Model 1

# In[ ]:


# build an ANN model
dnn_model = Sequential([
    layers.Flatten(input_shape = [28,28]),
    layers.Dense(300, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(150, activation = 'relu'),
    layers.Dropout(0.5),
    layers.Dense(75, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')    
])
dnn_model.summary()


# In[ ]:


optimizer = optimizers.SGD(0.03)
dnn_model.compile(optimizer= optimizer,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])


# In[ ]:


history = dnn_model.fit(x_train, y_train, epochs = 20, validation_data =(x_val, y_val))


# In[ ]:


# plot a learning curve 300,150,75,10
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()
# results 40 epochs
# loss: 0.2515 - accuracy: 0.9050 - val_loss: 0.2789 - val_accuracy: 0.9000


# In[ ]:


# tweak some hyper parameters
# this is a very legnthy process, better to prefer when training data is less or choose less hyperparameters.
# create function of model for randomized cv 

def build_model(neurons_arr = [300,200,100],learning_rate = 1e-3):
    model = Sequential()
    model.add(layers.Flatten(input_shape = [28,28]))
    for num_neuron in neurons_arr:
        model.add(layers.Dense(num_neuron ,activation = 'relu'))
    model.add(layers.Dense(10, activation = 'softmax'))
    optimizer = optimizers.SGD(learning_rate)
    model.compile(loss = 'sparse_categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
    return model


# In[ ]:


# wrap model to make sklearn like model then apply randomizedsearchcv
from sklearn.model_selection import RandomizedSearchCV
#model_wrap = tf.keras.wrappers.scikit_learn.KerasClassifier(build_model)

params = {
        'neurons_arr': [[500,300,200,100], [784, 392, 196, 98], [300,200,100]],
        'learning_rate': [1e-3, 1e-4, 3e-3, 3e-5]
}

#rnd_search  = RandomizedSearchCV(model_wrap, params, n_iter = 10, cv = 3)
#rnd_search.fit(x_train, y_train, epochs = 20, validation_data = (x_val, y_val))


# In[ ]:


#print(rnd_search.best_params_)
#print(rnd_search.best_score_)


# # Model 2

# In[ ]:


# final DNN model from randomizedcv's para score was 0.86 + some more manual trail/error tweaks
rnd_dnn_model = Sequential([
    layers.Flatten(input_shape = [28,28]),
    layers.Dense(392, activation = 'relu'),
    
    layers.Dense(196, activation = 'relu'),
    layers.Dropout(0.5),
    
    layers.Dense(98, activation = 'relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(10, activation = 'softmax')    
])
rnd_dnn_model.summary()


# In[ ]:


optimizer = optimizers.Adam(0.001)
rnd_dnn_model.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history = rnd_dnn_model.fit(x_train, y_train, epochs = 15, validation_data =(x_val, y_val))


# In[ ]:


# plot a learning curve 500,300,150
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_ylim(0,1)
plt.show()


# In[ ]:


# squeeze the last bits of accuracy with SGD at lower lr
optimizer = optimizers.SGD(0.0001)
rnd_dnn_model.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history = rnd_dnn_model.fit(x_train, y_train, epochs = 15, validation_data =(x_val, y_val))


# In[ ]:


# with 20 more epochs (5 with adam and 15 with sgd) accuracy landed around ~93.5
# predict classes using new model
p = rnd_dnn_model.predict_classes(x_test)
y = y_test
correct = np.nonzero(p==y)[0]
incorrect = np.nonzero(p!=y)[0]


# More Optimizations
# * Add more layers + dropouts
# * Neurons with less size gaps
# * More training(epochs) with SGD with lower lr at beginning and very low near end
# * LeakyReLU for gradient exploring/vanishing 
# * Maybe more BatchNormalization(in my case the results were better off without them)
# * Let me Know What you Know!!

# In[ ]:


print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])


# # Create a CNN

# In[ ]:


# reshape the data for cnn
x = train_arr[:,1:]/255
y = train_arr[:,0]
cnn_data = x.reshape(-1, 28,28,1)
cnn_data.shape, y.shape


# In[ ]:


# reshape cnn test data
x_test = test_arr[:,1:]/255
x_test = x_test.reshape(x_test.shape[0], 28, 28,1)
x_test.shape


# In[ ]:


x_train, x_val, y_train, y_val = train_test_split(cnn_data, y, test_size = 0.1)


# # Model 1

# In[ ]:


cnn_model1 = Sequential([
    layers.Conv2D(64, kernel_size = (3,3), padding ='same', activation = 'relu', input_shape = [28,28,1]),
    layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid'),
    layers.Conv2D(128, kernel_size = (3,3)),
    layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'valid'),
    layers.Flatten(),
    layers.Dense(64, activation = 'relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation = 'relu'),
    layers.Dense(10, activation = 'softmax')
])
cnn_model1.summary()


# In[ ]:


cnn_model1.compile(optimizer = 'Adam',loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history = cnn_model1.fit(x_train, y_train, epochs = 10, validation_data = (x_val, y_val))


# In[ ]:


# more epochs are leading to overfit the model
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_xlim(0,10)
plt.gca().set_ylim(0,1)
plt.show()


# # Model 2

# In[ ]:


from functools import partial
LeakyReLU = layers.LeakyReLU(alpha=0.1)
# create a default layer to save repetations
DefConv2D = partial(layers.Conv2D, kernel_size = (3,3),activation = LeakyReLU, padding = 'valid')

cnn_model = Sequential([
    layers.Conv2D(64, kernel_size = (3,3), padding ='same', input_shape = [28,28,1], activation = LeakyReLU),
    layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'same'),
    layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),

    DefConv2D(filters = 128),
    layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    
    DefConv2D(filters = 128),
    layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'same'),
    layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    
    DefConv2D(filters = 256),
    layers.MaxPool2D(pool_size = 2, strides = 2, padding = 'same'),
    layers.BatchNormalization(momentum=0.9, epsilon=1e-5, gamma_initializer="uniform"),
    layers.Dropout(0.5),
    
    layers.Flatten(),
    layers.Dense(64, activation = LeakyReLU),
    layers.Dropout(0.5),
    layers.Dense(10, activation = 'softmax')
])
cnn_model.summary()


# In[ ]:


optimizer = optimizers.Adam(0.003)
cnn_model.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history = cnn_model.fit(x_train, y_train, epochs = 20, validation_data = (x_val, y_val), verbose = 0)


# In[ ]:


# more epochs are leading to overfit the model
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.gca().set_xlim(0,10)
plt.gca().set_ylim(0,1)
plt.show()
# ignore this comments
# changes !!55th epoch
# conv 32,64,128(5,5)   dense layer 128(dropout 0.5),64(dropout 0.5),10   
# Results loss 071  acc 97   val_loss 33  val_acc 92(9215 to 9290)    found train loss is way low, almost hit 93!!


# In[ ]:


optimizer = optimizers.Adam(0.0003)
cnn_model.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history = cnn_model.fit(x_train, y_train, epochs = 10, validation_data = (x_val, y_val))


# In[ ]:


# squeeze the very last bit acc with ~very little val_acc
optimizer = optimizers.SGD(0.0003)
cnn_model.compile(optimizer = optimizer,loss = 'sparse_categorical_crossentropy',metrics = ['accuracy'])
history = cnn_model.fit(x_train, y_train, epochs = 10, validation_data = (x_val, y_val))


# In[ ]:


# this results around acc ~98 and val ~93.5
# last successful run: loss: 0.0501 - accuracy: 0.9815 - val_loss: 0.2669 - val_accuracy: 0.9363
p = cnn_model.predict_classes(x_test)
y = y_test
correct = np.nonzero(p==y)[0]
incorrect = np.nonzero(p!=y)[0]


# In[ ]:


print("Correct predicted classes:",correct.shape[0])
print("Incorrect predicted classes:",incorrect.shape[0])


# More Optimizations
# * Data augumentation will boost upto 1+ maybe 2 accuracy.
# * More layers + Normalization + dropouts, More dense layers.
# * Let me Know What you Know!!

# # Summary

# **Models parameter and results.**<br>
# DNN model 2.<br>
# Parameters: 405,436<br>
# Results: loss: 0.2148 - accuracy: 0.9218 - val_loss: 0.2932 - val_accuracy: 0.8983 (almost ~93 acc)<br>
# 
# CNN model 2.<br>
# Parameters: 585,802<br>
# Results: loss: 0.0501 - accuracy: 0.9815 - val_loss: 0.2669 - val_accuracy: 0.9363 (almost ~98 acc)<br>
# <br>
# The clear difference with a normal CNN is the parameters, they are less even with bigger model/complex model and we achieved higher accuracy(~5+) with less epochs(though at last I couldn't resist so did more) than a tuned DNN would need. <br>
# 
# With Adam at the beginning we can converge fast, later with SGD lower lr we can squeeze the remaining accuracy. Cheers !!!
