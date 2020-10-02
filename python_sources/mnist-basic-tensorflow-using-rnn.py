#!/usr/bin/env python
# coding: utf-8

# # MNIST - TensorFlow Basics using CNN
# 
# The objective of this notebook is to build a basic model for MNIST dataset using TensorFlow Recurrent Neural Network(CNN). This code is from [pythonprogramming.net](https://pythonprogramming.net/recurrent-neural-network-deep-learning-python-tensorflow-keras/)

# In[ ]:


import numpy as np

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM


# ## Loading the MNIST Data

# In[ ]:


def load_data():
    with np.load("../input/mnist.npz") as f:
        x_train, y_train = f['x_train'], f['y_train']
        x_test, y_test = f['x_test'], f['y_test']
    return (x_train, y_train), (x_test, y_test)

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape)
print(x_train[0].shape)


# ## Normalize the data

# In[ ]:


x_train =  x_train/255.0
x_test = x_test/255.0


# ## Build the Model

# In[ ]:


model = Sequential()

# Layers
model.add(CuDNNLSTM(128, input_shape=(x_train.shape[1:]), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(10, activation='softmax'))

# Optimizer
opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

#Compile
model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

#Fit
model.fit(x_train, y_train, epochs=3, validation_data=(x_test, y_test))



# In[ ]:




