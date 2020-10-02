#!/usr/bin/env python
# coding: utf-8

# The main idea of this passage is we should **never to use the SGD optimizer**.
# I used to love using the SGD optimizer until a problem occurred as below.
# I had built a simple model to fit the cube function, but I got a nan loss. 

# In[8]:


import random
import numpy as np
from keras import metrics, optimizers, Sequential
from keras.layers import Dense
from keras.losses import mean_squared_error
from sklearn.model_selection import train_test_split


def cube(x):
    return x ** 3 + random.random()


x = np.linspace(1.0, 5.0, num=1000).reshape((1000, 1))
cube_fun = np.vectorize(cube)
y = cube_fun(x)
BATCH_SIZE = 128
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42,shuffle=True)


def alexnet_model():
    alexnet = Sequential([
        Dense(10, activation='relu', input_dim=1),
        Dense(10),
        Dense(1)])
    return alexnet


def train_model(model):
    model.compile(optimizer=optimizers.SGD(), loss=mean_squared_error)
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=100,
              verbose=1,
              validation_data=(X_test, y_test), callbacks=[])


model = alexnet_model()
train_model(model)


# But when I use the RMSProp loss function, it will get the right result.

# In[9]:


def train_model(model):
    model.compile(optimizer=optimizers.RMSprop(), loss=mean_squared_error)
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=100,
              verbose=1,
              validation_data=(X_test, y_test), callbacks=[])
model = alexnet_model()
train_model(model)


# Why did this error occur? After read some materials, I find that the adaptive learning rate is much helpful in this problem.
# In this simple problem, if we use a large learning rate, the model parameters will change dramatically, which make a considerable loss value. If the loss value is too high, we will get a NAN error.
# So, after knowing that, let us try a model with a small fixed learning rate.

# In[10]:


def train_model(model):
    model.compile(optimizer=optimizers.SGD(lr=0.00001), loss=mean_squared_error)
    model.fit(X_train, y_train,
              batch_size=BATCH_SIZE,
              epochs=100,
              verbose=1,
              validation_data=(X_test, y_test), callbacks=[])
model = alexnet_model()
train_model(model)


# After changing the learning rate. We will get the right result.
# However, it is too dull to tune the learning rate to an appropriate value. So, in the neural network training, remember a rule-"never to use the SGD optimizer, although SGD is easy to understand".
