#!/usr/bin/env python
# coding: utf-8

# I add here 3 functions that you can predict with Neural Network.
# 
# I also add my solutions, so you can look how I solve it. 
# 
# The funcs is 2 dimensions, so you can visual your results.
# 
# You can use "show_my_prdict" def to visual your results
# 
# 
# I very recommend that you try to solve each one with different approach, 
# like diffrent  [optimizers](https://keras.io/optimizers/)
# or diffrent count of [layers](https://keras.io/layers/core/).
# 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import random 

# keras 
from keras.models import Sequential
from keras.layers import Dense, Dropout
import tensorflow as tf
import keras

my_seed = 512
np.random.seed(my_seed)
random.seed(my_seed)
tf.set_random_seed(my_seed)


# In[ ]:


def show_my_prdict(model, X, y):
    X_plot = np.arange(0, 30, 0.001).reshape(-1, 1)
    y_predict = model.predict(X_plot) 

    plt.scatter(X, y, s = 0.2)
    plt.plot(X_plot, y_predict, c= "red")
    plt.show()


# ### The functions: **Tanh(X) + 2COS(X) + X / sqrt(X)**

# In[ ]:


# Create the func
X = 30*np.random.random(1000)
y_true = (np.tanh(X) + 2*np.cos(X)+X)/np.sqrt(X)
y = y_true +np.random.normal(0, 0.2, 1000)

# plot 
plt.scatter(X, y, s = 0.3)
plt.show()


# My solution:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)

ANN = Sequential([
    Dense(1016, input_dim=1, activation='relu'),
    Dense(508, activation='relu'),
    Dense(254,  activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

opt = keras.optimizers.RMSprop(lr=0.003)
ANN.compile(loss='mean_squared_error', optimizer= opt)
history = ANN.fit(X_train, y_train, steps_per_epoch = 100,epochs=200, validation_data = (X_test, y_test), validation_steps = 100)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


show_my_prdict(ANN,X,y)


# ### The functions: **Sin(X) + Cos(X)** 

# In[ ]:


# Create the func
X = 30*np.random.random(1000)
y_true = np.sin(X) + np.cos(X)
y = y_true +np.random.normal(0, 0.2, 1000)

# plot 
plt.scatter(X, y, s = 0.3)
plt.show()


# My solution:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)

ANN = Sequential([
    Dense(508, input_dim=1, activation='relu'),
    Dense(254,  activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16,  activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, kernel_initializer='normal')
])
ANN.compile(loss='mean_squared_error', optimizer= 'adam')
history = ANN.fit(X_train, y_train, steps_per_epoch = 100,epochs=20, validation_data = (X_test, y_test), validation_steps = 100)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


show_my_prdict(ANN,X,y)


# ### The functions: **X2Cos(X)** 

# In[ ]:


# Create the func
X = 30*np.random.random(1000)
y_true = X*2*np.cos(X)
y = y_true +np.random.normal(0, 1, 1000)

# plot 
plt.scatter(X, y, s = 0.3)
plt.show()


# My solution:

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y)

ANN = Sequential([
    Dense(508, input_dim=1, activation='relu'),
    Dense(254,  activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16,  activation='relu'),
    Dense(8, activation='relu'),
    Dense(1, kernel_initializer='normal')
])
ANN.compile(loss='mean_squared_error', optimizer= 'adam')
history = ANN.fit(X_train, y_train, steps_per_epoch = 100,epochs=12, validation_data = (X_test, y_test), validation_steps = 100)


# In[ ]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# In[ ]:


show_my_prdict(ANN,X,y)

