#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras import optimizers
import math
import seaborn as sns


# ## The function that neural net try to approximate

# In[ ]:


x = np.linspace(-10, 10, 100)
y = np.cos(x)
plt.plot(x,y)
plt.show()


# ## Multi-layer perceptron

# In[ ]:


model = Sequential()
model.add(Dense(30, input_shape=(1,), activation='relu'))
model.add(Dense(30, input_shape=(1,), activation='relu'))
model.add(Dense(30, input_shape=(1,), activation='relu'))
model.add(Dense(1, activation='linear'))
ada = optimizers.Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=ada, metrics=['mean_squared_error'])
model.fit(x, y, epochs=100, verbose=0)


# Multi-layer perceptron almost perfetly predicted data that they already learned.

# In[ ]:


x = np.linspace(-10, 10, 100)
y = np.cos(x)
prediction = model.predict(x)
plt.plot(x, y) 
plt.plot(x, prediction)
plt.show()


# ## Prediction on unseen data
# But the problem was they can't generalize on unseen data

# In[ ]:


x = np.linspace(-20, 20, 200)
y = np.cos(x)
prediction = model.predict(x)
plt.plot(x, y)
plt.plot(x, prediction)
plt.show()


# ## Regularize model
# I applied L2 Regularization method for generalizaing model, but It doesn't help. It just decayed weights, but was not able to learn periodic pattern.

# In[ ]:


x = np.linspace(-10, 10, 100)
y = np.cos(x)

from keras import regularizers
model = Sequential()
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(30, input_shape=(1,), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dense(1, activation='linear'))
ada = optimizers.Adam(lr=0.01)
model.compile(loss='mean_squared_error', optimizer=ada, metrics=['mean_squared_error'])
model.fit(x, y, epochs=100, verbose=0)


# In[ ]:


x = np.linspace(-20, 20, 200)
y = np.cos(x)
prediction = model.predict(x)
plt.plot(x, y)
plt.plot(x, prediction)
plt.show()

