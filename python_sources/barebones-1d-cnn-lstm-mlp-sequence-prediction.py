#!/usr/bin/env python
# coding: utf-8

# <h1>Notebook Content</h1>
# 1. [Import Packages](#1)
# 1. [Helper Functions](#2)
# 1. [Input](#3)
# 1. [Model](#4)
# 1. [Prediction](#5)
# 1. [Complete Figure](#6)

# <h1 id="1">1. Import Packages</h1>
# Importing all necessary and useful packages in single cell.

# In[ ]:


import numpy as np
import keras
import tensorflow as tf
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras_tqdm import TQDMNotebookCallback
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
import random
from random import randint


# <h1 id="2">2. Helper Functions</h1>
# Defining Some helper functions which we will need later in code

# In[ ]:


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

def plot_multi_graph(xAxis,yAxes,title='',xAxisLabel='number',yAxisLabel='Y'):
    linestyles = ['-', '--', '-.', ':']
    plt.figure()
    plt.title(title)
    plt.xlabel(xAxisLabel)
    plt.ylabel(yAxisLabel)
    for key, value in yAxes.items():
        plt.plot(xAxis, np.array(value), label=key, linestyle=linestyles[randint(0,3)])
    plt.legend()


# <h1 id="3">3. Input</h1>
# 
# <h3 id="3-1">3-1. Providing Sequence</h3>
# Taking a sample input sequence (a simple 1 to 100 here) in *raw_seq*

# In[ ]:


# define input sequence
raw_seq = [i for i in range(100)]

# Try the following if randomizing the sequence:
# random.seed('sam') # set the seed
# raw_seq = random.sample(raw_seq, 100)

# choose a number of time steps for sliding window from data start to target start
sliding_window = 20

# split into samples
X, y = split_sequence(raw_seq, sliding_window)

print(X)
print(y)


# <h3 id="3-2">3-2. Reshaping Sequence</h3>
# Reshaping accordingly for CNN Layer

# In[ ]:


# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 20
n_steps = 1
X = X.reshape((X.shape[0], n_seq, n_steps, n_features))


# <h1 id="4">4. Model</h1>
# 
# <h3 id="4-1">4-1. Defining Layers</h3>
# Adding 1D Convolution, Max Pooling, LSTM and finally Dense (MLP) layer

# In[ ]:


# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mse'])


# <h3 id="4-2">4-2. Training Model</h3>
# Defined early stop, can be used in callbacks param of model fit, not using for now since it's not recommended at first few iterations of experimentation with new data

# In[ ]:


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
# fit model
history = model.fit(X, y, epochs=100, verbose=1, validation_data=(X,y))


# <h3 id="4-3">4-3. Evaluating Model</h3>
# Plotting Training and Validation mean square error

# In[ ]:


#Plot Error
# Mean Square Error
yAxes = {}
yAxes["Training"]=history.history['mean_squared_error']
yAxes["Validation"]=history.history['val_mean_squared_error']
plot_multi_graph(history.epoch,yAxes, title='Mean Square Error',xAxisLabel='Epochs')


# <h1 id="5">5. Prediction</h1>
# 
# <h3 id="5-1">5-1. Single Value Prediction</h3>
# Predicting a single value slided 20 (our provided figure for sliding window above) values ahead

# In[ ]:


# demonstrate prediction
x_input = array([i for i in range(100,120)])
print(x_input)
x_input = x_input.reshape((1, n_seq, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# <h3 id="5-2">5-2. Sequence Prediction</h3>
# Predicting complete sequence (determining closeness to target) based on data <br />
# <i>change variable for any other sequence though</i>

# In[ ]:


# demonstrate prediction in data
yhat = model.predict(X, verbose=0)
print(yhat)


# In[ ]:


print(y)


# <h1 id="6">6. Complete Figure</h1>
# Data, Target, Prediction - all in one single graph

# In[ ]:


xAxis = [i for i in range(len(y))]
yAxes = {}
yAxes["Data"]=raw_seq[0:len(raw_seq)-sliding_window]
yAxes["Target"]=y
yAxes["Prediction"]=yhat
plot_multi_graph(xAxis,yAxes,title='')


# In[ ]:




