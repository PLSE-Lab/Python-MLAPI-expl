#!/usr/bin/env python
# coding: utf-8

# # Sine wave RNN
# 
# Goal of this notebook is to use RNN in order to predict the sine wave.

# ![sine.png](attachment:sine.png)

# **Table content**
# 
# 1. [Exploratory Data Analysis](#eda)
# 2. [Split Data into Training and Testing](#split)
# 3. [implementation](#implementation)<br>
#     3.1. [Activation Function](#activation)<br>
#     3.2. [Forward Function](#forward)<br>
#     3.3. [Clip Min-Max Function](#clip)<br>
#     3.4. [Backward Function](#backward)<br>
#     3.5. [Optimize Function](#optimize)<br>
#     3.6. [Forward Function](#forward)<br>
#     3.7. [Loss/Value Loss Functions](#loss)<br>
#     3.8. [Training](#train)<br>
# 4. [Training Analysis](#analysis)

# <a id='eda'></a>
# # Exploratory Data Analysis

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math


# In[ ]:


df = pd.read_csv('../input/sinwave/Sin Wave Data Generator.csv', delimiter=',', nrows = 600)
df.head()


# In[ ]:


plt.plot(df)


# In[ ]:


sin_wave = (df.to_numpy()).reshape(len(df))


# <a id='split'></a>
# # Split Data into Training and Testing

# In[ ]:


def get_sequence_data(df, seq_len):
    X, Y = [], []
    nr_records = len(df) - seq_len
    sin_wave = df

    for i in range(nr_records - seq_len):
        X.append(sin_wave[i:i+seq_len])
        Y.append(sin_wave[i+seq_len])
    
    return X, Y


# In[ ]:


def get_test_data(df, seq_len, len_test):
    X, Y = [], []
    nr_records = len(df) - seq_len
    sin_wave = df
    
    for i in range(nr_records - len_test, nr_records):
        X.append(sin_wave[i:i+seq_len])
        Y.append(sin_wave[i+seq_len])

    return X, Y


# In[ ]:


def list_to_array(X, Y):
    
    X = np.array(X)
    Y = np.array(Y)
    
    X = np.array(X)
    X = np.expand_dims(X, axis=2)

    Y = np.array(Y)
    Y = np.expand_dims(Y, axis=1)
    
    return X, Y


# In[ ]:


seq_len = T =  100
len_test = 100


# In[ ]:


X_train, y_train = get_sequence_data(sin_wave[:len(sin_wave)], seq_len)
X_train, y_train = list_to_array(X_train, y_train)

X_test, y_test = get_test_data(sin_wave[:len(sin_wave)], seq_len, 100)
X_test, y_test = list_to_array(X_test, y_test)


# In[ ]:


len_data = X_train.shape[0]


# In[ ]:


X_train.shape, y_train.shape


# In[ ]:


X_test.shape, y_test.shape


# <a id='implementation'></a>
# # Implementation

# <a id='activation'></a>
# # Activation Function

# In[ ]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# <a id='forward'></a>
# # Forward Function

# In[ ]:


def forward(x, y, prev_s):
    layers = []
    
    for t in range(T):
        new_input = np.zeros(x.shape)
        new_input[t] = x[t]

        m = np.dot(U, new_input)
        n = np.dot(W, prev_s)

        o = n + m

        s = sigmoid(o)
        p = np.dot(V, s)

        layers.append({'s':s, 'prev_s':prev_s})
        prev_s = s
        
    return (m, n, o, s, p), layers


# <a id='clip'></a>
# # Clip Min-Max Function

# In[ ]:


def clip_min_max(dU, dV, dW):
    if dU.max() > max_clip_value:
        dU[dU > max_clip_value] = max_clip_value
    if dV.max() > max_clip_value:
        dV[dV > max_clip_value] = max_clip_value
    if dW.max() > max_clip_value:
        dW[dW > max_clip_value] = max_clip_value


    if dU.min() < min_clip_value:
        dU[dU < min_clip_value] = min_clip_value
    if dV.min() < min_clip_value:
        dV[dV < min_clip_value] = min_clip_value
    if dW.min() < min_clip_value:
        dW[dW < min_clip_value] = min_clip_value
            
    return dU, dV, dW


# <a id='backward'></a>
# # Backward Function

# In[ ]:


def backward(alpha, y, layers):
    m, n, o, s, p = alpha
    
    dU = np.zeros(U.shape)
    dV = np.zeros(V.shape)
    dW = np.zeros(W.shape)

    dU_t = np.zeros(U.shape)
    dV_t = np.zeros(V.shape)
    dW_t = np.zeros(W.shape)

    dU_i = np.zeros(U.shape)
    dW_i = np.zeros(W.shape)
    
    dp = (p - y)

    for t in range(T):
        dV_t = np.dot(dp, np.transpose(layers[t]['s']))
        dsv = np.dot(np.transpose(V), dp)

        ds = dsv
        do = o * (1 - o) * ds
        dn = do * np.ones_like(n)

        dprev_s = np.dot(np.transpose(W), dn)

        for j in range(t-1, max(-1, t-bptt_truncate-1), -1):
            dV_i = np.dot(dp, np.transpose(layers[j]['s']))

            ds = dsv + dprev_s
            do = o * (1 - o) * ds

            dn = do * np.ones_like(n)
            dm = do * np.ones_like(m)

            dW_i = np.dot(W, layers[t]['prev_s'])
            dprev_s = np.dot(np.transpose(W), dn)

            new_input = np.zeros(x.shape)
            new_input[t] = x[t]
            dU_i = np.dot(U, new_input)
            dx = np.dot(np.transpose(U), dm)

            dU_t += dU_i
            dV_t += dV_i
            dW_t += dW_i

        dU += dU_t
        dV += dV_t
        dW += dW_t
            
    return clip_min_max(dU, dV, dW)


# <a id='optimize'></a>
# # Optimize Function

# In[ ]:


def optimize(alpha, grads):
    dU, dV, dW = grads
    U, V, W = alpha
    
    U -= learning_rate * dU
    V -= learning_rate * dV
    W -= learning_rate * dW
    
    return U, V, W


# <a id='loss'></a>
# # Loss/Value Loss Functions

# In[ ]:


def loss_fn(alpha, y):
    m, n, o, s, p = alpha
    
    return (y - p)**2 / 2


# In[ ]:


def val_loss_fn(alpha):
    m, n, o, s, p = alpha
    val_loss = 0.0
    
    for i in range(y_test.shape[0]):
        x, y = X_test[i], y_test[i]
        prev_s = np.zeros((hidden_dim, 1))
        alpha = forward(x, y, prev_s)

        loss_per_record = (y - p)**2 / 2
        val_loss += loss_per_record
    return val_loss / float(len_data)


# <a id='train'></a>
# # Training

# In[ ]:


learning_rate = 0.001
epochs = 16
bptt_truncate = 4
min_clip_value = -1
max_clip_value = 1
hidden_dim = 100
output_dim = 1


# In[ ]:


U = np.random.uniform(0, 1, (hidden_dim, T))
W = np.random.uniform(0, 1, (hidden_dim, hidden_dim))
V = np.random.uniform(0, 1, (output_dim, hidden_dim))


# In[ ]:


for epoch in range(epochs):

    loss = 0.0
    
    for i in range(len_data):
        # initialize
        x, y = X_train[i], y_train[i]
    
        prev_s = np.zeros((hidden_dim, 1))
        
        # forward pass
        alpha, layers = forward(x, y, prev_s)
        
        # loss
        loss += loss_fn(alpha, y)
            
        # backward pass
        grads = backward(alpha, y, layers)
        
        # update
        U, V, W = optimize((U, V, W), grads)
        
    # loss
    loss = loss / float(len_data)
    
    # value loss
    if(epoch % 2 == 0):
        val_loss = val_loss_fn(alpha)

        print('Epoch: ', epoch + 1, ', Loss: ', loss, ', Val Loss: ', val_loss)


# <a id='analysis'></a>
# # Training Analysis

# In[ ]:


preds = []
for i in range(y_test.shape[0]):
    x, y = X_test[i], y_test[i]
    prev_s = np.zeros((hidden_dim, 1))
    # Forward pass
    for t in range(T):
        mulu = np.dot(U, x)
        mulw = np.dot(W, prev_s)
        add = mulw + mulu
        s = sigmoid(add)
        mulv = np.dot(V, s)
        prev_s = s

    preds.append(mulv)
    
preds = np.concatenate(preds).squeeze()

plt.plot(preds, 'b-')
plt.plot(y_test.squeeze(), 'r-')

