#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importing required Packages
import pandas as pd
import numpy as np
import scipy.stats as stat
get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
import random
import time


# In[ ]:


data = pd.read_csv("../input/SIMPLEPENDULUMOSCILLATIONDATA.txt", sep=" ", header=None, names=['l', 't'])
print(data.head())
print(data.tail())


# In[ ]:


l = data['l'].values
t = data['t'].values
tsq = t * t


# ## Mini-Batch Gradient Descent
# 
# In Mini-Batch Gradient Descent algorithm, rather than using  the complete data set, in every iteration we use a subset of training examples (called "batch") to compute the gradient of the cost function. 
# 
# Common mini-batch sizes range between 50 and 256, but can vary for different applications.
# 
# one_batch() : we will be calculating the essenial parts of the Gradient Descent method:  
# 
# $y = mx + c$
#         
# $E$ =$\frac{1}{n}$   $\sum_{i=1}^n (y_i - y)^2$
# 
# $\frac{\partial E }{\partial m}$ = $\frac{2}{n}$   $\sum_{i=1}^n  -x_i(y_i - (mx_i + c))$
#  
# $\frac{\partial E}{\partial c}$ = $\frac{2}{n}$   $\sum_{i=1}^n  -(y_i - (mx_i + c))$
# 
# one_step() : We will be splitting our data into batches.

# In[ ]:


def train_one_batch(x, y, m, c, eta):
    const = - 2.0/len(y)
    ycalc = m * x + c
    delta_m = const * sum(x * (y - ycalc))
    delta_c = const * sum(y - ycalc)
    m = m - delta_m * eta
    c = c - delta_c * eta
    error = sum((y - ycalc)**2)/len(y)
    return m, c, error

def train_batches(x, y, m, c, eta, batch_size):
    # Making the batches
    random_idx = np.arange(len(y))
    np.random.shuffle(random_idx)
    
    # Train each batch
    for batch in range(len(y)//batch_size):
        batch_idx = random_idx[batch*batch_size:(batch+1)*batch_size]
        batch_x = x[batch_idx]
        batch_y = y[batch_idx]
        m, c, err = train_one_batch(batch_x, batch_y, m, c, eta)
    
    return m, c, err

def train_minibatch(x, y, m, c, eta, batch_size=10, iterations=1000):
    for iteration in range(iterations):
        m, c, err = train_batches(x, y, m, c, eta, batch_size)
    return m, c, err


# In[ ]:


# Init m, c
m, c = 0, 0

# Learning rate
lr = 0.001

# Batch size
batch_size = 10


# In[ ]:


# Training for 1000 iterations, plotting after every 100 iterations:
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
plt.ion()
fig.show()
fig.canvas.draw()

for num in range(10):
    m, c, error = train_minibatch(l, tsq, m, c, lr, batch_size=90, iterations=100)
    print("m = {0:.6} c = {1:.6} Error = {2:.6}".format(m, c, error))
    y = m * l + c
    ax.clear()
    ax.plot(l, tsq, '.k')
    ax.plot(l, y)
    fig.canvas.draw()
    time.sleep(1)


# 
# ## Plotting error vs iterations

# In[ ]:


ms, cs,errs = [], [], []
m, c = 0, 0
lr = 0.001
batch_size = 10
for times in range(100):
    m, c, error = train_minibatch(l, tsq, m, c, lr, batch_size, iterations=100) # We will plot the value of for every 100 iterations
    ms.append(m)
    cs.append(c)
    errs.append(error)
epoch = range(0, 10000, 100)
plt.figure(figsize=(8, 5))
plt.plot(epoch, errs)
plt.xlabel("Iterations")
plt.ylabel("Error")
plt.title("Minibatch Gradient Descent")
plt.show()


# check the error value at saturation, and time it takes to reach saturation.

# In[ ]:




