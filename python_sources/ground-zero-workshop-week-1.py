#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip3 install pandas
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 


# In[ ]:


data_set = pd.read_csv('../input/ex1data1.txt', sep=',', header=None)
data_set.columns


# In[ ]:



X = data_set[0]
y = data_set[1]
m = len(y)
y = y.values.reshape([m, 1])


# In[ ]:


plt.scatter(X, y,  color='blue')

plt.show()


# In[ ]:


# make numpy array(matrices)
X_test = np.c_[np.ones(m), X] # add one's col. for vectorization
y_test = np.array(y) 
w = np.zeros([2, 1]) # weights 
alpha = 0.01 # step for gradient descent
num_iters = 50 # no. iterations


# In[ ]:


# make a model 
def h(X,w):
    return X@w


# 

# In[ ]:


#cost/loss/error funciton
def cost_func(X, y, w):
    J = #YOUR ANSWER
    return J.sum()
_func(X_test, y_test, w)


# In[ ]:


# gradient descent
def grad(X, y, w, alpha, num_iters):      
    for i in range(1500):
        w -= alpha * (1/m * X.T@(X@w - y))
    return w

grad(X_test, y_test, w, alpha, num_iters=1500)


# In[ ]:


plt.plot(X, h(X_test, w), color='red', linewidth=3)
plt.scatter(X, y, color='blue')

plt.show()


# In[ ]:


w


# In[ ]:


x1=[1,15]
x1@w


# In[ ]:




