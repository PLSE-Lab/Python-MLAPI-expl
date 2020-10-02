#!/usr/bin/env python
# coding: utf-8

# In[2]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets.samples_generator import make_regression
from matplotlib import pyplot as plt


# In[3]:


x, y = make_regression(n_samples=200, n_features=4, noise=20, random_state=1)

print(x.shape, y.shape)


# ## Normal Equation Formula
# $(X^TX)^{-1}* X^T * y$

# In[8]:


def NormalEquation(x, y):
    m = x.shape[0]
    y = y.reshape((y.shape[0], 1))
    X = np.concatenate((np.ones((m, 1)), x), axis=1)
    a = np.linalg.inv(X.T.dot(X))
    return (a.dot(X.T)).dot(y)


# In[9]:


NormalEquation(x, y)

