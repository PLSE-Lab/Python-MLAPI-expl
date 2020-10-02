#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv('/kaggle/input/breast-cancer-wisconsin-data/data.csv')
df.head()
X = df.iloc[:,2:32]
Y = df.iloc[:,1]


# In[ ]:


Y = Y.map({'M' : 0 , 'B' : 1})
Y = Y[:,np.newaxis]
Y.shape


# In[ ]:


from sklearn.model_selection import train_test_split
x,x_t,y,y_t = train_test_split(X,Y,random_state = 3,test_size = 0.2)
x,x_t,y,y_t = x.T,x_t.T,y.T,y_t.T


# In[ ]:


x = (x - np.min(x))/(np.max(x) - np.min(x)).values 


# In[ ]:


def sigmoid(z):
    return 1/(1 + np.exp(-z))


# In[ ]:


def costFunction(theta,x,y):
    z = np.dot(theta.T,x)
    pred = sigmoid(z)
    cost = -y*np.log(pred) - (1-y)*np.log(1-pred)
    loss = np.sum(cost)/x.shape[1]
    return loss
    


# In[ ]:


def gradient(theta,x,y):
    z = np.dot(theta.T,x)
    pred = sigmoid(z)
    grad = np.sum(np.dot(x,(pred-y).T))
    return grad/x.shape[1]


# In[ ]:


def gradientDescent(theta,x,y,a,iter_max):
    J_hist = []
    for _ in range(iter_max):
        grad = gradient(theta,x,y)
        theta -= a*grad
        J_hist.append(costFunction(theta,x,y))
    
    plt.plot(J_hist,range(iter_max))
    plt.show()
    return theta


# In[ ]:


def predict(theta,x):
    z = np.dot(theta.T,x)
    pred = sigmoid(z)
    res = [1 if i > 0.5 else 0 for i in pred[0] ]
    return res


# In[ ]:


def accuracy(y_hat,y_true):
    acc = np.sum(y_hat == y_true)/len(y_true)
    return acc


# In[ ]:


a = 0.01
i = 1000
theta = np.zeros(x.shape[0]).reshape((x.shape[0],1))
itheta = theta
theta.shape


# In[ ]:


costFunction(theta,x,y)


# In[ ]:


theta = gradientDescent(theta,x,y,a,i)


# In[ ]:


costFunction(theta,x,y)


# In[ ]:


x_t = (x_t - np.min(x_t))/(np.max(x_t) - np.min(x_t)).values 


# In[ ]:


y_hat = predict(theta,x_t)


# In[ ]:


accuracy(y_hat,y_t)


# import scipy.optimize as opt
# out = opt.fmin_cg(f = costFunction, x0 = itheta, fprime = gradient, args = (x,y.flatten()), maxiter = 400)

# In[ ]:




