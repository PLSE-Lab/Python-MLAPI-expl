#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


data = pd.read_csv('../input/train.csv')
data.head()


# In[ ]:


X = data['GrLivArea']
Y = data['SalePrice']

plt.scatter(X,Y)


# In[ ]:


X = (X - X.mean())/X.std()
X = np.c_[np.ones(X.shape[0]),X]
X.shape


# In[ ]:


theta = np.random.randn(2)

x1 = np.linspace(-3,10,1000)
y1 = theta[0] + theta[1]*x1

plt.scatter(X[:,1],Y)
plt.plot(x1,y1)


# In[ ]:


alpha = .01
iterations = 1000

def gradient(X,Y,alpha,iterations,theta):
    costs = []
    m = X.shape[0]
    for i in range(iterations):
        pred = np.dot(X,theta)
        error = pred - Y
        cost = (1/m)*np.dot(error.T,error)
        
        costs.append(cost)
        theta = theta - (alpha/m) * np.dot(X.T,error)
        
    return theta,costs


# In[ ]:


theta_f,costss = gradient(X,Y,alpha,iterations,theta)


# In[ ]:


theta_f


# In[ ]:


plt.plot(costss)


# In[ ]:


plt.scatter(X[:,1],Y,color = 'red')
x1  = np.linspace(-3,20,1000)
y1 = x1*theta_f[1] + theta_f[0]
plt.plot(x1,y1)


# In[ ]:




