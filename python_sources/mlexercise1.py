#!/usr/bin/env python
# coding: utf-8

# Converting Andrew Ng's first ml exercise to python.

# In[18]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[19]:


data = pd.read_csv('../input/ex1data1.txt', header=None, names=['Population', 'Profit'])

X = data.iloc[:,0] # read first column
y = data.iloc[:,1] # read second column
m = len(y) # number of training example
data.head() # view first few rows of the data


# In[4]:


data.describe()


# In[20]:


X = data['Population']
y = data['Profit']
colors = [0,0,0]
area = 50
plt.scatter(X, y, s=area, c='c', marker='x', alpha=0.5)
plt.title('Population vs Profit')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')


# In[21]:


# Another method to plot 
data.plot(kind='scatter', x='Population', y='Profit', figsize=(12,8))


# In[22]:


# Yet another method to plot
plt.scatter(X, y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.show()


# In[23]:


X = X[:,np.newaxis]
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01
m = len(y)
ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term


# In[24]:


def computeCost(X, y, theta):
   R = X @ theta - y
   U = R*R
   sum1 = np.sum(U)
   J = sum1 / (2*m);
   return J


# In[25]:


J = computeCost(X, y, theta)
print(J)


# You should expect to see a cost of 32.07.

# In[26]:


theta[0, 0] = -1
theta[1, 0] = 2
J = computeCost(X, y, theta)
print(J)


# Expected cost value (approx) 54.24

# In[27]:


def gradientDescent(X, y, theta, alpha, iterations):
    for _ in range(iterations):
        R = X @ theta - y
        U = X.T @ R
        theta = theta - (alpha/m) * U
    return theta


# In[28]:


theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)


# Expected theta values [-3.6303, 1.1664]

# In[29]:


theta[0, 0] = -1
theta[1, 0] = 2
theta = gradientDescent(X, y, theta, alpha, iterations)
print(theta)


# In[30]:


J = computeCost(X, y, theta)
print(J)


# It should give you a value of 4.483 which is much better than 32.07

# In[31]:


plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()


# **Linear Regression with multiple variables**
# 
# In this part, you will implement linear regression with multiple variables to predict the prices of houses. Suppose you are selling your house and you want to know what a good market price would be. One way to do this is to first collect information on recent houses sold and make a model of housing prices.
# 
# 

# In[32]:


data = pd.read_csv('../input/ex1data2.txt', header=None, names=['Size', '#Bedroom','Price'])
data.head()


# In[33]:


X = data.iloc[:,0:2] # read first two columns into X
y = data.iloc[:,2] # read the third column into y
m = len(y) # no. of training samples


# In[34]:


# this
X1 = (X - np.mean(X))/np.std(X)
# or this
mu = np.mean(X)
sigma = np.std(X)
X = (X - mu)/sigma
# above both method produce same result


# In[35]:


y = y[:,np.newaxis]

ones = np.ones((m,1))
X = np.hstack((ones, X)) # adding the intercept term

theta = np.zeros([3,1])
alpha = 0.01
num_iters = 400

print(X.shape)
print(theta.shape)


# In[36]:


def computeCostMulti(X, y, theta):
    temp = np.dot(X, theta) - y
    return np.sum(np.power(temp, 2)) / (2*m)
J = computeCostMulti(X, y, theta)
print(J)


# In[37]:


def gradientDescentMulti(X, y, theta, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        R = X @ theta - y;
        U = X.T @ R;
        theta = theta - (alpha/m) * U
    return theta


# In[38]:


theta = gradientDescentMulti(X, y, theta, alpha, num_iters);
print(theta)


# In[39]:


J = computeCostMulti(X, y, theta)
print(J)


# This should give you a value of 2105448288.6292474 which is much better than 65591548106.45744

# In[ ]:




