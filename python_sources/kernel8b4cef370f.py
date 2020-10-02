#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


#Lets do all the required imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# In[3]:


#Getting the data as csv format
housing_data = pd.read_csv("../input/housing_data.csv",names=['size','no_of_bedroom','price'])


# In[4]:


#Lets have a look on the data
housing_data.head()


# In[6]:


#Lets normalize the data,we will do mean normalization
housing_data = (housing_data - housing_data.mean())/housing_data.std()


# In[7]:


from sklearn.linear_model import LinearRegression


# In[8]:


model = LinearRegression()


# In[9]:


#Lets get the feature and label for the linear model
X = housing_data[['size','no_of_bedroom']]
y = housing_data['price']

#Lets divide the data into training and testing dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# In[10]:


#Lets train the model

model.fit(X_train,y_train)


# In[13]:


#Lets predict some values on the basis of X_test

y_predict = model.predict(X_test)
y_predict


# In[14]:


from sklearn.metrics import mean_squared_error


# In[15]:


rmse = np.sqrt(mean_squared_error(y_test, y_predict))

print(f"Root mean squared error is {rmse}")


# In[16]:


#Lets define our cost function first
def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2)/(2 * m)
    return J


# In[17]:


m = len(y_train)
x0 = np.ones(m).reshape(-1,1)
X = np.concatenate((x0,X_train.values),axis=1)
y = y_train.values


# In[18]:


theta = np.zeros(len(housing_data.columns))

inital_cost = cost_function(X, y, theta)
print(inital_cost)


# In[19]:


def gradient_descent(X, Y, B, alpha, iterations):
    cost_history = [0] * iterations
    m = len(Y)
    
    for iteration in range(iterations):
        # Hypothesis Values
        h = X.dot(B)
        # Difference b/w Hypothesis and Actual Y
        loss = h - Y
        # Gradient Calculation
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_history[iteration] = cost
        
    return B, cost_history


# In[20]:


newB, cost_history = gradient_descent(X, y, theta, 0.001, 100000)

# New Values of B
print(newB)

# Final Cost of new B
print(cost_history[-1])

