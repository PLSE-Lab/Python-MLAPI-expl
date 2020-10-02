#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Self Organizing Maps

# Importing Required Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


# Importing dataset
dataset = pd.read_csv('Credit_Card_Applications.csv')
dataset.head(10)


# In[3]:


dataset.columns


# In[4]:


X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


# In[5]:


# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
X = sc.fit_transform(X)


# In[6]:


# Training the SOM
from minisom import MiniSom


# In[7]:


som = MiniSom(x=10,y=10,input_len=15)


# In[8]:


som.random_weights_init(X)
som.train_random(data = X, num_iteration = 100)


# In[9]:


dataset.info()


# In[10]:


som.distance_map()


# In[11]:


# visualizing the results 
from pylab import bone, colorbar, pcolor, plot, show


# In[12]:


bone()
pcolor(som.distance_map().T)
colorbar()


# In[13]:


markers = ['o', 's']
colors = ['r', 'g']
for i,x in enumerate(X):
    w = som.winner(x)
    plot(w[0]+0.5,
         w[1]+0.5,
         markers[y[i]], 
         markeredgecolor=colors[y[i]], 
         markerfacecolor='None', 
         markersize=10, 
         markeredgewidth=2)
show()


# In[14]:


# Find frauds
mappings = som.win_map(X)
mappings


# In[17]:


frauds = mappings[(2,4)]


# In[18]:


frauds


# In[19]:


frauds = sc.inverse_transform(frauds)
frauds

