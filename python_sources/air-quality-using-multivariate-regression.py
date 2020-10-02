#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


x_test = pd.read_csv("../input/Test.csv")
Train = pd.read_csv("../input/Train.csv")


# In[3]:


Train = Train.values
print(Train.shape)


# In[4]:


X = Train[:,:5]
Y = Train[:,5]


# In[5]:


print(X.shape)
print(Y.shape)


# In[6]:


from sklearn.linear_model import LinearRegression


# In[7]:


lr = LinearRegression(normalize = True)
lr.fit(X,Y)


# In[8]:


xt = x_test.values
print(xt.shape)


# In[9]:


y_pred = lr.predict(xt)


# In[10]:


x_test = x_test.drop(['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'], axis=1)
x_test['target'] = y_pred


# In[11]:


x_test.to_csv('hardwork.csv', index=True)


# In[ ]:




