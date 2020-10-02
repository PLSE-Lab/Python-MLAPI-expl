#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[2]:


x_test = pd.read_csv("../input/Logistic_X_Test.csv")
X_Train = pd.read_csv("../input/Logistic_X_Train.csv")
Y_train = pd.read_csv("../input/Logistic_Y_Train.csv")


# In[3]:


X = X_Train.values
Y = Y_train.values
xt = x_test.values


# In[7]:


print(X.shape)
print(Y.shape)
Y = Y.reshape(3000,)
print(Y.shape)
print(xt.shape)


# In[12]:


model = LogisticRegression(solver='sag')


# In[13]:


model.fit(X, Y)


# In[14]:


y_pred = model.predict(xt)


# In[15]:


x_test = x_test.drop(['f1', 'f2', 'f3'], axis = 1)
x_test['label'] = y_pred


# In[16]:


x_test.to_csv('chemicals.csv', index=False)


# In[ ]:




