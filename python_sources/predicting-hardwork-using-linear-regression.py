#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


x_test = pd.read_csv("../input/Linear_X_Test.csv")
X_Train = pd.read_csv("../input/Linear_X_Train.csv")
Y_train = pd.read_csv("../input/Linear_Y_Train.csv")


# In[3]:


plt.scatter(X_Train,Y_train)
plt.show()


# In[4]:


print(X_Train.shape)
print(Y_train.shape)


# In[5]:


from sklearn.linear_model import LinearRegression


# In[6]:


lr = LinearRegression()
lr.fit(X_Train,Y_train)


# In[7]:


xt = x_test.values
y_pred = lr.predict(xt.reshape(-1,1))


# In[9]:


x_test = x_test.drop(['x'], axis=1)


# In[10]:


x_test['y'] = y_pred


# In[12]:


x_test.to_csv('hardwork.csv', index=False)


# In[ ]:




