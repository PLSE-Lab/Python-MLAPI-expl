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


df1=pd.read_csv("../input/Summary of Weather.csv")


# In[3]:


from sklearn.linear_model import LinearRegression

lr = LinearRegression()


# In[4]:


x=df1.MinTemp.values


# In[5]:


x.shape


# In[6]:


x=x.reshape(-1,1)


# In[7]:


x.shape


# In[8]:


y=df1.MaxTemp.values.reshape(-1,1)


# In[9]:


lr.fit(x,y)


# In[10]:


X=np.array([10,20,30,40,50]).reshape(-1,1)


# In[11]:


print("Results")
for i in X:
    print("Min:",i,"Predicted Max:",lr.predict([i]))


# In[12]:


#Visualize
import matplotlib.pyplot as plt
X
plt.scatter(x,y)
plt.show()


# In[13]:


y_head=lr.predict(X)
plt.scatter(X,y_head,color="red")
plt.show()


# In[ ]:




