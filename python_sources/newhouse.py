#!/usr/bin/env python
# coding: utf-8

# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train =pd.read_csv("../input/train.csv")
test =pd.read_csv("../input/test.csv")


# In[3]:


train.head()


# In[25]:


feature = ["LotArea","GrLivArea","FullBath","HalfBath","BedroomAbvGr","KitchenAbvGr","KitchenQual","GarageArea","YrSold"]
label = "SalePrice"


# In[26]:


X_train = train[feature]


# In[27]:


Y = train[label]


# In[28]:


X_train.isnull().sum()


# In[29]:


Y.isnull().sum()


# In[35]:


train["YrSold"].isnull().sum()


# In[31]:


lm = LinearRegression()


# In[33]:


lm.fit(X_train,train[label])


# In[ ]:




