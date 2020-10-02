#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


train = pd.read_csv("../input/result.csv") 
test = pd.read_csv("../input/result.csv")
#train and test the file


# In[5]:


train.head()


# In[6]:


train.describe()


# In[7]:


train.columns


# In[8]:


feature = ["product_quantity","customer_location","cost_Per_Item_vendor_v2","cost_Per_Item_vendor_v3","cost_Per_Item_vendor_v4"]
label = "cost_Per_Item_vendor_v1" 
#feature and label values


# In[9]:


X_train = train[feature] 
#train feature


# In[10]:


X_train.isnull().sum()
#check for null if it is encode fit and tranform


# In[11]:


Y=train[label]
#train label


# In[12]:


Y.isnull().sum()


# In[13]:


lm = LinearRegression()
#Linear Regression model


# In[14]:


lm.fit(X_train,Y)
#fit model


# In[15]:


lm.score(X_train,Y)
#score model


# In[16]:


X_test = test[feature].copy()
#test feature


# In[17]:


X_test.isnull().sum()


# In[23]:


lm.predict(X_test)
#predict feature test


# In[19]:


test["cost_Per_Item_vendor_v1"] = lm.predict(X_test)
#assign predicted to label


# In[20]:


plt.scatter(train[label],train["customer_location"], c="g",marker="+")
plt.ylabel("customer_location")
plt.xlabel("cost_Per_Item_vendor_v1")
#ploting graph for vendor1 by location


# In[21]:


plt.scatter(train[label],train["product_quantity"], c="g",marker="+")
plt.ylabel("product_quantity")
plt.xlabel("cost_Per_Item_vendor_v1")
#ploting vendor1 by product


# In[22]:


test[["zone","cost_Per_Item_vendor_v1"]].to_csv("Submission.csv",index = False)


# In[ ]:




