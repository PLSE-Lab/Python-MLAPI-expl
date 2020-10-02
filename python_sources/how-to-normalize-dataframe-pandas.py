#!/usr/bin/env python
# coding: utf-8

# Hi, in this tutorial you will learn to normalize values in dataframe.
# 
# **This notebook will cover:**
# 1. Normalizing a single row.
# 2. Normalizing entire dataframe.
# 3. Normalizing entire dataframe but not few columns.

# Reading the data

# In[52]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
data = pd.read_csv('../input/data.csv')


# Let's see what our dataset looks like.

# In[53]:


data.sample(5)


# First map the string values of diagnosis to integer.

# In[54]:


def mapping(data,feature):
    featureMap=dict()
    count=0
    for i in sorted(data[feature].unique(),reverse=True):
        featureMap[i]=count
        count=count+1
    data[feature]=data[feature].map(featureMap)
    return data


# In[55]:


data=mapping(data,"diagnosis")


# And drop the unrequired columns.

# In[56]:


data=data.drop(["id","Unnamed: 32"],axis=1)
data.sample(5)


# **Normalizing just a single column**
# 
# Let's normalize concavity_mean in the range 0 to 20

# In[57]:


data["concavity_mean"]=((data["concavity_mean"]-data["concavity_mean"].min())/(data["concavity_mean"].max()-data["concavity_mean"].min()))*20


# In[58]:


data.sample(5)


# **Normalizing full dataframe**

# In[59]:


dataf=((data-data.min())/(data.max()-data.min()))*20


# In[60]:


dataf.sample(5)


# **Normalizing full dataframe except few columns**
# 
# We don't want to normalize "diagnosis"

# In[61]:


def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min()))*20
    dataNorm["diagnosis"]=dataset["diagnosis"]
    return dataNorm


# In[62]:


data=normalize(data)
data.sample(5)


# So this was all about normalizing a dataframe. I made this tutorial because standard scalars work only on numpy arrays and if we try to convert dataframes to numpy array, we'll lose column names. I hope you like it.
