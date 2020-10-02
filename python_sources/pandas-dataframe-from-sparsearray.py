#!/usr/bin/env python
# coding: utf-8

# # Experimenting with making Pandas DataFrame from SparseArray
# Based on this issue:
# https://github.com/pandas-dev/pandas/issues/26378#issuecomment-492054680

# In[48]:


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


# ### Make Sparse Array

# In[49]:


np.random.seed(99)
arr = np.random.randn(1000)
arr[arr<0.9] = np.nan


# In[50]:


arr


# In[51]:


sparr = pd.SparseArray(arr)
sparr


# ### Make Series from SparseArray

# In[52]:


ser = pd.Series(sparr, name='name')


# In[53]:


ser.head()


# In[54]:


df = pd.DataFrame([ser])
df.shape


# ### This Pandas DataFrame does not look sparse

# In[55]:


df.info()


# In[56]:


df.to_dense().info()


# In[ ]:




