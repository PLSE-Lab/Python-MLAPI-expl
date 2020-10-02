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


# In[26]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[27]:


train.head()


# In[29]:


df3 = test[['item_id','price']].copy()


# In[30]:


df3['deal_probability'] =  train['deal_probability'].mean()


# In[31]:


df3.head()


# In[32]:


df3[['item_id','deal_probability']].to_csv('simple_mean_benchmark.csv', index=False)

