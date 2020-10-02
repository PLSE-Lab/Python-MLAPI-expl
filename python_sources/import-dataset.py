#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np 
import pandas as pd 
import os

print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[3]:


train_data = pd.read_csv('../input/train.csv', index_col = 0)


# In[4]:


train_data.head()


# In[5]:


train_data.shape


# In[6]:


train_data.info()


# ** Lots of columns have missing entries and some columns have very less entries...need further investigation.**
# 
# **Problem** : ** Supervised Regression Problem**
# 
# **Predict sales price (dependent variable) of houses given 79 other independent variables.**
