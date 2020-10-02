#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


train = pd.read_csv("../input/en_train.csv")
test = pd.read_csv("../input/en_test.csv")


# In[4]:


# check first 15 row in train set.
train.head(15)


# In[5]:


# chekc first 15 row in test set.
test.head(15)


# How many classes are there?

# In[6]:


print("Total class: ", train['class'].unique().size)
train['class'].unique()


# In[ ]:




