#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# * ### Checking if we can get Ad durations from periods

# In[2]:


train = pd.read_csv("../input/train.csv", usecols=['item_id', 'user_id', 'deal_probability'], 
                    dtype={'item_id': str, 'user_id': str, 'deal_probability': float})
train.head()


# In[3]:


periods_train = pd.read_csv("../input/periods_train.csv", parse_dates = ['activation_date', 'date_from', 'date_to'],
                    dtype={'item_id': str})
periods_train.head()


# In[4]:


#Check common item id's
print(len(set(train.item_id)))
print(len(set(periods_train.item_id)))
print(len(set(train.item_id).intersection(set(periods_train.item_id))))


# ** I am not able to find any common item_id between train and periods_train. Although data description mentions it **

# In[5]:


train_active = pd.read_csv("../input/train_active.csv", usecols=["item_id", "user_id", "item_seq_number"])
train_active.head()


# In[6]:


print(len(set(train_active.item_id)))
print(len(set(train.item_id).intersection(set(train_active.item_id))))


# In[ ]:




