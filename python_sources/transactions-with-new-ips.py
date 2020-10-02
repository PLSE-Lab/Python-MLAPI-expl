#!/usr/bin/env python
# coding: utf-8

# In[21]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# It was pointed out that over half of the IPs that appear in test data are new (not in the training data). In this notebook i will show how this translate if we compare 2 days from the training data. in this case we will compare day 7 and day 8 of November. 
# We could see that the same pattern exist in the training set. 77802 of the unique IPs in day 8 are new (didnt exist in day 7). However, if we check the amount of transactions in day 8 with new IPs, we notice that it only make a small percentage of ~ 2%. 
# 

# **Load train data from 2017-11-07

# In[22]:


train_day_7 = pd.read_csv('../input/train.csv', skiprows=range(1,9308568), nrows=59633310, usecols=['click_time', 'ip', 'is_attributed'])


# **Load train data from 2017-11-08

# In[23]:


train_day_8 = pd.read_csv('../input/train.csv', skiprows=range(1,68941878), nrows=62945075, usecols=['click_time', 'ip', 'is_attributed'])


# In[24]:


print (train_day_7.shape, train_day_7.is_attributed.mean())


# In[25]:


print (train_day_8.shape, train_day_8.is_attributed.mean())


# In[26]:


train_day_7.ip.nunique()


# In[27]:


train_day_8.ip.nunique()


# How many unique IPs are new in day8 (didnt appear in day7)

# In[28]:


train_day_8[~train_day_8.ip.isin(train_day_7.ip.unique())].ip.nunique()


# so about half of the unique IPs in day 8 are new. Now lets see how many transactions in day8 are from new IPs

# In[29]:


train_day_8[~train_day_8.ip.isin(train_day_7.ip.unique())].shape[0] / float(train_day_8.shape[0])


# As can be seen only 2.5 % of transactions in day 8 are from new IPs. What is the mean is_attributed for the transactions in day8 that are from new IPs 

# In[30]:


train_day_8[~train_day_8.ip.isin(train_day_7.ip.unique())].is_attributed.mean()


# As can be seen transactions from new IPs are about 15 times more likely to end in the app download. 0.037 compared to 0.0025
