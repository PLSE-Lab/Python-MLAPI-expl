#!/usr/bin/env python
# coding: utf-8

# A lot of insightful info has been found by kagglers. This kernel won't repeat work that has been done but focuses on one single point which may worth making a feature down the track.
# 
# While doing sanity check, I was a bit suprised price, as a crucial part of advertisement, is missing many values. Will that reduce the probability that the product would be sold?

# In[8]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[2]:


# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[4]:


train = pd.read_csv('../input/train.csv')


# In[5]:


train.shape


# There are many price data missing.

# In[6]:


train.isnull().sum()


# Missing rate:

# In[26]:


train.isnull().sum() / train.shape[0] * 100


# In[10]:


train.describe()


# Before heading to the illustration of products missing price, let's first have a quick look at the replation between price and deal probability.  Having the max price will suppress most points to the left of the chart and make it hard to read. We plot only 75% of the data.

# In[15]:


fig, ax = plt.subplots(figsize=(15, 8))
train.loc[train.price < 7e3, ['price', 'deal_probability']].plot(kind='scatter', x='price', y='deal_probability', ax=ax, alpha=0.1, color='r')
ax.grid()
plt.show()


# The plot above actually makes sense that most prices are populated around times of hundreds as appearing as the stripes in the chart.
# 
# Then quite suprisingly again, price-missing products actually have relatively higher chance to be sold.

# In[24]:


fig, ax = plt.subplots(figsize=(15, 8), nrows=2)
train.loc[train.price.isnull(), 'deal_probability'].plot(kind='hist', bins=20, ax=ax[0], color='r', grid=True)
ax[0].set_title('No price')
train.loc[train.price < 7e3, 'deal_probability'].plot(kind='hist', bins=20, ax=ax[1], color='b', grid=True)
ax[1].set_title('Price < 7,000')
ax[1].set_xlabel('Deal probability')
plt.show()


# Wondering what percentage of price data are mising in test set.

# In[27]:


test = pd.read_csv('../input/test.csv')


# In[28]:


test.isnull().sum() / test.shape[0] * 100


# The missing rate 6% is slightly higer than 5.6% for the train set.

# Anyway, this is a data quality issue that should be addressed. Maybe it means price negotiable, which probably explains the higher deal probability.

# In[ ]:




