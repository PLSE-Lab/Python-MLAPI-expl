#!/usr/bin/env python
# coding: utf-8

# # Beginner memory reduction analysis on IEEE-CIS Fraud Detection
# 
# This is a very simple beginner example on how to reduce memory usage for this dataset. The goal is just to get a reasonable starting point to start from.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Let's start by reading in the data and doing some minor analysis.

# In[ ]:


train_identity = pd.read_csv('/kaggle/input/train_identity.csv')
train_transaction = pd.read_csv('/kaggle/input/train_transaction.csv')


# In[ ]:


print(train_identity.shape)
print(train_transaction.shape)


# In[ ]:


original_memory_usage = train_transaction.memory_usage()
original_memory_usage


# In[ ]:


total_original_memory_usage = original_memory_usage.sum()


# That took up 2gigs of RAM (you should be able to see it in the right sidebar of your kernel). 
# 
# Let's look at the data to see if we can use some other datatypes for some of the columns.

# In[ ]:


a = (train_identity.dtypes == 'object')
id_cat_cols = list(a[a].index)
a = (train_transaction.dtypes == 'object')
trans_cat_cols = list(a[a].index)
id_num_cols = list(set(train_identity.columns) - set(id_cat_cols))
trans_num_cols = list(set(train_transaction.columns) - set(trans_cat_cols))


# In[ ]:


print(len(id_cat_cols))
print(len(id_num_cols))
print(len(trans_cat_cols))
print(len(trans_num_cols))


# In[ ]:


train_transaction.dtypes


# A lot of these columns look like they are integers. I suspect most of them are cast to float64 by default because they contain NaN values. Let's find out:

# In[ ]:


train_transaction[trans_num_cols].isnull().sum().sort_values(ascending=False)


# Yes, looks like a significant portion of these columns contain NaN values. 
# 
# Let's see if we can replace the NaNs with a known round number, e.g. -1.0 and get all the columns that are really integers.

# In[ ]:


trans_integer_cols = []
for c in trans_num_cols:
    try:
        if train_transaction[c].fillna(-1.0).apply(float.is_integer).all():
            trans_integer_cols += [c]
    except Exception as e:
        print("error: ", c, e)


# In[ ]:


len(trans_integer_cols)


# We got a couple of errors from passing in numeric columns that are already integer columns, so we can safely ignore them.
# 
# But wow, 297 columns! But just because they're integers it doesn't mean we can reduce memory for all of them. They might be very large integers that we need 64 bits to represent.
# 
# So let's look at the statistics of these columns using `describe()`

# In[ ]:


stats = train_transaction[trans_integer_cols].describe().transpose()
stats


# As you can see above, these are really small integers. They don't all need to be float64!

# Let's look at what can be cast as int8

# In[ ]:


int8columns = stats[stats['max'] < 256].index
print(int8columns.shape)
print(int8columns)


# That's 233 columns!

# What about int16?

# In[ ]:


int16columns = stats[(stats['max'] >= 256) & (stats['max'] <= 32767)].index
print(int16columns.shape)
print(int16columns)


# In[ ]:


int8columns.shape[0] + int16columns.shape[0]


# The max precision for all our integer columns is 16bits, where 233 columns only need 8 bits and 64 columns need 16 bits. Looks like we can reduce the memory usage by quite a bit!
# 
# So let's do that right now. We'll replace NaN values with `-1.0` as we previously did, and cast the columns to the corresponding integer type. Note that in practice you may want to do something different, e.g. use the median.
# 
# It might also helpful to mark which rows were previously null, which you could in theory derive from looking at whether it's `-1.0` or not, but if you're using something like the median you would lose track of them. The fact that the value is missing or NaN might be meaningful in your context, so it's a good idea to mark them, using a separate column.

# In[ ]:


for c in int8columns:
    train_transaction[f'{c}_isna'] = train_transaction[c].isnull()
    train_transaction[c].fillna(-1.0, inplace=True)
    train_transaction[c] = train_transaction[c].astype('int8')


# In[ ]:


for c in int16columns:
    train_transaction[f'{c}_isna'] = train_transaction[c].isnull()
    train_transaction[c].fillna(-1.0, inplace=True)
    train_transaction[c] = train_transaction[c].astype('int16')


# In[ ]:


new_memory_usage = train_transaction.memory_usage()
new_memory_usage


# In[ ]:


total_new_memory_usage = new_memory_usage.sum()
total_new_memory_usage


# In[ ]:


total_original_memory_usage


# In[ ]:


total_original_memory_usage - total_new_memory_usage


# So, we shaved about a gigabyte of RAM from this. You can now save your processed file to a feather format, for further processing.

# In[ ]:


train_transaction.to_feather('train_transaction_reduced_memory')

