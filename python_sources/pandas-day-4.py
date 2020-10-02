#!/usr/bin/env python
# coding: utf-8

# # Pandas tutorial : Day 4
# Here's what we are going to do today : 
# * [Sorting data](#1)
# * [Renaming columns](#2)
# * [Defining a new column](#3)
# * [Changing index name](#4)
# * [Making all columns lowercase](#5)
# * [Making all columns uppercase](#6)
# * [Using Groupby](#7)
# 
# Let's get started!
# 
# [Data for daily news for stop market prediction](https://www.kaggle.com/aaron7sun/stocknews)

# In[ ]:


# import libraries
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


# load data
df = pd.read_csv('/kaggle/input/stocknews/upload_DJIA_table.csv')


# ## Sorting data<a id='1'></a>
# **Method 1 (sort by single column) :** `df.sort_values(by = ['column_name'], ascending = False/True)`
# 
# The default of sorting is `ascending=True` i.e from low to high. But if you want sorting in decending order i.e from high to low make `ascending=False`
# 
# **Method 2 (sort by multiple column) :** `df.sort_values(by = ['column_name1, column_name2,...'], ascending = False/True)`
# 
# **Method 3 :** `df.sort_index()`
# 
# This will sort dataframe index from low to high.

# In[ ]:


df.sort_values(by = ['Date'], ascending = False)


# In[ ]:


df.sort_values(by = ['Open', 'Close'], ascending = False)


# In[ ]:


df.sort_index()


# ## Renaming columns<a id='2'></a>
# Syntax : `df.rename(columns = {'Old_name' : 'New_name', inplace = True})`
# 
# `inplace = True` will permanently overwrite the dataset. By default `inplace = False`

# In[ ]:


df.rename(columns= {'Date' : 'new_date'}).head()


# In[ ]:


# we will make the data as it is by again renaming new_date to Date
df.rename(columns= {'new_date' : 'Date'}).head(1)


# ## Defining a new column<a id='3'></a>
# If you want to make you own column you can do in this way
# 
# Syntax : `df['new_column_name'] = userdefine_operation`

# In[ ]:


df['Difference'] = df.High - df.Low
df.head()


# In this way we can design our own custom columns.

# ## Changing index name<a id='4'></a>

# In[ ]:


# check for the current name of the index
print(df.index.name)


# In[ ]:


# giving name to the index 
df.index.name = 'index'
df.head()


# ## Making all columns lowercase<a id='5'></a>

# In[ ]:


df.columns = map(str.lower, df.columns)
df.columns


# ## Making all columns uppercase<a id='6'></a>

# In[ ]:


df.columns = map(str.upper, df.columns)
df.columns


# ## Groupby<a id='7'></a>
# Group DataFrame using a mapper or by a Series of columns.
# 
# A groupby operation involves some combination of splitting the object, applying a function, and combining the results. This can be used to group large amounts of data and compute operations on these groups.

# In[ ]:


# let's make a dataframe df2
df2 = pd.DataFrame({'Animal': ['Falcon', 'Falcon',
                              'Parrot', 'Parrot'],
                   'Max Speed': [380., 370., 24., 26.]})
df2


# In[ ]:


# grouping 'Animals' with the mean of their max-speed
df2.groupby(['Animal']).mean()


# ### Hierarchical Indexes
# We can groupby different levels of a hierarchical index using the level parameter:
# 
# Syntax : `df.groupby(level = index)` OR `df.groupby(level='column_name')`

# In[ ]:


# creating a hierarchical index dataframe df3
arrays = [['Falcon', 'Falcon', 'Parrot', 'Parrot'],
          ['Captive', 'Wild', 'Captive', 'Wild']]

index = pd.MultiIndex.from_arrays(arrays, names=('Animal', 'Type'))

df3 = pd.DataFrame({'Max Speed': [390., 350., 30., 20.]},
                  index=index)
df3


# In[ ]:


# grouping based on 'Animal'== (level=0), because it is the first index
df3.groupby(level=0).mean()


# In[ ]:


df3.groupby(level="Type").mean()


# That's all for today! we have learnt sorting of data, renaming of columns and defining new columns. In next tutorial we'll see how to drop data and convert data types.
