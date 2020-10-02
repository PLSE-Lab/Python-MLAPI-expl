#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# Learning Python with the given data ....

# In[ ]:


# Importing Various Library 
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[ ]:


print(os.listdir('../input'))


# In[ ]:


# nRowsRead = 2000 # specify 'None' if want to read whole file
# clash-of-clans.csv has 50001 rows in reality, but we are only loading/previewing the first 1000 rows
#df1 = pd.read_csv('../input/clash-of-clans.csv', delimiter=',', nrows = nRowsRead)
df1 = pd.read_csv('../input/clash-of-clans.csv', delimiter=',')
df1.dataframeName = 'clash-of-clans.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')


# Let's take a quick look at what the data looks like:

# In[ ]:


df1.head(5)


# Distribution graphs (histogram/bar graph) of sampled columns:

# In[ ]:


df1.info()


# In[ ]:


df1.shape


# In[ ]:


df1.size


# In[ ]:


len(df1)


# In[ ]:


# Use the count method to find the number of non-missing values for each column.
df1.count()


# In[ ]:


#The describe method is very powerful and calculates all the descriptive 
# statistics and quartiles in the preceding steps all at once
df1.describe()


# In[ ]:


#To get a count of the missing values
df1.isnull().sum().sum()


# So we have some Name which are Blank in our data set

# In[ ]:


df1.info()


# In[ ]:


df1.describe(include=[np.number]).T


# In[ ]:


df1.describe(include=[np.object, pd.Categorical]).T


# In[ ]:


# Inspect the data types of each column:
df1.dtypes


# In[ ]:


# Find the memory usage of each column with the memory_usage method
original_mem = df1.memory_usage(deep=True)
original_mem


# In[ ]:


'''There is no need to use 64 bits for the Rating column as it contains only 1 - 5 values. 
Let's convert this column to an 8-bit (1 byte) integer with the astype method:'''
df1['Rating'] = df1['Rating'].astype(np.int8)


# In[ ]:


# Use the dtypes attribute to confirm the data type change:
df1.dtypes


# In[ ]:


# Find the memory usage of each column again and note the large reduction
df1.memory_usage(deep=True)


# In[ ]:


'''To save even more memory, you will want to consider changing object data types
to categorical if they have a reasonably low cardinality (number of unique
values). Let's first check the number of unique values for both the object columns:'''
df1.select_dtypes(include=['object']).nunique()


# In[ ]:


'''The Date column is a good candidate to convert to Categorical as less than one
percent of its values are unique:'''
df1['Date'] = df1['Date'].astype('category')
df1.dtypes


# In[ ]:


# Compute the memory usage again:
new_mem = df1.memory_usage(deep=True)
new_mem


# In[ ]:


# let's compare the original memory usage with our updated memory
new_mem / original_mem


# In[ ]:




