#!/usr/bin/env python
# coding: utf-8

# In this notebook, I will demonstrate on how to reduce the memory consumption of the Pandas Dataframe which will enable you to carry out the data pre-processing and analysis steps in your less-powerful machine.

# In[ ]:


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


# In[ ]:


merchants = pd.read_csv('../input/merchants.csv', index_col='merchant_id')
merchants.head()


# Lets check out the memory usage of the merchants.csv by pandas dataframeb

# In[ ]:


print(merchants.info())


# Lets make use of the Block Manager to see the data types used across the dataframe

# There are 3 data types class; Integer, Float and Object.
# 
# Lets check out how much each column takes up the memory usage. 
# 
# The float64 type represents each floating point value using 64 bits, or 8 bytes. We can save memory by converting within the same type (from float64 to float32 for example), or by converting between types (from float64 to int32) and so on..

# In[ ]:


print(merchants.memory_usage(deep=True))


# We can see that the DataTypes with object takes up the most memory because Pandas uses pointers to access the string values. 
# 
# Lets first try to reduce float columns by downcasting. 

# In[ ]:


float_cols = merchants.select_dtypes(include=['float'])
#print(float_cols.dtypes)


for cols in float_cols.columns:
	merchants[cols] = pd.to_numeric(merchants[cols], downcast ='float')

print(merchants.info())


# :Tada: Here we can see the reduce of more than 10MB memory usage by just downcasting the float datatype. It can be significant gain seemingly

# In[ ]:


merchants.head()


# In[ ]:


int_cols = merchants.select_dtypes(include=['int'])


for i in int_cols.columns:
	merchants[i] = pd.to_numeric(merchants[i], downcast ='integer')

print(merchants.info())


# In[ ]:


merchants.head()


# Object takes up the most memory. Below I've identified much more appropriate data types for Object type. Pandas has category type which is much more memory efficient. However keep in mind that you may not be able to apply numeric operations on the category variable.

# In[ ]:


for cols in ['category_1', 'category_4', 'most_recent_purchases_range', 'most_recent_sales_range']:
	merchants[cols] = merchants[cols].astype('category')

print(merchants.info())


# As you can see from 56.2+ MB we went down to around 19.5MB memory. This should ease the numeric computation for large MB data size most likely

# In[ ]:


print(merchants.memory_usage(deep=True))


# Much better approach might be to encode the categorical variables as integers. That will further reduce the memory usage by the pandas dataframes of merchant.csv. I hope this notebook has been helpful. Let me know your thoughts on comments. 

# In[ ]:




