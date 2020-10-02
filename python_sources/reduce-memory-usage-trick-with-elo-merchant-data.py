#!/usr/bin/env python
# coding: utf-8

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


hist_df = pd.read_csv('../input/historical_transactions.csv')
hist_df.head()


# In[ ]:


hist_df.size


# ### History data file size is quite large. even preview of this file is not supported. 
# 
# **File size is 2.65 GB**
# 
# ### How can we preview data? or read just a chunk of data?
# 
# Loading through panda dataframe read_csv function , occupies 9.7GB space. it's very costly at the middle of the program when you have already load couple of data and run out of memory.
# 
# Here are two options:
# ### 1) Read in a chunk 
# 
# Get idea or preview of data
# 
# ### 2) Reduce memory usage.
# 
# * Load objects as categories
# * Binary values are switched to int8
# * Binary values with missing values are switched to float16 (int does not understand nan)
# * 64 bits encoding are all switched to 32 or 16bits if possible.

# ### 1) Read in a chunk

# In[ ]:


history_reader = pd.read_csv('../input/historical_transactions.csv', chunksize = 10)
type(history_reader)

hist_chunk = None
for chunk in history_reader:
    hist_chunk = chunk
    print(hist_chunk)
    break


# In[ ]:


type(hist_chunk)


# Wow! with very less memory usage we can preview our data, can read column names. check the data types of each column and can reduce memory to load the whole file.

# In[ ]:


history_columns = list(hist_chunk.columns)
print(history_columns)


# In[ ]:


hist_chunk.dtypes


# Ref:[ Ashish Patel's kernel](http://www.kaggle.com/ashishpatel26/lightgbm-goss-dart-parameter-tuning) 
# ### 2) Reduce memory usage.
# 
# * Load objects as categories
# * Binary values are switched to int8
# * Binary values with missing values are switched to float16 (int does not understand nan)
# * 64 bits encoding are all switched to 32 or 16bits if possible.

# In[ ]:


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if df[col].dtypes == 'object':
            df[col] = df[col].astype('category')
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


import gc

historical_transactions = reduce_mem_usage(hist_df)
gc.collect()


# Wow!  Reading the same file just took 1GB memory space instad of reading original file took 9GB space. That is a magic of converting data types.

# In[ ]:


historical_transactions.dtypes


# In[ ]:


historical_transactions.head()

