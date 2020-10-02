#!/usr/bin/env python
# coding: utf-8

# Hi, in this kernel, I will show you how to reduce the memory usage of the train dataset by 65% percent.

# # 1. Readin data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_i = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
train_t = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')


# In[ ]:


train = pd.merge(left=train_t, right=train_i, on='TransactionID', how='left')

print(train.shape)


# In[ ]:


del train_i, train_t
gc.collect()


# First, let's check the memory usage before we implement reduction.

# In[ ]:


print(train.info(memory_usage='deep'))


# The `train` dataset has 399 columns with type 'float64', 4 columns with type 'int64', and 31 columns with type 'object'. The total usage of memory is 2.5 GB.

# # 2. Reduce memory usage
# ## 2.1 Memory usage of different types and subtypes
# As you may know, pandas groups the columns of the same types into blocks of values (NumPy arrays). 
# ![](https://cdn.shortpixel.ai/client/to_webp,q_glossy,ret_img/https://www.dataquest.io/wp-content/uploads/2019/01/df_blocks.png)
# *image source: https://www.dataquest.io/blog/pandas-big-data/*
# 
# These blocks are optimized for storing the actual values in the dataframe. The [BlockManager](https://kite.com/python/docs/pandas.core.internals.BlockManager) class is responsible for maintaining the mapping between the row and column indexes and the actual blocks.
# 
# Different types have different subtypes, which takes the different size of memory, e.g., `float` type has three subtypes: `float16`, `float32`, `float64`. The subtypes are named as the **type** part combined with the **usage of byte** part. So, `float64` is the subtype of `float` that takes 64 byte (8 bytes).
# All subtypes and the corresponding bytes usage are listed here:
# * 1 bytes subtypes: int8,  uint8, **bool**
# * 2 bytes subtypes: int16, uint16, float16
# * 4 bytes subtypes: int32, uint32, float32
# * 8 bytes subtypes: int64, uint64, float64, **datetime64**
# 
# The `object` type is special. It represents values using Python string objects. Each element in an  `object` columns is really a pointer that contains the "address" for the actual value's location in memory. And in my opinion, the total usage of each element is partially decided by the length of the element. That makes the memory usage of `object` type variable.
# 
# A diagram showing how numeric data (stored in NumPy) and string data (stored in Python's buildin types) is attached here.
# ![](http://jakevdp.github.io/images/array_vs_list.png)
# *image source: https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/*
# 
# 
# #### **What will we do?** 
# * For `int` and `float` columns, we will check each column and reset the best subtype to it. 
# * For `object` columns, we will check if the column only contains a limited set of values. If so, we convert this column to `category` type, since this type uses integer values under the hood to represent the values in a column and a separate mapping dictionary that maps the integer values to the raw ones. Note that if the number of unique values in this columns is more than 50% of the total number of values, converting to `category` may consume more memory.
#     * You may choose to skip this process for `object` columns, depending on the method you will use to analyze the data.
# 
# First, let's check the memory usage by type

# In[ ]:


# Check memory usage by type
for dtype in ['float','int','object']:
    selected_dtype = train.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024**2
    print('Average memory usage for {} columns: {:03.2f} MB'.format(dtype, mean_usage_mb))


# Let's define a function to calculate memory usage.

# In[ ]:


def mem_usage(pandas_obj):
    if isinstance(pandas_obj, pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else:
        usage_b = pandas_obj.memory_usage(deep=True)
    
    usage_mb = usage_b / 1024**2
    return "{:03.2f} MB".format(usage_mb)


# ## 2.2 `int` type
# * Use the `numpy.iinfo` class to verify the minimum and maximum values for each integer subtype.

# In[ ]:


int_types = ['uint8', 'int8', 'int16', 'int32']
for it in int_types:
    print(np.iinfo(it))


# * Use `pd.DataFrame.select_dtypes()` function to select columns with certain types;
# * Use `pd.to_numeric` with `downcast` parameter to downcast to the smallest subtype that could contain all values of this column. That is, for a `int` column, if the minimun is larger than 0 and the maximum value is smaller than 255, the smallest subtype will be `uint8`. 

# In[ ]:


train_int = train.select_dtypes(include=['int'])
converted_int = train_int.apply(pd.to_numeric, downcast='unsigned')


# In[ ]:


print('`int` type: ')
print('Before:',mem_usage(train_int))
print('After:',mem_usage(converted_int))


# In[ ]:


compare_ints = pd.concat([train_int.dtypes, converted_int.dtypes], axis=1)
compare_ints.columns = ['before', 'after']
compare_ints.apply(pd.Series.value_counts, axis=0)


# ## 2.3 `float` type

# In[ ]:


train_float = train.select_dtypes(include=['float'])
converted_float = train_float.apply(pd.to_numeric, downcast='float')

print('`float` type:')
print('Before:', mem_usage(train_float))
print('After:', mem_usage(converted_float))


# In[ ]:


### 
compare_floats = pd.concat([train_float.dtypes, converted_float.dtypes], axis=1)
compare_floats.columns = ['before', 'after']
compare_floats.apply(pd.Series.value_counts, axis=0)


# ## 2.3 `object` type
# 

# In[ ]:


train_obj = train.select_dtypes(include=['object']).copy()
max_uniq = max(train_obj.apply(pd.Series.nunique))
print('The max unique values in `object` columns is:', max_uniq)
print('The corresponding percentage is: {:02.2f}%'.format(max_uniq/train_obj.shape[0]*100))


# The max unique values in `object` columns is less than 1% of the total value. So, we will convert the columns to `category` tpye.

# In[ ]:


converted_obj = train_obj.apply(pd.Series.astype, dtype='category')

print('`object` type: ')
print('Before:',mem_usage(train_obj))
print('After:',mem_usage(converted_obj))


# # 3. Summary

# In[ ]:


optimized_train = train.copy()
optimized_train[converted_int.columns] = converted_int
optimized_train[converted_float.columns] = converted_float
optimized_train[converted_obj.columns] = converted_obj

print('Before memory reduction:', mem_usage(train))
print('After memory reduction:', mem_usage(optimized_train))


# In[ ]:


del train_int, train_float, train_obj
del converted_int, converted_float, converted_obj
gc.collect()


# So, we have reduced the memory usage of the training data from 2598 MB to 939 MB.
# 
# That's all about this kernel, please let me know if you have any questions. Thanks!
# 
# *Reference: https://www.dataquest.io/blog/pandas-big-data/*
