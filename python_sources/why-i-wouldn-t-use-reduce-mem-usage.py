#!/usr/bin/env python
# coding: utf-8

# Whenever the size of dataset goes above 1.5GB there are some memory issues when working with Kaggle Kernels, particlarly when you want to fit everything in one kernel. In the public kernls of this competition I saw a very commonly used function `reduce_mem_usage()` to reduce memory usage introduced in [here](https://www.kaggle.com/mjbahmani/reducing-memory-size-for-ieee) that is basically using the function first introduced in [here](https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65). The same (or very similar function) was being used in Predicting Molecular Properties competition as well. 
# 
# As cool as it seems to use this function, it is not the best idea to use a function blindly. First of all, this function automatically fills in your null values for you! that is not exactly what you asked for and is actually a big deal. Moreover, there are some hidden pitfalls in using that function as described in [here](https://www.kaggle.com/c/champs-scalar-coupling/discussion/96655#latest-566225). I think that is the reason why Pandas (with all of its genious developers) doesn't have this basic function built-in.
# 
# This kernel runs a simple check on one column to see if `reduce_mem_usage()` results in percision loss.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
          
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(df[col]).all(): 
                NAlist.append(col)
                df[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = df[col].fillna(0).astype(np.int64)
            result = (df[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        df[col] = df[col].astype(np.uint8)
                    elif mx < 65535:
                        df[col] = df[col].astype(np.uint16)
                    elif mx < 4294967295:
                        df[col] = df[col].astype(np.uint32)
                    else:
                        df[col] = df[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)    
            # Make float datatypes 32 bit
            else:
                df[col] = df[col].astype(np.float32)
            
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist

train = pd.read_csv('../input/train_transaction.csv')
train1, NAlist = reduce_mem_usage(train)


# In[ ]:


train = pd.read_csv('../input/train_transaction.csv')

for i in range(len(train)):
    if not np.isnan(train.V314[i]):
        if train.V314[i]!=train1.V314[i]:
            print(i, train.V314[i], train1.V314[i])
    if i > 1000:
        break


# ## Turned out it does
# * Fill in your `NaN` values for you that is not what exactly what you asked for.
# * Result in percision loss in some columns
# * Probably result in other hidden bugs that are not easily detectable

# In[ ]:


# By the way did you know that np.nan != np.nan?
np.nan == np.nan

