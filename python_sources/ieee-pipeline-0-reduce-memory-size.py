#!/usr/bin/env python
# coding: utf-8

# Input - Competition Data
# Output - Shortened transaction + Identity files for train & test 
# Next kernel - https://www.kaggle.com/priteshshrivastava/ieee-pipeline-1-create-validation-set

#  #  <div style="text-align: center">  Reducing  Memory Size for IEEE </div> 
#  <div style="text-align:center">  </div>
# ![mem](http://s8.picofile.com/file/8367719234/mem.png) 
# <div style="text-align:center"> last update: <b> 19/07/2019</b></div>
# 

# ## Objective of the Kernel: Save Time & Memory
# If you would like to create a kernel for this Competition. this is a good idea to add this kernel as a **data set** to your own kernel. due to you can save your time and memory.

# ___MEMORY USAGE  BEFORE AND AFTER COMPLETION FOR TRAIN:___
# <br/>
# Memory usage before running this script : 1975.3707885742188  MB
# <br/>
# Memory usage after running this script  : ~ **480  MB**
# <br/>
# This is ~ 28 % of the initial size

# 
# ___MEMORY USAGE  BEFORE AND AFTER COMPLETION FOR TEST:___
# <br/>
# Memory usage before running this script : 1693.867820739746  MB
# <br/>
# Memory usage after running this script: ~ **480  MB**
# <br/>
# This is ~  28  % of the initial size

# ## Import

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import pickle
import time
import gc
gc.enable()
import warnings
warnings.filterwarnings("ignore")


# What do we have in input

# In[ ]:


print(os.listdir("../input"))
get_ipython().system('ls -GFlash  ../input')


# ## Import Dataset to play with it

# In[ ]:


get_ipython().run_cell_magic('time', '', '# import Dataset to play with it\ntrain_identity= pd.read_csv("../input/train_identity.csv", index_col=\'TransactionID\')\ntrain_transaction= pd.read_csv("../input/train_transaction.csv", index_col=\'TransactionID\')\ntest_identity= pd.read_csv("../input/test_identity.csv", index_col=\'TransactionID\')\ntest_transaction = pd.read_csv(\'../input/test_transaction.csv\', index_col=\'TransactionID\')\nprint ("Done!")')


# In[ ]:


print('Shape of Data:')
print(train_transaction.shape)
print(test_transaction.shape)
print(train_identity.shape)
print(test_identity.shape)


# ### Creat our train & test dataset

# In[ ]:


# Creat our train & test dataset
#%%time
train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)


# ### Before Reducing Memory
# When I have just read the data set and join them!I saw that the status of my RAM is more than 9GB!

# ![ram1](http://s9.picofile.com/file/8366931918/ram1.png)

# Then we shoud just delete some dt!

# In[ ]:


del train_identity,train_transaction,test_identity, test_transaction


# ![ram2](http://s8.picofile.com/file/8366932526/ram2.png)
# 3GB of RAM has got free! now just check the size of our train & test

# In[ ]:


train.info()


# In[ ]:


test.info()


# # IEEE Reducing  Memory Size
# It is necessary that after using this code, carefully check the output results for each column.

# In[ ]:


#Based on this great kernel https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65
def reduce_mem_usage(df):
    start_mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in df.columns:
        if df[col].dtype != object:  # Exclude strings            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",df[col].dtype)            
            # make variables for Int, max and min
            IsInt = False
            mx = df[col].max()
            mn = df[col].min()
            print("min for this col: ",mn)
            print("max for this col: ",mx)
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
            
            # Print new column type
            print("dtype after: ",df[col].dtype)
            print("******************************")
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = df.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    return df, NAlist


# Reducing for train data set:

# In[ ]:


train, NAlist = reduce_mem_usage(train)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# Reducing for test data set:

# In[ ]:


test, NAlist = reduce_mem_usage(test)
print("_________________")
print("")
print("Warning: the following columns have missing values filled with 'df['column_name'].min() -1': ")
print("_________________")
print("")
print(NAlist)


# Check again! our RAM. 2 GB has got free!

# ![ram3](http://s8.picofile.com/file/8366940442/ram3.png)

# In[ ]:


train.info()


# In[ ]:


test.info()


# ## Add this kernel as Dataset
# Now we just save our output as csv files. then you can simply add them to your own kernel.you will save time and  memory.

# In[ ]:


train.to_pickle('train.pkl')
test.to_pickle('test.pkl')


# ## How about other ways!
# I have used this [great kernel](https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65) but there are also other ways such as:
# 1. https://www.dataquest.io/blog/pandas-big-data/
# 2. [optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment](https://medium.com/@vincentteyssier/optimizing-the-size-of-a-pandas-dataframe-for-low-memory-environment-5f07db3d72e)
# 3. [pandas-making-dataframe-smaller-faster](https://www.ritchieng.com/pandas-making-dataframe-smaller-faster/)
