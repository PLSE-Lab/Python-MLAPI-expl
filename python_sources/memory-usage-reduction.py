#!/usr/bin/env python
# coding: utf-8

# #  **REDUCING MEMORY USAGE OF DATASETS BY MORE THAN 60%**

# *In this example we will be using the M5 Forecasting Accuracy dataset by walmart which consists of a collection of datasets which make up to nearly 1.5GB in total.*

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


price = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


# In[ ]:


price.info(memory_usage = 'deep')


# But if you look close enough most of the column has a dtype "Object" type so lets explore the memory usage of different datatypes before we jump into the solution 

# In[ ]:


price.memory_usage(deep = True) * 1e-6    #for MB representation


# In[ ]:


price.head(2)


# This data clearly shows that the columns "store_id" and "item_id" occupy the most amount of memory. Looking into these columns we can clearly see that these are categorical data and hence converting them to their relevent dtype will efficiently reduce the memory usage.

# In[ ]:


price[['store_id','item_id']] = price[['store_id','item_id']].astype('category')


# In[ ]:


price.memory_usage(deep = True) * 1e-6   #for MB representation


# Clearly we can see that converting the dtype "Object" to "Categorical" has reduced the memory size by a whopping 60 times. Similarly converting the dtypes from from higher bit representation to their lower bit counterparts will hugely leverage our goal to decrease the memory usage of the dataset by more that 60%. To achieve this I will be using a function that I learnt from https://guillaume-martin.github.io/ which efficiently changes the dtypes of a whole dataframe

# In[ ]:


import numpy as np
def reduce_mem_usage(props):
    start_mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    NAlist = [] # Keeps track of columns that have missing values filled in. 
    for col in props.columns:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
            print("******************************")
            print("Column: ",col)
            print("dtype before: ",props[col].dtype)
            
            # make variables for Int, max and min
            IsInt = False
            mx = props[col].max()
            mn = props[col].min()
            
            # Integer does not support NA, therefore, NA needs to be filled
            if not np.isfinite(props[col]).all(): 
                NAlist.append(col)
                props[col].fillna(mn-1,inplace=True)  
                   
            # test if column can be converted to an integer
            asint = props[col].fillna(0).astype(np.int64)
            result = (props[col] - asint)
            result = result.sum()
            if result > -0.01 and result < 0.01:
                IsInt = True

            
            # Make Integer/unsigned Integer datatypes
            if IsInt:
                if mn >= 0:
                    if mx < 255:
                        props[col] = props[col].astype(np.uint8)
                    elif mx < 65535:
                        props[col] = props[col].astype(np.uint16)
                    elif mx < 4294967295:
                        props[col] = props[col].astype(np.uint32)
                    else:
                        props[col] = props[col].astype(np.uint64)
                else:
                    if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                        props[col] = props[col].astype(np.int8)
                    elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                        props[col] = props[col].astype(np.int16)
                    elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                        props[col] = props[col].astype(np.int32)
                    elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                        props[col] = props[col].astype(np.int64)    
            
            # Make float datatypes 32 bit
            else:
                props[col] = props[col].astype(np.float32)
            
            # Print new column type
            print("dtype after: ",props[col].dtype)
            print("******************************")
    
    # Print final result
    print("___MEMORY USAGE AFTER COMPLETION:___")
    mem_usg = props.memory_usage().sum() / 1024**2 
    print("Memory usage is: ",mem_usg," MB")
    print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")


# In[ ]:


prices = pd.read_csv('/kaggle/input/m5-forecasting-accuracy/sell_prices.csv')


# In[ ]:


prices.info(memory_usage = 'deep')


# In[ ]:


reduce_mem_usage(prices)


# In[ ]:


prices.info()


# In[ ]:


prices[['store_id','item_id']] = prices[['store_id','item_id']].astype('category')


# In[ ]:


prices.info(memory_usage = 'deep')


# We can see that we managed to reduce the memory usage of "sell_prices.csv" from 957.5MB to 59MB with all its data unchanged and untouched
