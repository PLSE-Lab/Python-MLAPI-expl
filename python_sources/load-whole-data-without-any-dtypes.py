#!/usr/bin/env python
# coding: utf-8

# Small manual to load whole data and reduce memory usage in Malware challenge. 
# This method is inspired from this [kernel](https://www.kaggle.com/gemartin/load-data-reduce-memory-usage).

# In[ ]:


import pandas as pd
import numpy as np
import gc


# In[ ]:


def df_reader(path, chunksize):
    reader = train_reader = pd.read_csv(path, chunksize=chunksize)
    dflist = []
    for df_part in reader:
        dflist.append(df_part)
    data = pd.concat(dflist,sort=False)
    return data

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
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
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = df_reader('../input/train.csv', 100000)\nprint(train.shape)\ntrain = reduce_mem_usage(train)")


# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "test = df_reader('../input/test.csv', 100000)\nprint(test.shape)\ntest = reduce_mem_usage(test)")


# In[ ]:


gc.collect()

