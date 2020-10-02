#!/usr/bin/env python
# coding: utf-8

# ### Preprocessing of 29 million transaction records
# 
# The dataset provides access to 2 files. One of the files `new_merchant_transactions.csv` contains over 29million records. Pandas is considered as a great choice for solving data science problems. It provides some unique features like automatically detecting the datatype of attributes but it's not that efficient. Here comes the role of data engineers to correctly cast datatype for each of the attribute. 
# 
# In this notebook, I will be sharing my approach of reducing the size of a pandas df for further Machine Learning/Deep Learning implementation on it. 
# I achieved reduction in the dataframe's memory usage to around 8% of original size. 

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


# ### Let's read the historical_transactions
# - Read the csv as a pandas dataframe using `read_csv()` function of pandas
# - Get info about each of the attribute of that dataframe along with memory usage using `info()` function.

# In[ ]:


historical_tx_df = pd.read_csv('../input/historical_transactions.csv')


# In[ ]:


historical_tx_df.info(memory_usage='deep')


# In[ ]:


for dtype in ['float','int','object']:
    selected_dtype = historical_tx_df.select_dtypes(include=[dtype])
    mean_usage_b = selected_dtype.memory_usage(deep=True).mean()
    mean_usage_mb = mean_usage_b / 1024 ** 2
    print("Average memory usage for {} columns: {:03.2f} MB".format(dtype,mean_usage_mb))


# In[ ]:


def mem_usage(pandas_obj):
    if isinstance(pandas_obj,pd.DataFrame):
        usage_b = pandas_obj.memory_usage(deep=True).sum()
    else: # we assume if not a df it's a series
        usage_b = pandas_obj.memory_usage(deep=True)
    usage_mb = usage_b / 1024 ** 2 # convert bytes to megabytes
    return "{:03.2f} MB".format(usage_mb)


# In[ ]:


'''
Function to count unique values in the columsn. 

Arguments:
df_columns: List of columns in `his_int` dataframe
'''

def count_unique( df_columns):
    for i in df_columns:
        print("Number of unique vals in",i,"are",len(his_int[i].unique()))


# ### Preprocessing of Integer columns 
# The dataframe reads these columns as integer `city_id, installments, merchant_category_id, month_lag, state_id,subsector_id`. As you can see most of the columns are contain specific ids. These can be mapped to categorical data. 
# 
# In the next couple of code cells, I will be downcasting the integer columns to category. 

# In[ ]:


his_int = historical_tx_df.select_dtypes(include=['integer'])


# In[ ]:


count_unique(his_int.columns)


# In[ ]:


converted_obj = pd.DataFrame()

for col in his_int.columns:
    num_unique_values = len(his_int[col].unique())
    num_total_values = len(his_int[col])
    if num_unique_values / num_total_values < 0.5:
        converted_obj.loc[:,col] = his_int[col].astype('category')
    else:
        converted_obj.loc[:,col] = his_int[col]


# The integer columns memory usage is now reduced from 1332 MB to 222 MB.

# In[ ]:


print(mem_usage(his_int))
print(mem_usage(converted_obj))


# In[ ]:


converted_obj.info()


# ### Creation of optimized transaction records dataframe
# We will be creating a new dataframe `optimized_hist_df` and mapping all the processed columns to that dataframe. 

# In[ ]:


optimized_hist_df = pd.DataFrame()


# In[ ]:


optimized_hist_df[converted_obj.columns] = converted_obj
mem_usage(optimized_hist_df)


# In[ ]:


his_int = historical_tx_df.select_dtypes(include=['float'])


# In[ ]:


his_int_na = his_int.category_2.fillna(6)
conv_obj = his_int_na.astype('int')
mem_usage(conv_obj)


# In[ ]:


conv_obj2 = conv_obj.astype('category')
mem_usage(conv_obj2)


# In[ ]:


his_int["category_2"] = conv_obj2


# In[ ]:


optimized_hist_df[his_int.columns] = his_int
mem_usage(optimized_hist_df)


# In[ ]:


his_int = historical_tx_df.select_dtypes(include=['object'])
his_int.head()


# In[ ]:


count_unique(his_int.columns)


# In[ ]:


converted_obj = pd.DataFrame()

for col in his_int.columns:
    num_unique_values = len(his_int[col].unique())
    num_total_values = len(his_int[col])
    if num_unique_values/num_total_values < 0.33:
        converted_obj.loc[:,col] = his_int[col].astype('category')
    else:
        converted_obj.loc[:,col] = his_int[col]


# In[ ]:


print(mem_usage(his_int))
print(mem_usage(converted_obj))


# In[ ]:


optimized_hist_df[converted_obj.columns] = converted_obj
mem_usage(optimized_hist_df)


# In[ ]:


optimized_hist_df['purchase_date'] =  pd.to_datetime(historical_tx_df.purchase_date, format='%Y%m%d %H:%M:%S')
mem_usage(optimized_hist_df)


# In[ ]:


optimized_hist_df.head()


# ## Reduction in the size of our dataframe

# In[ ]:


per_df_red = (float(mem_usage(optimized_hist_df).split()[0]) / float(mem_usage(historical_tx_df).split()[0]) )*100
per_df_red


# **As per the above computation, you can clearly see that we have reduced the 13+GB pandas dataframe to 1064MB dataframe. This leads to a reduction in size to around 8%.**

# In[ ]:




