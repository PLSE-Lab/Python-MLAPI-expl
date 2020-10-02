#!/usr/bin/env python
# coding: utf-8

# <p>This notebook demonstrates the thinking process of preparing a complex dateset (nested data structure) for further analysis.
# <p>Key techniques:
#     * ast.literal_eval()
#     * pd.DataFrame.from_dict()
#     * df.merge()
#     
# > "[{'id': 313576, 'name': 'Hot Tub Time Machine Collection', 'poster_path': '/iEhb00TGPucF0b4joM1ieyY026U.jpg', 'backdrop_path': '/noeTVcgpBiD48fDjFVic1Vz7ope.jpg'}]"
# 

# # Data Import

# In[ ]:


import pandas as pd
import ast


# In[ ]:


# Variables
PATH_INPUT = "/kaggle/input/"
PATH_WORKING = "/kaggle/working/"
PATH_TMP = "/tmp/"


# In[ ]:


# Reading data into a df
df_raw = pd.read_csv(f'{PATH_INPUT}train.csv', low_memory=False, skipinitialspace=True)
df_raw.shape


# In[ ]:


# Take a look at the first 10 rows
df_raw.head(10)


# # Data Preparation

# Some columns contain data which look like `dict`. Let's see how we can parse them

# In[ ]:


# define columns with data of dict type to process
cols = ['belongs_to_collection', 'genres', 'production_companies', 'spoken_languages', 'Keywords', 'cast', 'crew']


# In[ ]:


# check the data type
df_raw[cols].dtypes


# ## Trial with the first column

# Looks like the columns are `string`. See how we can parse the column

# In[ ]:


# copy the column to a pandas series
s = df_raw[cols[0]].copy()
s.shape


# In[ ]:


# check the first record
s[0]


# In[ ]:


# evaluate as a list
l = ast.literal_eval(s[0])
l


# In[ ]:


# check the data type
print(type(l), type(l[0]))


# Looks good. Let's try the 3rd row with NaN value

# In[ ]:


s[3]


# literal_eval(nan) will return an error. Replace with an empty dict in a list wrapped as str `'[{}]'`

# Let's put the steps together and parse a single column

# In[ ]:


# copy one column to a pandas series
s = df_raw[cols[0]].copy()
# fillna with [None]
s.fillna('[{}]', inplace=True)

l = []  # init an empty list

for i in s:
    if i == [{}]:
        # append [{}] to the list
        l += i
    else:
        # evaluate as a list
        l += ast.literal_eval(i)


# In[ ]:


len(l)  # should be 3000 if processed correctly


# In[ ]:


l[:10]


# Note that `nan` are processed as empty `dict`

# In[ ]:


for i in range(10):
    print(type(l[i]))


# Looks good. See how we can make a df from the list of dict

# In[ ]:


df_tmp = pd.DataFrame.from_dict(l)
df_tmp[:10]


# # Rewrite as functions

# In[ ]:


def to_list_of_dict(series):
    """
    Evaluate a pandas series as a list of dict
    
    Input:
    "[{'one': 1, 'two': 2, 'three': 3}]"
    
    Output:
    [{'one': 1,
      'two': 2,
      'three' : 3}]
    """
    l = []  # init an empty list
    s = series.fillna('[{}]')  # map nan to [{}] for further eval
    
    # loop through the whole series
    for i in s:
        if i == [{}]:
            # append [{}] to the list
            l += i
        else:
            # evaluate as a list
            l += ast.literal_eval(i)
    
    return l


# In[ ]:


def column_conversion(col, df):
    """
    Merge a pandas series with data like list of dict back to the dataframe
    
    Input:
    "[{'one': 1, 'two': 2, 'three': 3}]"
    
    Output:
    A dataframe with the original column removed, each dict's key in a new column
    """
    l = to_list_of_dict(df[col])  # convert to list of dict
    df_right = pd.DataFrame.from_dict(l)  # convert to df
    df_merged = df.merge(df_right.add_prefix(col+'_'),  # add the original column name as prefix
                         left_index=True, right_index=True)  # merge df with df_right
    df_merged.drop(col, axis=1, inplace=True) # drop the original column
    
    return df_merged


# In[ ]:


# Test
column_conversion(cols[0], df_raw)[:3]


# ## Process all columns at once

# In[ ]:


# check the columns to process
cols


# In[ ]:


# make a copy
df = df_raw.copy()

# process the columns one by one
for col in cols:
    df = column_conversion(col, df)


# In[ ]:


# check the first record
df[:1]


# Cool! The dataframe is now flattened for further analysis.
