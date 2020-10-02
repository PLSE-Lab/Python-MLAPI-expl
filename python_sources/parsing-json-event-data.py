#!/usr/bin/env python
# coding: utf-8

# **Small hint on parsing json, I hope it will be useful.**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from pandas.io.json import json_normalize
import json


# In[ ]:


test = pd.read_csv('/kaggle/input/data-science-bowl-2019/test.csv')
test = test.head(20) # limiting number of rows for speeding up the proccess (it's just a demo)


# In[ ]:


test.shape


# it is 11 columns in the beginning, but 'event_data' needs to be parsed

# In[ ]:


test.head()


# Function to copy-paste

# In[ ]:


def json_parser(dataframe, column):
    parsed_set = dataframe[column].apply(json.loads)
    parsed_set = json_normalize(parsed_set)
    
    merged_set = pd.merge(
        test, 
        parsed_set,
    
        how='inner', 
        left_index=True, 
        right_index=True
    )
    
    del merged_set[column]
   
    return merged_set


# In[ ]:


parsed_df = json_parser(test, 'event_data')

parsed_df.shape


# it is now 30 columns (this number can rise up due to we have taken only first 20 rows)

# In[ ]:


parsed_df.head()


# Feel free to upvote :)
