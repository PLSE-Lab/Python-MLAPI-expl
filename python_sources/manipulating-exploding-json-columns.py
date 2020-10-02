#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Import json packages to explode json columns like related_same_day_brand
import json
from pandas.io.json import json_normalize

# Any results you write to the current directory are saved as output.


# In[2]:


pd.set_option('display.max_colwidth', -1)
nRowsRead = 1000 # specify 'None' if want to read whole file
df = pd.read_csv('../input/cbg_patterns.csv', delimiter=',', nrows = nRowsRead)
df.head(3)


# In[3]:


df.describe()


# In[4]:


parsed_days = df['popularity_by_day'].apply(lambda x: json.loads(x))
parsed_days[:3]


# In[6]:


parsed_days = json_normalize(parsed_days)
parsed_days.head(3)


# In[8]:


parsed_days = parsed_days[['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']]
parsed_days.head(3)


# In[9]:


df.merge(parsed_days, left_index=True, right_index=True).head(3)


# In[10]:


parsed_visitor_home_cbgs = df['visitor_home_cbgs'].apply(lambda x: json.loads(x))
parsed_visitor_home_cbgs[:3]


# In[11]:


parsed_visitor_home_cbgs = json_normalize(parsed_visitor_home_cbgs)
parsed_visitor_home_cbgs.head(3)


# In[12]:


df.merge(parsed_visitor_home_cbgs, left_index=True, right_index=True).head(3)


# In[ ]:




