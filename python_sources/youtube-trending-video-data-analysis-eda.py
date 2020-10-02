#!/usr/bin/env python
# coding: utf-8

# In[10]:


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


# ### import data 

# create data frame by read csv file with index col video_id 

# In[17]:


us_df = pd.read_csv('../input/USvideos.csv', index_col='video_id')


# In[18]:


us_df.shape


# In[19]:


us_df.info()


# read  json data for video category and create a dictionary which map category_id with category.

# In[20]:


import json

id_category_dict = {}

with open('../input/FR_category_id.json', 'r') as f:
    content = json.load(f)
    for item in content['items']:
        id_category_dict[item['id']] = item['snippet']['title']

id_category_dict


# ### Preprocessing on data

# In[21]:


us_df.head()


# In[25]:


us_df[['trending_date', 'publish_time']].head()


# change trending date formate as same as publish time

# In[27]:



us_df['trending_date'] = pd.to_datetime(us_df['trending_date'], format='%y.%d.%m')
us_df['trending_date'].head()


# next step to divide publish_time column into pusblish_date & publish_time 

# In[28]:


us_df['publish_time'] = pd.to_datetime(us_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
us_df['publish_time'].head()


# In[29]:


#DataFrame.insert(loc, column, value, allow_duplicates=False)[source]
us_df.insert(3, 'publish_date', us_df['publish_time'].dt.date)
us_df['publish_time'] = us_df['publish_time'].dt.time


# In[30]:


us_df[['publish_time', 'publish_date', 'trending_date']].head()


# In[34]:


us_df.head(2)


# insert new column for categoryin dataframe.
# convert data type of category_id into string

# In[36]:



for column in ['category_id']:
    us_df[column] = us_df[column].astype(str)


# In[37]:


us_df.insert(3, 'category', us_df['category_id'].map(id_category_dict))


# In[38]:


us_df[['category_id', 'category']].head()


# In[39]:


us_df.head(2)


# ### visualization on data

# #### most Trendeing Category visualization

# In[40]:


get_ipython().run_line_magic('matplotlib', 'inline')
us_df.category.value_counts().plot(kind='bar', title='Most trending category')


# In[ ]:




