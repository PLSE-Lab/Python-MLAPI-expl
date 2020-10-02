#!/usr/bin/env python
# coding: utf-8

# In[11]:


import math
import seaborn as sns
sns.set(style="whitegrid", color_codes=True)


from wordcloud import WordCloud, STOPWORDS

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # linear algebra
import matplotlib 
import matplotlib.pyplot as plt
import sklearn
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.rcParams["figure.figsize"] = [16, 12]
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory


# In[2]:


# import package with helper functions 
import bq_helper

# create a helper object for this dataset
baseball = bq_helper.BigQueryHelper(active_project="bigquery-public-data",
                                   dataset_name="baseball")

# print all the tables in this dataset (there's only one!)
baseball.list_tables()


# In[3]:


baseball.table_schema('schedules')


# In[4]:


baseball.table_schema('games_wide')


# In[5]:


baseball.table_schema('games_post_wide')


# In[19]:


# Your Code Goes Here
query = """SELECT extract(month from startTime) as month, dayNight, avg(attendance) as attendance, avg(duration_minutes) as duration_minutes
            FROM `bigquery-public-data.baseball.schedules`
            GROUP BY dayNight, extract(month from startTime)
            """
res = baseball.query_to_pandas_safe(query)


# In[20]:


res.head()


# In[29]:


sns.barplot(x = 'month', y = 'attendance', hue = 'dayNight', data = res)


# More attendance during days than night. The difference decreases for July and October(why? because of (semi-)finals?).
# Nighttime attendance is smallest during April, May, and September. 

# In[31]:


sns.barplot(x = 'month', y = 'duration_minutes', hue = 'dayNight', data = res)


# Game duration has little dependence on day/night or month. However October games tend to be shorter. 
