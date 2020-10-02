#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


dc = pd.read_csv("../input/fivethirtyeight-comic-characters-dataset/dc-wikia-data.csv")
marvel = pd.read_csv("../input/fivethirtyeight-comic-characters-dataset/marvel-wikia-data.csv")


# In[ ]:


dc.head()


# In[ ]:


marvel.head()


# In[ ]:


import plotly.express as px


# In[ ]:


df = px.data.tips()
df.head()


# In[ ]:


from pandasql import sqldf
pysqldf = lambda q: sqldf(q, globals())


# In[ ]:


marvel_filter = marvel.drop(['GSM'], axis = 1).dropna()
dc_filter = dc.drop(['GSM'], axis = 1).dropna()
q = """select 'marvel' as Universe,SEX, 
count(*) as count from marvel_filter
group by 1,2"""
q = """select 'marvel' as universe, SEX, EYE from marvel_filter"""
marvel_filter = pysqldf(q)
q = """select 'dc' as Universe,SEX, 
count(*) as count from dc_filter
group by 1,2"""
q = """select 'dc' as universe, SEX, EYE from dc_filter"""
dc_filter = pysqldf(q)


# In[ ]:


merged_filter = pd.concat([marvel_filter, dc_filter], axis = 0, ignore_index = True)
merged_filter.columns


# In[ ]:


merged_filter.head()


# In[ ]:


fig = px.parallel_categories(merged_filter, dimensions=['universe', 'SEX'])
fig.show()


# In[ ]:


marvel_filter = marvel.drop(['GSM'], axis = 1).dropna()
dc_filter = dc.drop(['GSM'], axis = 1).dropna()
q = """select 'marvel' as Universe,SEX, EYE,
count(*) as count from marvel_filter
group by 1,2,3"""
# q = """select 'marvel' as universe, SEX, EYE from marvel_filter"""
marvel_filter = pysqldf(q)
q = """select 'dc' as Universe,SEX, EYE,
count(*) as count from dc_filter
group by 1,2,3"""
# q = """select 'dc' as universe, SEX, EYE from dc_filter"""
dc_filter = pysqldf(q)
merged_filter = pd.concat([marvel_filter, dc_filter], axis = 0, ignore_index = True)
merged_filter.head()


# In[ ]:


px.scatter(merged_filter,x = "SEX", y="EYE", size = "count", color = "Universe")


# In[ ]:




