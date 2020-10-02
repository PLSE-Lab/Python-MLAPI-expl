#!/usr/bin/env python
# coding: utf-8

# # Play with HN Data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/HN_posts_year_to_Sep_26_2016.csv',parse_dates=['created_at'])
print(df.head(3))


# ## Top 10 most upvoted posts ##

# In[ ]:


df[['title', 'num_points']].sort_values(by='num_points', ascending=False).head(10)


# ## Top 10 domains with the most upvoted posts ##

# In[ ]:


import re

# Do some basic sanitation
df1 = df.copy().dropna(subset=['url'])

df1['domain'] = df1['url'].str.extract('https?://([\w\.]+)/?.*', flags=re.IGNORECASE, expand=False)
df1_domain = df1[['domain', 'num_points']].groupby('domain')
df1_domain.sum().sort_values(by='num_points', ascending=False).head(10).reset_index()


# ## What is the best time to post? ##

# In[ ]:


df['hour'] = df['created_at'].dt.hour
df1 = df.groupby('hour')
df1['num_points'].mean().sort_values(ascending=False).head(5)


# ## How many posts are related to Python? ##

# In[ ]:


len(df[df['title'].str.contains("python") | df['title'].str.contains("Python")].index)

