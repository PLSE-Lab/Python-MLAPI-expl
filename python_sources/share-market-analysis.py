#!/usr/bin/env python
# coding: utf-8

# # **AXIS BANK SHARES TIME SERIES GRAPH**

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


import pandas as pd
import plotly.graph_objects as go
import plotly.express as px


# In[ ]:


data=pd.read_csv('/kaggle/input/stock-market-india/FullDataCsv/AXISBANK__EQ__NSE__NSE__MINUTE.csv')


# **BASIC INFORMATION OF THE DATASET -**

# In[ ]:


data


# Data Information- 

# In[ ]:


data.info()


# Data Description-

# In[ ]:


data.describe()


# **TIME SERIES GRAPHS -**

# In[ ]:


fig = px.line(data, x='timestamp', y='high', range_x=['2017-01-02 09:15:00+05:30','2020-06-26 15:29:00+05:30'], title="EVERY MINUTE'S HIGH TIME SERIES GRAPH  ")
fig.show()


# Removing time in timestamp and keeping only date -

# In[ ]:


data['timestamp'] = pd.to_datetime(data['timestamp']).dt.date
data


# Creating a new Dataframe containing everyday's high-

# In[ ]:


dz=data.pivot_table(index='timestamp',values=['high'],aggfunc=max)
dz.reset_index(inplace = True) 
dz


# In[ ]:


fig = px.line(dz, x='timestamp', y='high', range_x=['2017-01-02','2020-06-26'],title="EVERY DAY'S HIGH TIME SERIES GRAPH ")
fig.show()


# Creating a new Dataframe containing everyday's low-

# In[ ]:


dt=data.pivot_table(index='timestamp',values=['low'],aggfunc=min)
dt.reset_index(inplace = True) 
dt


# In[ ]:


fig = px.line(dt, x='timestamp', y='low', range_x=['2017-01-02','2020-06-26'],title="EVERDAY'S LOW TIME SERIES GRAPH")
fig.show()


# # **IF YOU LIKE THIS NOTEBOOK, UPVOTE WOULD BE APPRECIATED**
