#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Resouce and Assistance
# - https://gist.github.com/emmanuelle/2c96b5976caa5055caca03dca1c9d089
# - https://www.kaggle.com/avkash/bigquery-in-depth-using-google-analytics-data/edit
# - Plotly 
# 

# In[ ]:


import plotly.express as px 
import pandas as pd
from datetime import datetime
import json


# In[ ]:


import pandas as pd
df = pd.read_csv("../input/mycsvfile.csv")


# In[ ]:


def func_visitStartTime_updated(value):
    return datetime.fromtimestamp(value)

def func_visitStartTime_to_hour(value):
    return datetime.fromtimestamp(value).hour

def func_geoNetwork_to_continent(value):
    json_data = value.replace("\'", "\"")
    json_obj = json.loads(json_data)
    return json_obj["continent"]

def func_geoNetwork_to_subContinent(value):
    json_data = value.replace("\'", "\"")
    json_obj = json.loads(json_data)
    return json_obj["subContinent"]

def func_geoNetwork_to_country(value):
    json_data = value.replace("\'", "\"")
    json_obj = json.loads(json_data)
    return json_obj["country"]


# In[ ]:


df['visitStartTime_updated'] = df['visitStartTime'].apply(func_visitStartTime_updated)


# In[ ]:


df['hour'] = df['visitStartTime'].apply(func_visitStartTime_to_hour)


# In[ ]:


df['geoNetwork_continent'] = df['geoNetwork'].apply(func_geoNetwork_to_continent)


# In[ ]:


df['country'] = df['geoNetwork'].apply(func_geoNetwork_to_country)


# In[ ]:


df.head(10)


# In[ ]:


df['unit'] = 1
# df['hour'] = pd.to_datetime(df['serverTimePretty']).dt.hour
# For each country and hour, count number of visitors
dfg = df[['unit', 'country', 'hour']].groupby(['country', 'hour']).sum()


# In[ ]:


dfg.reset_index(inplace=True)
# Data cleaning, remove unknown
dfg = dfg.query("country != 'Unknown'")
dfg.sort_values(by='hour', inplace=True)


# In[ ]:


# Plotly figure
fig = px.choropleth(dfg, locations='country', color='unit', locationmode='country names',
                    animation_frame='hour', range_color=(0, dfg['unit'].max()),
                    color_continuous_scale='RdPu')

# Add a different title for each frame
for i, frame in enumerate(fig.frames):
    frame.layout.title = 'GMT time %d:00, visitors of Google Merchandise Store' %i
fig.show()

