#!/usr/bin/env python
# coding: utf-8

# ### H3 Border Crossing - Simple Exploration (EDA)

# In[ ]:


import os
import warnings
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt
import chart_studio.plotly as py
warnings.filterwarnings("ignore")
import plotly.graph_objects as go


# In[ ]:


df = pd.read_csv("../input/border-crossing-entry-data/Border_Crossing_Entry_Data.csv")


# *1. Basic Exploration*

# In[ ]:


df.info()


# In[ ]:


print("Shape of DataFrame", df.shape)
print(" DataFrame Columns", df.columns)


# ** DATA SET COLUMNS DESCRIPTION ** <br/>
# > **Port Name** Name of CBP Port of Entry <br/>
# > **State** State <br/>
# > **Port Code** CBP port code <br/>
# > **Border** US-Canada Border [](http://)or US-Mexico Border <br/>
# > **Date** Year, Month <br/>
# > **Measure** Conveyances, containers, passengers, pedestrians <br/>
# > **Value** Count <br/>
# > **Location** Longitude and Latitude Location

# ** PORT NAME **

# In[ ]:


print("Different Crossing Ports along with there counts")
print(df['Port Name'].value_counts())


# In[ ]:


def getLat(word):
    x = word.split(' ')
    lat = x[1][1:]
    return lat

def getLong(word):
    x = word.split(' ')
    long = x[2][:-1]
    return long

df['Latitude'] = df['Location'].apply(getLat)
df['Longitude'] = df['Location'].apply(getLong)


# In[ ]:


BoundaryDimensions = (df.Longitude.min(),   df.Longitude.max(),df.Latitude.min(), df.Latitude.max())
print(BoundaryDimensions)


# In[ ]:


x = df['State'].value_counts().index
y = df['State'].value_counts().values
fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
            marker={'color': 'royalblue'}
        )])

fig.update_layout(
    title={
    'text': "State Wise Distribution",
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    xaxis_title="States",
    yaxis_title="Instances",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)


# **Seems like North Dakota was the famous point of Crossing**

# In[ ]:


x = df['Border'].value_counts().index
y = df['Border'].value_counts().values

fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
            marker={'color': 'royalblue'},
            name='BORDER'
        )
])

fig.update_layout(
    title={
    'text': "Border Distribution",
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    xaxis_title="Borders",
    yaxis_title="Instances",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)


# 1. ** Looks like CANADA border was the most prefered choice (almost 4 times as compared to Mexico Border) **

# In[ ]:


df['Measure'].value_counts()


# In[ ]:


x = df['Measure'].value_counts().index
y = df['Measure'].value_counts().values

fig = go.Figure(data=[go.Bar(
            x=x, y=y,
            text=y,
            textposition='auto',
            marker={'color': 'royalblue'},
        )
])

fig.update_layout(
    title={
    'text': "Measure Distribution",
    'y':0.9,
    'x':0.5,
    'xanchor': 'center',
    'yanchor': 'top'},
    xaxis_title="Measures of Cross",
    yaxis_title="Instances",
    font=dict(
        family="Courier New, monospace",
        size=18,
        color="#7f7f7f"
    )
)


# **It seems that all measures are equally used**

# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])


# In[ ]:


df.dtypes


# In[ ]:


df['Year'] =  df.Date.dt.year
df['Month'] = df.Date.dt.month


# ** Checking 5 random dates ** 

# In[ ]:


df[['Date', 'Year', 'Month']].sample(5)


# In[ ]:


frame = { 'Index': df[df['Year'] < 2019].Year.value_counts().sort_index().index, 'Values': df[df['Year'] < 2019].Year.value_counts().sort_index() } 
result = pd.DataFrame(frame) 

fig = px.line(result, x="Index", y="Values", title="Year-Wise Crossings", labels={'Index': 'Year', 'Values': 'Number of Crossings'})
fig.show()


# *** Number of vehicles crossing the border has been on a decline starting 2016 ***

# > *** Please upvote if it was useful or worth looking at. ***
# 
