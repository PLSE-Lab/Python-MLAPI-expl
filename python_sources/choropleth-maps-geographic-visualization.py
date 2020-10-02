#!/usr/bin/env python
# coding: utf-8

# # Choropleth Maps Exercise 
# 
# Welcome to the Choropleth Maps Exercise! In this exercise we will give you some simple datasets and ask you to create Choropleth Maps from them. Due to the Nature of Plotly we can't show you examples
# 
# [Full Documentation Reference](https://plot.ly/python/reference/#choropleth)
# 
# ## Plotly Imports

# In[ ]:


import plotly.graph_objs as go 
from plotly.offline import init_notebook_mode,iplot
init_notebook_mode(connected=True)


# ** Import pandas and read the csv file: 2014_World_Power_Consumption**

# In[ ]:


import pandas as pd


# In[ ]:


df=pd.read_csv('../input/2014_World_Power_Consumption.csv')


# ** Check the head of the DataFrame. **

# In[ ]:


df.head()


# **  create a Choropleth Plot of the Power Consumption for Countries using the data and layout dictionary. **

# In[ ]:


data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        reversescale = True,
        locations = df['Country'],
        locationmode = "country names",
        z = df['Power Consumption KWH'],
        text = df['Country'],
        colorbar = {'title' : 'Power Consumption KWH'},
      ) 

layout = dict(title = '2014 Power Consumption KWH',
                geo = dict(showframe = False,projection = {'type':'stereographic'})
             )


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)


# ## USA Choropleth
# 
# ** Import the 2012_Election_Data csv file using pandas. **

# In[ ]:


df=pd.read_csv('../input/2012_Election_Data.csv')


# ** Check the head of the DataFrame. **

# In[ ]:


df.head()


# ** Now create a plot that displays the Voting-Age Population (VAP) per state. If you later want to play around with other columns, make sure you consider their data type. VAP has already been transformed to a float for you. **

# In[ ]:


data = dict(
        type = 'choropleth',
        colorscale = 'Viridis',
        reversescale = True,
        locations = df['State Abv'],
        locationmode = "USA-states",
        z = df['Voting-Age Population (VAP)'],
        text = df['State'],
        marker = dict(line = dict(color = 'rgb(255,255,255)',width = 1)),
        colorbar = {'title' : 'Voting-Age Population (VAP)'},
      ) 


# In[ ]:


layout = dict(title = '2012_Election_Data',
                geo = dict(scope='usa',
                         showlakes = True,
                         lakecolor = 'rgb(85,173,240)')
             )


# In[ ]:


choromap = go.Figure(data = [data],layout = layout)
iplot(choromap,validate=False)

