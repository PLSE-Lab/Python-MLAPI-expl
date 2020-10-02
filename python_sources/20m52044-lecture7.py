#!/usr/bin/env python
# coding: utf-8

# **20M52044_lecture7**

# # 1. Global Animation of COVID-19 Cases per country
# 
# To Understand COVID-19 cases in a full picture, this section consisted of
# 
# 1.1 Confirmed Cumulative Cases per Country
# 
# 1.2 Remaining Confirmed Cases per Country of each Day
# 
# (Both figure titles appear as my name in order to fulfill the report task)

# # 1.1 Confirmed Cumulative Cases per Country

# In[ ]:


import pandas as pd
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Reds',range_color=(0.1,100000.))
fig.update_layout(title_text='Linux Farungsang',title_x=0.5)
fig.show()


# # 1.2 Remaining Confirmed Cases per Country of each Day

# In[ ]:


df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

df['daily_existing'] = df ['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='daily_existing',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Reds',range_color=(0.1,100000.))
fig.update_layout(title_text='Linux Farungsang',title_x=0.5)
fig.show()


# # 2. Hourly Global Map of Earthquake Epicenter
# 
# Data Source: U.S. Geological Survey, 2020, Earthquake Catalog, accessed June 21, 2020 at URL https://earthquake.usgs.gov/earthquakes/search/?fbclid=IwAR2MnGKyPSc411eE6dBEY2auINVNGr__399ULcw9VZYQ2yPEGu7Vnd8-PGM

# In[ ]:


df = pd.read_csv('../input/april-25-2015-earthquake/query (1).csv',header=0)
df.index = pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')

fig = px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='plasma',range_color=(4.,7.))
fig.update_layout(title_text='April 25th, 2015 Hourly Global Earthquate',title_x=0.5)
fig.show()

