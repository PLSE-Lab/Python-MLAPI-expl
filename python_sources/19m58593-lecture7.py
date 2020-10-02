#!/usr/bin/env python
# coding: utf-8

# # Worldwide Daily COVID-19 Cases Animation
# The animated map of the global day-to-day COVID-19 confirmed cases is created by using plotly. 

# In[ ]:


import pandas as pd 
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

df['daily remaining case']=df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='daily remaining case',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Reds',range_color=(0.1,100000))
fig.update_layout(title_text='WANG Zihan(Remaining Daily Confirmed Cases per Country)', title_x=0.5)
fig.show()


# # Hourly Global Map of Earthquake Epicenters
# The dataset is for the Nepal Earthquake that occurred on April 25th, 2015 (World Vision, 2020)
# 
# Source: *2015 Nepal Earthquake: Facts, FAQs, and how to help.* World Vision. Retrieved from
# https://www.worldvision.org/disaster-relief-news-stories/2015-nepal-earthquake-facts

# In[ ]:


df = pd.read_csv('../input/april-25-2015-nepal-earthquake/query.csv',header = 0)
df.index = pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')
fig = px.scatter_geo(df,lat='latitude', lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Rainbow',range_color=(5.,7.))
fig.update_layout(title_text='WANG Zihan', title_x=0.5)
fig.show()

