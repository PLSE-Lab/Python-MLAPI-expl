#!/usr/bin/env python
# coding: utf-8

# ### Interactive Map COVID-19 Cases per Country
# As demonstrated in the last class of how to make an interactive map containing information of countries' confirmed cases of COVID-19, shown below is the map I created based on the inquiry.

# In[ ]:


import pandas as pd
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

df['daily_existing'] = df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Inferno',range_color=(1000,1000000))
fig.update_layout(title_text='Shandy Cecilia',title_x=0.5)
fig.show()


# ### 2015's Nepal Hourly Earthquake Interactive Map

# In[ ]:


df3 = pd.read_csv('../input/nepal-earthquake/April 25th Earthquake.csv',header=0)

df3.index = pd.to_datetime(df3['time'])
df3['time'] = df3.index.strftime('%Y-%m-%d %H:00:00')
fig = px.scatter_geo(df3,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Inferno',range_color=(3.,6.))
fig.show()


# It was unfortunate that I was going to make a state-based Brazil's interactive COVID-19 map, but could not manage to find a right example or an effective coding.
