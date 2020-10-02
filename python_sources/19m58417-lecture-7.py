#!/usr/bin/env python
# coding: utf-8

# # Exercice 7: Covid-19 and Earthquake - 19M58417 CHEN Tong

# ## global animation of Covid-19 cases per country
# In this part, I made two animation figures. One is for global Covid-19 confirmed cumulative cases per country, and another one is for global Covid-19 daily existing cases per country. Their case range is from 0 to 100K.

# In[ ]:


import pandas as pd
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv', header = 0)
df = df.groupby(['ObservationDate', 'Country/Region']).sum().reset_index()

df['Daily_existing'] = df['Confirmed'].values - df['Deaths'].diff()-df['Recovered'].diff()

fig1 = px.choropleth(df, locations = 'Country/Region', locationmode = 'country names', color = 'Confirmed', hover_name = 'Country/Region', animation_frame = 'ObservationDate', color_continuous_scale = 'Matter', range_color = (0.1, 100000.))
fig1.update_layout(title_text='CHEN TONG - Confirmed Cumulative Cases per Country', title_x = 0.5)
fig1.show()

fig2 = px.choropleth(df, locations = 'Country/Region', locationmode = 'country names', color = 'Daily_existing', hover_name = 'Country/Region', animation_frame = 'ObservationDate', color_continuous_scale = 'Matter', range_color = (0.1, 100000.))
fig2.update_layout(title_text='CHEN TONG - Daily Existing Cases per Country', title_x = 0.5)
fig2.show()


# ## Global data on the day of Nepal Earthquake (April 25th, 2015)
# In this part, I made an animation figure for global earthquake on April 25th, 2015.

# In[ ]:


df = pd.read_csv('../input/earthquake-20150425/query.csv', header = 0)

df.index = pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')
fig = px.scatter_geo(df, lat='latitude', lon = 'longitude', color = 'mag', animation_frame = 'time', color_continuous_scale = 'turbid', range_color=(1, 8.))
fig.show()

