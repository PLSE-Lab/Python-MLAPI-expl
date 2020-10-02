#!/usr/bin/env python
# coding: utf-8

# # Exercise 7- Mako Mizuno (20M51808)
# # email: mizuno.m.af@m.titech.ac.jp

# # Day-to-day cases of Corona Virus Worldwide (country-based)
# 
# Creating maps showing "Confirmed Cumulative Cases" and "Remaining Confirmed Cases".

# In[ ]:


import pandas as pd
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='matter',range_color=(0.1,300000.))
fig.update_layout(title_text='Confirmed Cumulative Cases per Country',title_x=0.5)
fig.show()


# In[ ]:


import pandas as pd
import plotly.express as px

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

df['daily_existing'] = df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='daily_existing',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='matter',range_color=(0.1,200000.))
fig.update_layout(title_text='Remaining Confirmed Cases per Country',title_x=0.5)
fig.show()


# # Hourly global map of earthquake epicenters
# Focusing on the earthquake in Nepal on April 25, 2015.
# 
# The main shock occurred around 6:00 am (UTC).

# In[ ]:


df = pd.read_csv('../input/april-25-2015-earthquake-in-nepal/query (2).csv',header=0)
print(df)


# In[ ]:


df = pd.read_csv('../input/april-25-2015-earthquake-in-nepal/query (2).csv',header=0)

df.index = pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')
fig = px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='matter',range_color=(4.,8.))
fig.show()

