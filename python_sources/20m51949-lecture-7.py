#!/usr/bin/env python
# coding: utf-8

# # Day-to-day cases of Corona Virus Worldwide (country-based)
# 
# Supatat Pongpattanayok 

# In[ ]:


import plotly.express as px  
import pandas as pd  # data processing

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='Confirmed',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Rainbow',range_color=(0.1,200000.))
fig.update_layout(title_text='Confirmed Cumulative Cases per Country', title_x=0.5)
fig.show()


# In[ ]:


import plotly.express as px  
import pandas as pd  # data processing

df = pd.read_csv('../input/novel-corona-virus-2019-dataset/covid_19_data.csv',header=0)
df = df.groupby(['ObservationDate','Country/Region']).sum().reset_index()

df['daily_existing']=df['Confirmed'].values-df['Deaths'].diff()-df['Recovered'].diff()

fig = px.choropleth(df,locations='Country/Region',locationmode='country names',color='daily_existing',hover_name='Country/Region',animation_frame='ObservationDate',color_continuous_scale='Rainbow',range_color=(0.1,200000.))
fig.update_layout(title_text='Remaining confirmed Cases per Country', title_x=0.5)
fig.show()


# # Hourly global map of Nepal earthquake epicenters on April 2015

# In[ ]:


df = pd.read_csv('../input/nepal-april-2015-eq/nepal.csv',header=0)

df.index = pd.to_datetime(df['time'])
df['time'] = df.index.strftime('%Y-%m-%d %H:00:00')

fig = px.scatter_geo(df,lat='latitude',lon='longitude',color='mag',animation_frame='time',color_continuous_scale='Rainbow',range_color=(1,9.))
fig.update_layout(title_text='Nepal Earthquake on April 25th, 2015', title_x=0.5)
fig.show()

