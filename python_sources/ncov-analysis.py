#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(16,6))

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/novel-corona-virus-2019-dataset/2019_nCoV_data.csv')
data.head()


# In[ ]:


data.describe()


# In[ ]:


data.describe(include="O")


# In[ ]:


data.info()


# In[ ]:


data.isna().sum()


# <h1> **EDA** </h1>

# Excluding country name Others

# In[ ]:


data=data[data.Country!='Others']
data.head()


# [](http://)Extracting Date and Time from the 'Last Update' column

# In[ ]:


data["Date"] = pd.to_datetime(data["Date"])


# In[ ]:


data['Last Update'] = pd.to_datetime(data['Last Update'])
data['Last Update'].head()


# Checking Date-wise Trend of confirmed patients and number of death

# In[ ]:


data.groupby('Date').sum()['Confirmed'].plot()
plt.grid()


# In[ ]:


data.groupby('Date').sum()['Deaths'].plot()
plt.grid()


# Country-wise Distribution

# In[ ]:


country_count = data['Country'].value_counts().reset_index()
country_count = country_count.rename(columns={"index": "Country", "Country": "Count"})
country_count.head()


# In[ ]:


country_count = pd.DataFrame(data.groupby('Country')['Confirmed'].max()).reset_index()
country_count.head()


# In[ ]:


country_count_wo_mc=country_count[country_count.Country != 'Mainland China']
country_count_wo_mc.head()


# In[ ]:


import plotly.graph_objects as go
fig = go.Figure(data = [go.Pie(labels = country_count_wo_mc.Country,values = country_count_wo_mc.Confirmed)])
fig.update_layout(title="Country-wise distribution except Mainland China")
fig.show()


# In[ ]:


state_count = pd.DataFrame(data[data['Country']=='Mainland China'].groupby('Province/State').max()['Confirmed']).reset_index()
fig = go.Figure(data = [go.Pie(labels = state_count['Province/State'],values = state_count.Confirmed)])
fig.update_layout(title="State-wise distribution of Mainland China")
fig.show()


# Growth trend of Corona virus in each Country individually except Mainland China

# In[ ]:


growth_country = pd.pivot_table(data[data.Country!='Mainland China'], values='Confirmed', columns='Country', index='Date')
growth_country.plot(figsize=(18,12))


# Growth trend of Corona virus in each Province/State of Mainland China individually

# In[ ]:


growth_state = pd.pivot_table(data[data['Country']=='Mainland China'], values='Confirmed', columns='Province/State', index='Date')
growth_state.plot(figsize=(18,12))


# Country affected on the first day and number of confirmed cases on that date in each country.

# In[ ]:


initial_country=data[data['Date'] == data['Date'][0]]['Country'].unique()
initial_country


# In[ ]:


data[data['Date'] == data['Date'][0]].groupby('Country').sum()['Confirmed']


# Total Confirmed, Recovery and Death in each country

# In[ ]:


global_case_wo_mc = data[data.Country != 'Mainland China'].groupby('Country')['Confirmed','Deaths','Recovered'].max().reset_index()
global_case_wo_mc.head()


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Confirmed', x=global_case_wo_mc['Country'], y=global_case_wo_mc['Confirmed']),
    go.Bar(name='Deaths', x=global_case_wo_mc['Country'], y=global_case_wo_mc['Deaths']),
    go.Bar(name='Recovered', x=global_case_wo_mc['Country'], y=global_case_wo_mc['Recovered']),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# Total Confirmed, Recovery and Death in each Province/State of Mainland China

# In[ ]:


mainland_case = data[data['Country']=='Mainland China'].groupby('Province/State')['Confirmed','Deaths','Recovered'].sum().reset_index()
mainland_case.head()


# In[ ]:


fig = go.Figure(data=[
    go.Bar(name='Confirmed', x=mainland_case['Province/State'], y=mainland_case['Confirmed']),
    go.Bar(name='Deaths', x=mainland_case['Province/State'], y=mainland_case['Deaths']),
    go.Bar(name='Recovered', x=mainland_case['Province/State'], y=mainland_case['Recovered']),
])
# Change the bar mode
fig.update_layout(barmode='group')
fig.show()


# Death Trend Globally

# In[ ]:


recovery_country = pd.pivot_table(data, values='Recovered', columns='Country', index='Date')
recovery_country.plot(figsize=(18,12))


# In[ ]:


death_country = pd.pivot_table(data, values='Deaths', columns='Country', index='Date')
death_country.plot(figsize=(18,12))


# <h3> Confirmed, Recovery and Death globally on world map </h3>

# In[ ]:


global_case = data.groupby('Country')['Confirmed','Deaths','Recovered'].max().reset_index()
global_case


# In[ ]:


import folium 
from folium import plugins
world_data = pd.DataFrame({
    'name':list(global_case['Country']),
    'lat':[-25.2744,50.503887,-14.235004,12.565679,56.130366,35.86166,26.820553,61.92411,46.227638,51.165691,22.39643,20.593684,41.87194,-22.49,36.204824,22.198745,39.93,4.210484,23.634501,28.394857,12.879721,61.52401,1.352083,35.907757,40.463669,7.873054,60.128162,23.69781,15.870032,55.378052,37.09024,23.424076,14.058324],
    'lon':[133.7751,4.469936,-51.92528,104.990963,-106.346771,104.195397,30.802498,25.748151,2.213749,10.451526,114.1095,78.96288,12.56738,17.01,138.252924,113.543873,116.38,101.975766,-102.552784,84.124008,121.774017,105.318756,103.819836,127.766922,-3.74922,80.771797,18.643501,120.960515,100.992541,-3.435973,-95.712891,53.847818,108.277199],
    'Confirmed':list(global_case['Confirmed']),
    'Recovered':list(global_case['Recovered']),
    'Deaths':list(global_case['Deaths']),
})

# create map and display it
world_map = folium.Map(location=[10, -20], zoom_start=2.1,tiles='openstreetmap')

for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Confirmed'], world_data['name']):
    if value>1000:
        folium.CircleMarker([lat, lon],
                        radius=value*0.0004,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases as on 30th Jan 2020</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
    elif value>100:
        folium.CircleMarker([lat, lon],
                        radius=value*0.02,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases as on 30th Jan 2020</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
    elif value<10:
        folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases as on 30th Jan 2020</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
    else:
        folium.CircleMarker([lat, lon],
                        radius=value*0.2,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Confirmed Cases as on 30th Jan 2020</strong>: ' + str(value) + '<br>'),
                        color='red',
                        
                        fill_color='red',
                        fill_opacity=0.7 ).add_to(world_map)
world_map


# In[ ]:


world_map = folium.Map(location=[10, -20], zoom_start=2.1,tiles='openstreetmap')
for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Recovered'], world_data['name']):
    if value>0.0:
        folium.CircleMarker([lat, lon],
                        radius=10,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Recovered Cases as on 30th Jan 2020</strong>: ' + str(value) + '<br>'),
                        color='green',
                        
                        fill_color='green',
                        fill_opacity=0.7 ).add_to(world_map)
world_map


# In[ ]:


world_map = folium.Map(location=[10, -20], zoom_start=2.1,tiles='openstreetmap')

for lat, lon, value, name in zip(world_data['lat'], world_data['lon'], world_data['Deaths'], world_data['name']):
    if value>0:
        folium.CircleMarker([lat, lon],
                        radius=8,
                        popup = ('<strong>Country</strong>: ' + str(name).capitalize() + '<br>'
                                '<strong>Death Cases as on 30th Jan 2020</strong>: ' + str(value) + '<br>'),
                        color='black',
                        
                        fill_color='black',
                        fill_opacity=0.7 ).add_to(world_map)


# In[ ]:


import pycountry
countries = {}
for country in pycountry.countries:
    countries[country.name] = country.alpha_3
country_count["iso_alpha"] = country_count['Country'].map(countries.get)


# There are some countries whose iso_alpha value is not present and it is returning it as None. Droping those records.

# In[ ]:


filetered_country=country_count.dropna()
filetered_country.head()


# In[ ]:


import plotly.express as px
df = px.data.gapminder().query("year == 2007")
fig = px.choropleth(filetered_country, locations="iso_alpha",
                    color='Confirmed',
                    color_continuous_scale=px.colors.sequential.Plasma,
                    projection="natural earth")
fig.show()


# In[ ]:




