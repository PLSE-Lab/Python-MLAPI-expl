#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import Point, Polygon
from IPython.display import display


# In[ ]:


data = pd.read_csv("../input/covid19/coronavirus_data.csv")
display(data.head())


# In[ ]:


display(data.columns)


# In[ ]:


display(data.columns.str.replace(r'\n','', regex=True))


# In[ ]:



data.columns = data.columns.str.replace(r'\n','', regex=True)
display(data.columns)


# In[ ]:



data.rename(columns={'Province/State':'Province_State','Country/Region':'Country_Region'},inplace=True)
display(data.columns)


# In[ ]:


#shape of dataset
display(data.shape)


# In[ ]:


display(data.head())


# In[ ]:


data = data[['Province_State', 'Country_Region', 'Lat', 'Long', 'Date','Confirmed', 'Deaths', 'Recovered']]


# In[ ]:


#detecting missing values
display(data.isna().sum())


# In[ ]:


display(data.describe())


# In[ ]:


display(data.head())


# In[ ]:


display(data.columns)


# In[ ]:


#number of case per date/day (sum)
display(data.groupby('Date')['Confirmed','Deaths', 'Recovered'].sum())


# In[ ]:



#number of case per date/day (max)
display(data.groupby('Date')['Confirmed','Deaths', 'Recovered'].max())


# In[ ]:


data_per_day = data.groupby('Date')['Confirmed','Deaths', 'Recovered'].max()
display(data_per_day.head())


# In[ ]:


display(data_per_day.describe())


# In[ ]:


#min no of cases
print('Minimum Number of Cases:', data_per_day['Confirmed'].min())
#max no of cases
print('Maximum Number of Cases:', data_per_day['Confirmed'].max())


# In[ ]:


#date for min number cases
print('The Date for Minimum Number Cases:', data_per_day['Confirmed'].idxmin())
#date for max number cases
print('The Date for Maximum Number Cases:', data_per_day['Confirmed'].idxmax())


# In[ ]:


#number of cases per country
display(data.groupby(['Country_Region'])['Confirmed','Deaths', 'Recovered'].max())


# In[ ]:


#number of cases per province/country
display(data.groupby(['Province_State','Country_Region'])['Confirmed','Deaths', 'Recovered'].max())


# In[ ]:


display(data['Country_Region'].value_counts())


# In[ ]:


data1 = data['Country_Region'].value_counts().plot(color='y',edgecolor='black',kind='bar',figsize=(17,6))
data1.tick_params(axis='x', colors='red')
data1.tick_params(axis='y', colors='red')
data1.set_facecolor('black')
plt.show()


# In[ ]:


print('Number of countrys affected:', len(data['Country_Region'].unique()))


# In[ ]:


#how many countrys affected(names)
display(data['Country_Region'].unique())


# In[ ]:


display(data.head())


# In[ ]:


#convert data to geodataframe
geodata = gpd.GeoDataFrame(data,geometry=gpd.points_from_xy(data['Long'],data['Lat']))
display(geodata.head())


# In[ ]:


#map_plot
geodata1 = geodata.plot(color='y',edgecolor='black',figsize=(17,8))
geodata1.tick_params(axis='x', colors='red')
geodata1.tick_params(axis='y', colors='red')
geodata1.set_facecolor('black')
plt.show()


# In[ ]:


#overlapping with world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(color='black',edgecolor='red',figsize=(18,8))
ax.tick_params(axis='x', colors='red')
ax.tick_params(axis='y', colors='red')
ax.axis('off')
plt.show()


# In[ ]:


#overlap
fig,ax = plt.subplots(figsize=(18,8))
geodata.plot(cmap='Purples',ax=ax)
world1 = world.geometry.boundary.plot(color=None,edgecolor='y',linewidth=2,ax=ax)
world1.tick_params(axis='x', colors='red')
world1.tick_params(axis='y', colors='red')
world1.set_facecolor('black')
plt.show()


# In[ ]:


#getting geographic regions
display(world['continent'].unique())


# In[ ]:


#spliting regions
Asia = world[world['continent'] == 'Asia']
display(Asia.head())


# In[ ]:


Africa = world[world['continent'] == 'Africa']
display(Africa.head())


# In[ ]:


North_america = world[world['continent'] == 'North America']
display(North_america.head())


# In[ ]:


Europe = world[world['continent'] == 'Europe']
display(Europe.head())


# In[ ]:


#ploting some of the affected countrys
#mainland_china
fig,ax = plt.subplots(figsize = (16,6))
geodata[geodata['Country_Region'] == 'Mainland China'].plot(cmap = 'Purples',ax=ax)
mainland_china = Asia.geometry.boundary.plot(color=None,edgecolor='y',linewidth=2,ax = ax)
mainland_china.tick_params(axis='x', colors='red')
mainland_china.tick_params(axis='y', colors='red')
mainland_china.set_facecolor('black')
plt.show()


# In[ ]:


#thailand
fig,ax = plt.subplots(figsize = (16,6))
geodata[geodata['Country_Region'] == 'Thailand'].plot(cmap = 'Purples',ax=ax)
thailand = Asia.geometry.boundary.plot(color=None,edgecolor='y',linewidth=2,ax = ax)
thailand.tick_params(axis='x', colors='red')
thailand.tick_params(axis='y', colors='red')
thailand.set_facecolor('black')
plt.show()


# In[ ]:


#us
fig,ax = plt.subplots(figsize = (16,6))
geodata[geodata['Country_Region'] == 'US'].plot(cmap = 'Purples',ax=ax)
us = Asia.geometry.boundary.plot(color=None,edgecolor='y',linewidth=2,ax = ax)
us.tick_params(axis='x', colors='red')
us.tick_params(axis='y', colors='red')
us.set_facecolor('black')
plt.show()


# In[ ]:


#analizing covid-19
display(data.head())


# In[ ]:


display(data_per_day.head())


# In[ ]:


#copying covid-19 data
data2 = data
data['cases_date'] = pd.to_datetime(data2['Date'])
display(data)


# In[ ]:


data3 = data['cases_date'].plot(figsize=(16,6),color='y',linestyle='-.')
plt.style.use('seaborn-whitegrid')
data3.tick_params(axis='x', colors='red')
data3.tick_params(axis='y', colors='red')
data3.set_facecolor('black')
plt.figure()
plt.show()


# In[ ]:


#cases by date
cd = data2.set_index('cases_date')
display(cd.loc['2020-02'])


# In[ ]:


cd2 = cd.loc['2020-01-24' :'2020-02-25'][['Confirmed','Recovered']].plot(color=['red','green'],kind='line',figsize= (17,5))
cd2.tick_params(axis='x', colors='red')
cd2.tick_params(axis='y', colors='red')
cd2.set_facecolor('black')
plt.show()


# In[ ]:



data_date = cd.groupby(['cases_date']).sum().reset_index(drop=None)
dd = data_date[['Confirmed','Recovered','Deaths']].plot(color=['blue','green','red'],kind='line',figsize=(17,5),linestyle='-.')
dd.tick_params(axis='x', colors='red')
dd.tick_params(axis='y', colors='red')
dd.set_facecolor('black')
plt.show()


# In[ ]:




