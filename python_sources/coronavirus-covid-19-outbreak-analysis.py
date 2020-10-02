#!/usr/bin/env python
# coding: utf-8

# ![](https://storage.googleapis.com/kagglesdsdata/datasets/607959/1089325/corona.jpg?GoogleAccessId=web-data@kaggle-161607.iam.gserviceaccount.com&Expires=1588246063&Signature=KPIDfJ8%2F5HaXZLZu6GmwVDwISHlyikHqE6WkCQies17vdNWlSHD2htQIserFAc7GP3GjsIdGdijPn6rdy680yB8SBCtSLTUi69Bfu1jbMR50xWkVvEZH1c8scWCneTihfPyG7Lqt8EgwtzneDopA6Iw39rl%2B3JPSytfWAO9AZEtKGWZoQcpmMYgvPMOcS%2BUWfAfMpLYik4C9nQU0EGzh6d2rmW8TokJqY%2FGGLgrfvm1G8NIWOgl4Wkr8LDIL%2FN8o96m%2FAVQ21DuqCO9ahrCSwBMsBh7BzXArsNKVUaf0G0OxKCMtvLYOLmEcgIdhbmP9CJ5UAOdTH01P9Jjrf7TyAQ%3D%3D)

# The Coronavirus (COVID-19) outbreak is an ongoing outbreak which started in 2019 hence the number 19 in the COVID-19. It is caused by the Severe Acute Respiratory Syndrome(SARS) -CoV-2 virus. In December 2019, a pneumonia outbreak was reported in Wuhan, China.

# # From this dataset I will be answering some basic questions:
# * Number of Cases (Recovered,Confirmed,Deaths)
# * Which country has the highest cases?
# * List of countries affected
# * Distribution Per Continents and Country
# * Cases Per Day
# * Cases Per Country
# * Timeseries Analysis

# By analysing our data we can see that it consist of data about
# 
# * Country/Region
# * Latitude and Longitude
# * Date/ Time
# * Numerical Data

# Hence by these simple overview we can perform the following types of analysis
# 
# * Geo-spatial analysis from the Latitude and Longitudes.
# * Time series analysis from the Date/Time.
# * Statistical analysis from the numerical data.

# Let us start with the basic EDA and then the rest.
# 
# We will be using pandas,matplotlib and geopandas to help us with our analysis.

# In[ ]:


#Load EDA pkgs
import pandas as pd
import numpy as np


# In[ ]:


#Load Data Visualization pkgs
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#Load GeoPandas
import geopandas as gpd
from shapely.geometry import Point, Polygon
import descartes


# In[ ]:


#Load Dataset
data = pd.read_csv('../input/corona-virus-report/covid_19_clean_complete.csv')


# In[ ]:


data.head()


# In[ ]:


data.columns


# In[ ]:


data.rename(columns={'Province/State':'Province_State','Country/Region':'Country_Region'},inplace=True)


# In[ ]:


data.columns


# In[ ]:


#shape of data
data.shape


# In[ ]:


# datatypes
data.dtypes


# In[ ]:


# First 10
data.head(10)


# In[ ]:


data=data[['Province_State', 'Country_Region', 'Lat', 'Long', 'Date', 'Confirmed', 'Deaths', 'Recovered' ]]


# In[ ]:


data.isna().sum()


# In[ ]:


data.describe()


# In[ ]:


# Number of case per Date/Day
data.head()


# In[ ]:


data.groupby('Date')['Confirmed','Deaths','Recovered'].sum()


# In[ ]:


data.groupby('Date')['Confirmed','Deaths','Recovered'].max()


# In[ ]:


data_per_day = data.groupby('Date')['Confirmed','Deaths','Recovered'].max()


# In[ ]:


data_per_day.head()


# In[ ]:


data_per_day.describe()


# In[ ]:


# Max No of cases
data_per_day['Confirmed'].max()


# In[ ]:


# Min No of cases
data_per_day['Confirmed'].min()


# In[ ]:


# Date for Maximum Number Cases 
data_per_day['Confirmed'].idxmax()


# In[ ]:


# Date for Minimun Number Cases
data_per_day['Confirmed'].idxmin()


# In[ ]:


#Number of Case Per Country/Province
data.groupby(['Country_Region'])['Confirmed','Deaths','Recovered'].max()


# In[ ]:


# Number of Case Per Country/Province
data.groupby(['Province_State','Country_Region'])['Confirmed','Deaths','Recovered'].max()


# In[ ]:


data['Country_Region'].value_counts()


# In[ ]:


data['Country_Region'].value_counts().plot(kind='bar',figsize=(30,10))


# In[ ]:


#How Many Country Affect
data['Country_Region'].unique()


# In[ ]:


# How Many Country Affect
len(data['Country_Region'].unique())


# In[ ]:


plt.figure(figsize=(30,30))
data['Country_Region'].value_counts().plot.pie(autopct="%1.1f%%")


# # Check for Distribution on Map
# * Lat/Long
# * Geometry/ Poin

# In[ ]:


dir(gpd)


# In[ ]:


data.head()


# In[ ]:


# Convert Data to GeoDataframe
gdata01 = gpd.GeoDataFrame(data,geometry=gpd.points_from_xy(data['Long'],data['Lat']))


# In[ ]:


gdata01.head()


# In[ ]:


type(gdata01)


# In[ ]:


# Method 2
points = [Point(x,y) for x,y in zip(data.Long,data.Lat)]


# In[ ]:


gdata02 = gpd.GeoDataFrame(data,geometry=points)


# In[ ]:


gdata02


# In[ ]:


#Map Plot
gdata01.plot(figsize=(20,10))


# In[ ]:


# Overlapping With World Map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(20,10))
ax.axis('off')


# In[ ]:


# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01.plot(cmap='Purples',ax=ax)
world.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)


# In[ ]:


fig,ax = plt.subplots(figsize=(20,10))
gdata01.plot(cmap='Purples',ax=ax)
world.geometry.plot(color='Yellow',edgecolor='k',linewidth=2,ax=ax)


# In[ ]:


world['continent'].unique()


# In[ ]:


asia = world[world['continent'] == 'Asia']


# In[ ]:


asia


# In[ ]:


africa = world[world['continent'] == 'Africa']
north_america = world[world['continent'] == 'North America']
europe = world[world['continent'] == 'Europe']


# In[ ]:


# Cases in China
data.head()


# In[ ]:


data[data['Country_Region'] == 'China']


# In[ ]:


gdata01[gdata01['Country_Region'] == 'China']


# In[ ]:


# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'China'].plot(cmap='Purples',ax=ax)
world.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)


# In[ ]:


# Overlaph
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'China'].plot(cmap='Purples' ,ax=ax)
asia.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)


# In[ ]:


#Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region']== 'India'].plot(cmap='Purples',ax=ax)
asia.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)


# In[ ]:


# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'Egypt'].plot(cmap='Purples',ax=ax)
africa.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)


# In[ ]:


# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'US'].plot(cmap='Purples',ax=ax)
north_america.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)


# In[ ]:


# Overlap
fig,ax = plt.subplots(figsize=(20,10))
gdata01[gdata01['Country_Region'] == 'United Kingdom'].plot(cmap='Purples',ax=ax)
europe.geometry.boundary.plot(color=None,edgecolor='k',linewidth=2,ax=ax)


# # Time Series Analysis

# In[ ]:


data.head()


# In[ ]:


data_per_day


# In[ ]:


data2 = data


# In[ ]:


data.to_csv("E:\covid_19_clean_complete.csv")


# In[ ]:


import datetime as dt


# In[ ]:


data['cases_date'] = pd.to_datetime(data2['Date'])


# In[ ]:


data2.dtypes


# In[ ]:


data['cases_date'].plot(figsize=(20,10))


# In[ ]:


ts = data2.set_index('cases_date')


# # ts

# In[ ]:


# Select For January
ts.loc['2020-01']


# In[ ]:


# Select For January
ts.loc['2020-01']


# In[ ]:


ts.loc['2020-02-24':'2020-02-25']


# In[ ]:


ts.loc['2020-02-24':'2020-02-25']


# In[ ]:


ts.loc['2020-02-24':'2020-02-25'][['Confirmed','Recovered']].plot(figsize=(20,10))


# In[ ]:


ts.loc['2020-02-2':'2020-02-25'][['Confirmed','Deaths']].plot(figsize=(20,10))


# In[ ]:


data_by_date = ts.groupby(['cases_date']).sum().reset_index(drop=None)


# In[ ]:


data_by_date


# In[ ]:


data_by_date.columns


# In[ ]:


data_by_date[['Confirmed', 'Deaths', 'Recovered']].plot(kind='line',figsize=(20,10))

