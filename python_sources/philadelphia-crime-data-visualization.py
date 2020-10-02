#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from folium.plugins import HeatMap
from folium import plugins
from collections import namedtuple
from shapely.geometry import Point

import geopandas # working with geospatial data in python easier
import folium #visualize spatial data in an interactive manner

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization library based on matplotlib
import missingno as msn
import matplotlib.pyplot as plt


# <h1><center>City of Philadelphia - Crime Data</center></h1>
# <img src="https://www.uwishunu.com/wp-content/uploads/2011/01/east-steps-150dpi-680uw.jpg" widht = 300 height = 500>

# # Introduction
# This dataset contains crime incidents from the Philadelphia Police Department. Part I crimes include violent offenses such as aggravated assault, rape, arson, among others. Part II crimes include simple assault, prostitution, gambling, fraud, and other non-violent offenses. The dataset previously had separate endpoints for various years and types of incidents. These have since been consolidated into a single dataset.

# # Feature Exploration, Engineering and Cleaning
# I'll start by first exploring the data on hand, identify possible feature engineering opportunities as well as numerically encode any categorical features.

# ### Loading the data

# In[ ]:


# acp = All Crime in Philidelphia
acp = pd.read_csv("/kaggle/input/philadelphiacrimedata/crime.csv")
# returns the first 5 parts of the acp dataframe
acp.head() 


# ## Feature Engineering & Cleaning

# In[ ]:


# remove all na values from the dataset
# acpc == All Crime Philidelphia Cleaned
acpc = acp.dropna()

# changing dtype police district to int
acpc = acpc.astype({'Police_Districts': 'int64'})

# order based on date
acpc['Dispatch_Date_Time'] = pd.to_datetime(acpc['Dispatch_Date_Time'])
acpc = acpc.sort_values(by='Dispatch_Date_Time', ascending=True)

# creating a seperate list for each year, month and day 
acpc['Year_Nr'] = acpc['Dispatch_Date_Time'].dt.year
acpc['Month_Nr'] = acpc['Dispatch_Date_Time'].dt.month
acpc['Day_Nr'] = acpc['Dispatch_Date_Time'].dt.day

# removing the year 2017 since it's not a complete year
acpc = acpc[acpc.Year_Nr != 2017]

# showing the first 5 rows
acpc.head()


# # Feature Exploration

# # Visualisations

# <b>Figure 1: Total number of crimes commited per year</b> <br>
# The first figure shows you the total nubmer of crimes commited per year. From this figure we can conclude that almost every year there is a decrease in the number of crimes happening. Good job Philly PD!!

# In[ ]:


sns.catplot(x='Year_Nr',
            kind='count',
            palette=("pastel"),
            height=6,
            aspect=2,
            data=acpc)

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Total number of crimes commited per year", fontsize=16)


# <b>Figure 2: Number of crimes commited per Month</b> <br>
# Figure 2 shows us the number of crimes happening in each month. Given this figure we can see that a decrease happens towards the end of the year throughout the beginning of the year. This might be due to the outside temperature of these months.
# 

# In[ ]:


sns.catplot(x='Month_Nr',
           kind='count',
           height=6,
           aspect=2,
           palette=("pastel"),
           data=acpc)

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel("Month", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Number of Crimes Commited per Month", fontsize=16)


# <b>Figure 3: Number of crimes commited per hour</b><br>
# In the third figure we can see that the number of crimes happening reaches a peak at 16:00h and is at its lowest point around 06:00h.
# Which is probably due to the sleeping scheduele of people. Furthermore it might be intersting to look at the different types of crime happening around each hour.

# In[ ]:


sns.catplot(x='Hour',
           kind='count',
           height=6,
           aspect=2,
           palette=("pastel"),
           data=acpc)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel("Hour", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Number of Crimes Commited per Hour", fontsize=16)


# <b>Figure 4: Number of Times a Specific Crime was Commited</b><br>
# In figure 4, the number of times each crime is commited is visualized. Where the named category Assualts and Theft are crimes that are commited the most within Philadelphia

# In[ ]:


sns.catplot(y='Text_General_Code',
           kind ='count',
           height = 8,
           aspect =1.5, 
           palette=("pastel"),
           order=acpc.Text_General_Code.value_counts().index,
           data=acpc)

plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel("Count", fontsize=14)
plt.ylabel("Type of Crime", fontsize=14)
plt.title("Number of Times a Specific Crime was Commited", fontsize=16)


# <b>Figure 5: Number of Times a Specific Crime was Commited</b><br>
# In the last figure, figure 5, the number of times a crime is commited in each district is visualized. From this we can see that most crimes happen district 11

# In[ ]:


sns.catplot(x='Police_Districts',
           kind='count',
           height=6,
           aspect=2,
           palette=("pastel"),
           data=acpc)
plt.xticks(size=12)
plt.yticks(size=12)
plt.xlabel("Police District", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.title("Number of Crimes Commited per Police District", fontsize=16)


# Given the previous image, an indication is given of where each district is placed in the map below.

# In[ ]:


lon = acpc['Lon']
lat = acpc['Lat']
avgLon = sum(lon)/len(lon)
avgLat = sum(lat)/len(lat)

districts_location = acpc[['Police_Districts', 'Lon', 'Lat']]
districts = districts_location.groupby(['Police_Districts']).mean().reset_index()


crime_map = folium.Map(location=[avgLat, avgLon], 
                       tiles = "Stamen Toner",
                      zoom_start = 11)

# Add data for heatmp 
data_heatmap = acpc[-25000:]
data_heatmap = data_heatmap[['Lat','Lon']]
data_heatmap = [[row['Lat'],row['Lon']] for index, row in data_heatmap.iterrows()]
HeatMap(data_heatmap, radius=10).add_to(crime_map)
for i in range(len(districts)):
    folium.Marker([districts['Lat'][i], districts['Lon'][i]], popup='District ' + str(districts['Police_Districts'][i])).add_to(crime_map)

# Plot!
crime_map


# # Heatmap
# The following heatmaps shows the areas in which most crimes, thef crime and violent crime happen

# In[ ]:


lon = acpc['Lon']
lat = acpc['Lat']
avgLon = sum(lon)/len(lon)
avgLat = sum(lat)/len(lat)

crime_map = folium.Map(location=[avgLat, avgLon], 
                       tiles = "Stamen Toner",
                      zoom_start = 11)

# Add data for heatmp 
data_heatmap = acpc[-25000:]
data_heatmap = data_heatmap[['Lat','Lon']]
data_heatmap = [[row['Lat'],row['Lon']] for index, row in data_heatmap.iterrows()]
HeatMap(data_heatmap, radius=10).add_to(crime_map)

# Plot!
crime_map


# # Voilent Crimes

# In[ ]:


#showing all types of crimes commited
acpc.Text_General_Code.unique()

# The most dangerous crime (weapon involved) types that can be commited
dangerous = ['Weapon Violations', 'Robbery Firearm', 'Homicide - Criminal', 'Aggravated Assault Firearm', 'Homicide - Gross Negligence', 'Homicide - Justifiable']
# creating a new dataframe with all the dangerous crimes in it
dangerous_data = acpc[acpc['Text_General_Code'].isin(dangerous)]

dangerous_data.head()


# In[ ]:


lon = dangerous_data['Lon']
lat = dangerous_data['Lat']
avgLon = sum(lon)/len(lon)
avgLat = sum(lat)/len(lat)

crime_map = folium.Map(location=[avgLat, avgLon], 
                       tiles = "Stamen Toner",
                       zoom_start = 11)

# Add data for heatmp 
data_heatmap = dangerous_data[dangerous_data.Year_Nr == 2016]
data_heatmap = data_heatmap[['Lat','Lon']]
data_heatmap = [[row['Lat'],row['Lon']] for index, row in data_heatmap.iterrows()]

HeatMap(data_heatmap, radius=10).add_to(crime_map)

# Plot!
crime_map


# # Theft Crimes

# In[ ]:


Theft = ['Thefts', 'Theft from Vehicle', 'Motor Vehicle Theft', 'Receiving Stolen Property', 'Recovered Stolen Motor Vehicle']
theft_data = acpc[acpc['Text_General_Code'].isin(Theft)]


# In[ ]:


lon = theft_data['Lon']
lat = theft_data['Lat']
avgLon = sum(lon)/len(lon)
avgLat = sum(lat)/len(lat)

crime_map = folium.Map(location=[avgLat, avgLon], 
                       tiles = "Stamen Toner",
                       zoom_start = 11)

# Add data for heatmap 
data_heatmap = theft_data[theft_data.Year_Nr == 2016]
data_heatmap = data_heatmap[['Lat','Lon']]
data_heatmap = [[row['Lat'],row['Lon']] for index, row in data_heatmap.iterrows()]
HeatMap(data_heatmap, radius=10).add_to(crime_map)

# Plot!
crime_map

