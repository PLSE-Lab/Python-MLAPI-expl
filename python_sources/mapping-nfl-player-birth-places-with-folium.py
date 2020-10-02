#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import folium
import pandas as pd
import time


# **Objective**
# 
# The goal of this notebook to identify geographic origin points of NFL football players. The analysis will focus on the birthplace and high school locations for each player (where data is available)

# In[ ]:


#create a data frame 
stats = pd.read_csv('../input/nflstatistics/Basic_Stats.csv')


# In[ ]:


#understand the dimensions of our data
stats.shape


# In[ ]:


# print out the first 10 rows
stats.head(10)


# In[ ]:


#determine the number of missing values for each column
stats.isna().sum()


# In[ ]:


stats_dropna = stats.replace('', np.nan)
stats_clean = stats_dropna.dropna(subset=['Birth Place'], how = 'any')
stats_clean.isna().sum()


# We now have values for Birth Place for all rows.. Out of 17172 rows we see that 12467 values for 'High School Location' are NAs! The amount of missing data points devalues analysis of 'High School Location'. We will concentrate on the 'Birthplace location' field.

# In[ ]:


stats_clean.shape


# In[ ]:


# we can geocode the birthplace cities
#lets rename Birth Place field to remove the space
stats_clean.rename(columns={'Birth Place': 'BirthPlace'}, inplace=True)


# In[ ]:


stats_clean.head(10)


# In[ ]:


#Retrieve the number of players borm in city
counts = stats_clean['BirthPlace'].value_counts()
countsdf = pd.DataFrame(counts)
countsdf.shape


# In[ ]:


countsdf  = countsdf.reset_index().rename(columns={' ':'CityState'})


# In[ ]:


countsdf.rename(columns={'index': 'CityState','BirthPlace': 'freq'}, inplace=True)


# In[ ]:


countsdf.head()


# In[ ]:


countsdf['CityState'] = countsdf['CityState'].str.replace(' ,', ',')


# In[ ]:


countsdf.head()


# In[ ]:


citiesreduced = countsdf['freq'] > 10 #there needs to be ten or more football players per city 
countsdf = countsdf[citiesreduced ]


# In[ ]:


countsdf.shape


# # Geocoding and Extracting Population

# In[ ]:


"""
import geocoder

for index,row in countsdf.iterrows():
    b = geocoder.geonames(row['CityState'], key='YOUR-USER-NAME')
    countsdf.loc[index,'Y_LAT'] = b.lat
    countsdf.loc[index,'X_LONG'] = b.lng
    countsdf.loc[index,'population'] = b.population
    print(b.population,b.lat,b.lng,b.address)
    time.sleep(7)

print("done geocoding with geonames")

results of this code snippet pushd to csv below. This code works great. Applying an
appropriate sleep timer is crucial to retrieving the information from geonames. For more info 
on GeoNames see http://www.geonames.org/export/web-services.html. I prefer geocoder over geopy
becuase extracting population from GeoNames was simple with the geocoder library.

"""


# In[ ]:


"""
Once the coordinates and poplation are retreived we can filter the datafrma down to the info we really need. 

data = countsdf.filter(['CityState','freq','Y_LAT','X_LONG','population'])

"""


# In[ ]:


"""

per1000 = countsdf['population']/ 1000
data['playersPer1000'] = countsdf['freq'] / per1000

For Kaggle we create "geocodedcities2.csv" with the below

csv = data.to_csv('geocodedcities2.csv')
"""


# In[ ]:


geo_coded_cities = pd.read_csv("../input/geocoded-cities-with-population/geocodedcities2.csv")


# In[ ]:


#check NAs
dropna = geo_coded_cities .replace('', np.nan)
data_clean = dropna.dropna(how = 'any')
data_clean.isna().sum()


# In[ ]:


#Reduce data to only those city with population data
reduced = data_clean['population'] > 0
data_reduced = data_clean[reduced ]


# In[ ]:


# Double checking to ensure lat/longs are floats so can plot our lat/longs
data_reduced['Y_LAT'] = data_reduced['Y_LAT'].astype(float)
data_reduced['X_LONG'] = data_reduced['X_LONG'].astype(float)


# In[ ]:


#create a folium map centered in the USA
usa_center = (37.1369916,-103.8264166)

# create empty map zoomed in on San Francisco
folium_map = folium.Map(location=usa_center,
                        zoom_start=4,tiles='Stamen Terrain')

# add a marker for every record in the reduced data. set color to call out cities with more than 100 players born
for coord in data_reduced.iterrows():
    perCapita = coord[1]['freq']
    popup_text = "CityState: {}<br> Total Players Born: {}<br>"
    popup_text = popup_text.format(coord[1]['CityState'],coord[1]['freq'])
    color="#E37222"
    
    folium.CircleMarker(
        location = [coord[1]['Y_LAT'],coord[1]['X_LONG']],popup=popup_text,radius = coord[1]['freq']/10,color=color,fill=True).add_to(folium_map)
    

folium_map


# # Compare the Population Map (above) with the Per 1000 Person Map (below)

# In[ ]:


#create a folium map centered in the USA
usa_center = (37.1369916,-103.8264166)

# create empty map zoomed in on San Francisco
folium_map = folium.Map(location=usa_center,
                        zoom_start=4,tiles='Stamen Terrain')

# add a marker for every record in the reduced data. set color to call out cities with more than 100 players born
for coord in data_reduced.iterrows():
    perCapita = coord[1]['playersPer1000']
    pop = coord[1]['population']
    popup_text = "CityState: {}<br> Players Born Per 1000 People: {}<br> City Population: {}<br>"
    popup_text = popup_text.format(coord[1]['CityState'],coord[1]['playersPer1000'],coord[1]['population'])
    color="#E37222"
    
    folium.CircleMarker(
        location = [coord[1]['Y_LAT'],coord[1]['X_LONG']],popup=popup_text,radius = coord[1]['playersPer1000']*20,color=color,fill=True).add_to(folium_map)
    

folium_map


# In[ ]:




