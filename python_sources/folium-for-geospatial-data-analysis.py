#!/usr/bin/env python
# coding: utf-8

# **What is Folium?.**
# 
# Folium is a powerful Python library that helps you create several types of Leaflet maps. The fact that the Folium results are interactive makes this library very useful for dashboard building.
# 
# Plan : 
# 1. I'm going to use the geospacial data for this tutorial, There are 100 rows in the selected dataset(i.e 100 different co-ordinates).
# 1. I will use Pandas groupby to categorize the venues(example : Hotels, Beaches, Parks).
# 1. Locate the categories with different colors(ex : Hotels in red, Parks in green color and Beech in blue color).
# 1. Distance calculation using Geopy module.
# 1. Finding the nearest path from my current location.
# 

# In[ ]:


# install required modules
get_ipython().system('pip install geopy')
get_ipython().system('pip install folium')


# In[ ]:


# import libraries
import numpy as np 
import pandas as pd 
import folium as fl
import matplotlib.pyplot as plt
import seaborn as sns
from geopy.distance import GreatCircleDistance


# In[ ]:


#create DataFrame
df = pd.read_csv("../input/bournemouth-venues/bournemouth_venues.csv")
df.head(5)


# In[ ]:


# rename columns
df = df.rename(columns = {'Venue Latitude':'latitude','Venue Longitude': 'longitude', 'Venue Category': 'category','Venue Name':'place'})

# Visualize the categories
print(df.category.value_counts().iloc[0:11])
print("total categories :",df.category.value_counts().shape)
fig = plt.figure(figsize = (20,5))
sns.countplot(df["category"][0:10])


# There are 52 categories, Example : Hotels, Beach and Parks etc.
# I will select Hotels, Beaches and Parks and visualize them in the folium map object.
# * Hotels in Red color
# * Beach in Blue color
# * Parks in Green color
# 
# Syntax : Folium.Map(Location = [latitude,longitude], zoom_start = 6)

# In[ ]:


# Folium map
map = fl.Map([50.720913,-1.879085],zoom_start = 15)

# grouping dataframe by category
df = df.groupby("category")
map


# In[ ]:


# extracting  all the rows with hotels
hotels = df.get_group('Hotel')
print(hotels.head(5))

# Separating the hotel locations and converting each attribute into list
lat = list(hotels["latitude"])
lon = list(hotels["longitude"])
place = list(hotels["place"])
cat = list(hotels["category"])

# visualize / locate hotels--->Markers in red
for lt,ln,pl,cat in zip(lat,lon,place,cat):
    fl.Marker(location = [lt,ln], tooltip = str(pl) +","+str(cat), icon = fl.Icon(color = 'red')).add_to(map)
map 


# In[ ]:


parks = df.get_group('Park')

# Separating the all park locations and converting each attribute into list
lat = list(parks["latitude"])
lon = list(parks["longitude"])
place = list(parks["place"])
cat = list(parks["category"])
print(parks)

# parks in green colors
for cat,lt,ln,pl in zip(cat,lat,lon,place):
    fl.Marker(location = [lt,ln], tooltip = str(pl) +","+str(cat), icon = fl.Icon(color = 'green')).add_to(map)
map


# In[ ]:


beach = df.get_group('Beach')
lat_beach = list(beach["latitude"])
lon_beach = list(beach["longitude"])
place = list(beach["place"])
cat = list(beach["category"])
print(beach)

# Beach in blue color
for cat,lt,ln,pl in zip(cat,lat_beach,lon_beach,place):
    fl.Marker(location = [lt,ln], tooltip = str(pl) +","+str(cat), icon = fl.Icon(color = 'blue')).add_to(map)
map


# **Calculating distance between two locations using Geopy**
# 
# My goal is to find the nearest beach from my current location(My location == Hallmark Hotel), I'm going to use the GreatCircleDistance to calculate the distance between two locations.
# Destination points/locations are all the beaches in Bournemouth, and the source location is Hallmark Hotel((50.718742,-1.890372))).
# For more information about GreatCircledistance [click here](https://en.wikipedia.org/wiki/Great-circle_distance).
# I'm going to use PolyLine function of the folium module to draw a line between points, Line with color green indicates the nearest distance to the destination(beach) from the source(Hotel - hallmark).
# 
# *Note : Distance is calculated in kilometers not miles*
# 

# In[ ]:


# latitude and longitude of Hallmark Hotel
Source = (50.718742,-1.890372)

# Empty list to store the distance
distance = []
for lt,ln in zip(lat_beach,lon_beach):
    dist = GreatCircleDistance(Source,(lt,ln))
    distance.append(dist)

# Draw lines between points
for dist,lt,ln in zip(distance,lat_beach,lon_beach):
    if (dist > 0) and (dist <= 0.6):
        fl.PolyLine([Source,(lt,ln),],color = "green", weight = 4).add_to(map)  
    elif (dist > 0.6) and (dist <= 0.9):
        fl.PolyLine([Source,(lt,ln)],color = "orange", weight = 3).add_to(map)
    else :
        fl.PolyLine([Source,(lt,ln)],color = "red", weight = 2).add_to(map)
map


# In[ ]:


distance


# **Work in progress, please upvote if you like it.**
