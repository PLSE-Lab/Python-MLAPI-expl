#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# In[ ]:


# Loading the Dataset
ffr = pd.read_csv('../input/fast-food-restaurants/FastFoodRestaurants.csv')
ffr = ffr.drop(ffr.columns[[3,9]],axis=1)  # dropping columns 3 and 9
ffr.head()


# In[ ]:


# checking for the highest number of restaurants
ffr['name'].value_counts()


# In[ ]:


# Visualisation Using GeoPandas & MatPlotLib
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# In GeoPandas, you need to import a .shp file to plot on. You can find these kinds of files on Google!
state_map = gpd.read_file('../input/shape-files/states.shp')
fig,ax = plt.subplots(figsize = (15,15))
state_map.plot(ax = ax)


# In[ ]:


crs = {'init': 'epsg:4326'} #Coordinate Reference System
geometry = [Point(xy) for xy in zip( ffr["longitude"], ffr["latitude"])]
geometry [:3]


# In[ ]:


# Creating the Geo Dataset for plotting using GeoPandas
geo_ffr = gpd.GeoDataFrame(ffr, crs = crs, geometry = geometry)
geo_ffr.head()


# In[ ]:


# Visualisation for McDonalds
fig,ax = plt.subplots(figsize = (15,15))
state_map.plot(ax = ax, alpha = 0.4, color = "grey")
geo_ffr[geo_ffr['name'] == "McDonald's"].plot(ax=ax,  markersize=1, color = "black",  marker = "o", label = "McD")
plt.legend(prop={'size':15})


# In[ ]:


#Visualisation for Burger King
fig,ax = plt.subplots(figsize = (15,15))
state_map.plot(ax = ax, alpha = 0.4, color = "grey")
geo_ffr[geo_ffr['name'] == "Burger King"].plot(ax=ax, markersize=2, color = "blue", marker = "o", label = "BK")
plt.legend(prop={'size':15})


# **Conclusion**
# 
# The objective was to plot using the latitude and the longitude points using GeoPandas library. We mainly faced two types of issues doing that:
# 
# 1. The 'names' of restaurants repeated many time. For example, Mcdonalds, McDonald's and McDonalds were all same variables under the restaurant 'names' variable. It occured for many other restaurants, and i found it exhausting to run a loop in python to change the names. Instead, you can change it through the excel function of Find and Replace. 
# 
# 2. After using GeoPandas for visualisation, as the plot suggest, some of the points are somewhere in Europe, Central America, or Asia. The Map for Burger King has a longer frame, because one of the lat and lon point is probably in Australia. One way to solve that is by limiting the ability of the code to output lattitudes or longitudes outside the points of mainland US.  
# 
# 
# Do leave a comment for questions, suggestions and feedback. Cheers!!
