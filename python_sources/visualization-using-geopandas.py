import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
print(os.listdir("../input"))


# Loading the Dataset
ffr = pd.read_csv('../input/fast-food-restaurants/FastFoodRestaurants.csv')
ffr = ffr.drop(ffr.columns[[3,9]],axis=1)  # dropping columns 3 and 9
ffr.head()

# checking for the highest number of restaurants
ffr['name'].value_counts()


# Visualisation Using GeoPandas & MatPlotLib
import matplotlib.pyplot as plt
import descartes
import geopandas as gpd
from shapely.geometry import Point, Polygon

%matplotlib inline

# In GeoPandas, you need to import a .shp file to plot on. You can find these kinds of files on Google!
state_map = gpd.read_file('../input/shape-files/states.shp')
fig,ax = plt.subplots(figsize = (15,15))
state_map.plot(ax = ax)


crs = {'init': 'epsg:4326'} #Coordinate Reference Systemgpd.read_file('../input/shape-files/states.shp')
geometry = [Point(xy) for xy in zip( ffr["longitude"], ffr["latitude"])]
geometry [:3]


# Creating the Geo Dataset for plotting using GeoPandas
geo_ffr = gpd.GeoDataFrame(ffr, crs = crs, geometry = geometry)
geo_ffr.head()


# Visualisation for McDonalds
fig,ax = plt.subplots(figsize = (15,15))
state_map.plot(ax = ax, alpha = 0.4, color = "grey")
geo_ffr[geo_ffr['name'] == "McDonalds"].plot(ax=ax,  markersize=1, color = "black",  marker = "o", label = "McD")
plt.legend(prop={'size':15})

#Visualisation for Burger King
fig,ax = plt.subplots(figsize = (15,15))
state_map.plot(ax = ax, alpha = 0.4, color = "grey")
geo_ffr[geo_ffr['name'] == "Burger King"].plot(ax=ax, markersize=2, color = "blue", marker = "o", label = "BK")
plt.legend(prop={'size':15})



#I think there is an issue with the dataset. There were quite a few challenges with the dataset. 
#First names of restaurants repeated many time. For example, Mcdonalds, McDonald's and McDonalds were all same 
#variables under the restaurant names variable. 
#It occured for many other restaurants, and i found it exhausting to run a loop in python to change the names. 
#Instead, I changed it through the excel function of Find and Replace. 

#After using GeoPandas for visualisation, as the plot suggest, some of the points are somewhere in Europe, 
#Central America, or Asia. The Map for Burger King has a longer frame, because one of the lat and lon point is 
#probably in Australia. Hence, the lat and lon data was not just for US, but few other points which makes for 
#ugly visualisation. 