#!/usr/bin/env python
# coding: utf-8

# This notebook follows this course: https://www.kaggle.com/learn/geospatial-analysis
# 
# The comments and notes are personal approaches.

# ## Introduction and Goals
# 
# In this notebook, we will explore the coordinate reference system. 
# 
# We have two dataset: 
# 
# 1. The first Geopandas data containing the regions in GHANA using crs : **epsg:32630**
# 2. The second one Pandas data containing the location (lat,long) of the medical facilities, we convert this dataset to geodata using crs: **epsg=4326**
# 
# We locate the facilities in the map of Ghana, we chase the oppurtinuty to explore the Geodataframe, to calculate the area using the polygone from geomtry variable. 
# 
# 

# In[ ]:


import geopandas as gpd
import pandas as pd


# ## Setting CRS

# In[ ]:


# Load a GeoDataFrame containing regions in Ghana
regions = gpd.read_file("../input/geospatial-learn-course-data/ghana/ghana/Regions/Map_of_Regions_in_Ghana.shp")
print(regions.crs)
regions.head()


# In[ ]:


# Create a DataFrame with health facilities in Ghana
facilities_df = pd.read_csv("../input/geospatial-learn-course-data/ghana/ghana/health_facilities.csv")
facilities_df.head()


# In this dataset, we have a longitude and latitude that we wish to transform into Geodataframe. 
# 

# In[ ]:


facilities = gpd.GeoDataFrame(facilities_df, geometry=gpd.points_from_xy(facilities_df.Longitude, facilities_df.Latitude))
# Setting the CRS cordinate reference system to EPSG 4326 

facilities.crs={"init": 'epsg: 4326'}
#changing the dataset from pandas to geopandas
print(type(facilities), type(facilities_df))
facilities.head()


# ### Reprojecting
# To reproject two map in the same plot they must have the same CRS.

# In[ ]:


ax = regions.to_crs(epsg=4326).plot(figsize=(8,8), color='whitesmoke', linestyle=':', edgecolor='black')
facilities.plot(markersize=1, ax=ax)


# * a Point for the epicenter of an earthquake,
# * a LineString for a street, or
# * a Polygon to show country boundaries.
# 
# From a point we can get  coordinates, from a polygon we can calculate the area, from a linestring we can get the lenght. 
# 

# In[ ]:


facilities.geometry.x.head()


# In[ ]:


# Calculate the area (in square meters) of each polygon in the GeoDataFrame 
regions.loc[:, "AREA"] = regions.geometry.area / 10**6

print("Area of Ghana: {} square kilometers".format(regions.AREA.sum()))
print("CRS:", regions.crs)
regions.head()


# In[ ]:




