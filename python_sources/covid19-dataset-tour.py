#!/usr/bin/env python
# coding: utf-8

# **Feacthing the real updated time dataset**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import Point,Polygon

import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster

import requests 


# **World CoronaData**

# In[ ]:


URL = "https://coronavirus-tracker-api.herokuapp.com/v2/locations"


# In[ ]:


# sending get request and saving the response as response object 
r = requests.get(url = URL) 

# extracting data in json format 
data = r.json() 


# In[ ]:


asian_countries = [ 'Afghanistan', 'Armenia', 'Azerbaijan', 'Bahrain', 'Bangladesh', 'Bhutan, Brunei', 'Burma,Cambodia',
                   'China', 'East Timor', 'Georgia', 'Hong Kong','India', 'Indonesia', 'Iran', 'Iraq','Israel', 'Japan', 
                   'Jordan', 'Kazakhstan', 'Kuwait', 'Kyrgyzstan', 'Laos', 'Lebanon', 'Malaysia', 'Mongolia', 'Nepal', 
                   'North Korea', 'Oman', 'Pakistan', 'Papua New Guinea', 'Philippines', 'Qatar', 'Russia', 
                   'Saudi Arabia', 'Singapore', 'South Korea', 'Sri Lanka', 'Syria', 'Taiwan', 'Tajikistan', 
                   'Thailand', 'Turkey', 'Turkmenistan', 'United Arab Emirates', 'Uzbekistan', 'Vietnam', 'Yemen']
south_asian = ['Sri Lanka', 'India', 'Bangladesh', 'Bhutan', 'Nepal', 'Pakistan','Maldives']


# In[ ]:


coordintes_confirmed = []
for values in data['locations']:
    country = values['country']
    latitude = float((values['coordinates'])['latitude'])
    longitude = float((values['coordinates'])['longitude'])
    confirmed = int((values['latest'])['confirmed'])
    deaths = int((values['latest'])['deaths'])
    if(values['province'] != ''):
        province = values['province']
    else:
        province = 'Nan'
    coordintes_confirmed.append((country,latitude,longitude,province,confirmed,deaths))


# **Make Pandas DataFrame from the list**

# In[ ]:


world_frame = pd.DataFrame(coordintes_confirmed, columns=['Country', 'Latitude', 'Longitude', 'Province','Confired_Cases','Deaths'])


# In[ ]:


world_frame.head()


# **Copy in other dataset for further analysis for Maps**

# In[ ]:


world_frame_map = world_frame.copy()


# In[ ]:


world_frame_map.head()


# **Save a copy of final Dataframe with coordiantes**

# In[ ]:


world_frame_map_copy = world_frame_map.copy()
geometry_copy = [Point(xy) for xy in zip(world_frame_map.Latitude, world_frame_map.Longitude)]
crs = {'init' : 'epsg:4326'}
gdf_world_copy = GeoDataFrame(world_frame_map_copy, crs=crs, geometry=geometry_copy)


# In[ ]:


gdf_world_copy.head()


# In[ ]:


# save the GeoDataFrame
gdf_world_copy.to_file(driver = 'ESRI Shapefile', filename= "CoronaVirus.shp")


# **Couting Country with no of confirmed Case**

# In[ ]:


world_frame_map = world_frame_map.reindex(world_frame.index.repeat(world_frame.Confired_Cases))


# **Readint the datainto GeoDataFrame**

# In[ ]:


geometry = [Point(xy) for xy in zip(world_frame_map.Latitude, world_frame_map.Longitude)]
world_frame_map = world_frame_map.drop(['Confired_Cases', 'Province', 'Deaths'], axis = 1)
crs = {'init' : 'epsg:4326'}
gdf_world = GeoDataFrame(world_frame_map, crs=crs, geometry=geometry)


# **Preprocessing for Choropleth maps**

# In[ ]:


gdf_world_indexed = gdf_world.set_index('Country')


# **Count No of Confirmed in each Country**

# In[ ]:


plot_dict = gdf_world.Country.value_counts()
plot_dict.head()


# In[ ]:


#read world map shape file
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))


# **HeatMap World**

# In[ ]:


# Create a base map
m_2 = folium.Map(location=[25.0376,76.4563], tiles='openstreetmap', zoom_start=2)

# Add a heatmap to the base map
HeatMap(data=world_frame_map[['Latitude', 'Longitude']], radius=10).add_to(m_2)

# Display the map
m_2


# In[ ]:




