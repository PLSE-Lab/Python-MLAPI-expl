#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pip install --upgrade pip==9.0.3


# In[ ]:


import os


# In[ ]:


import folium


# In[ ]:


import geopandas as gpd


# In[ ]:


pip install earthpy


# In[ ]:


import earthpy as et


# In[ ]:


# Get the data and set working directory
data = et.data.get_data('spatial-vector-lidar')
os.chdir(os.path.join(et.io.HOME, 'earth-analytics'))


# In[ ]:


# Create interactive map with default basemap
map_osm = folium.Map(location=[-37.840935, 144.946457
])
map_osm


# ### Add Vector Data to Interactive Map

# In[ ]:


# Import SJER plot locations using geopandas
SJER_plot_locations_path = os.path.join("data", "spatial-vector-lidar",
                                        "california", "neon-sjer-site",
                                        "vector_data", "SJER_plot_centroids.shp")

SJER_plot_locations = gpd.read_file(SJER_plot_locations_path)


# In[ ]:


# Project to WGS 84 and save to json for plotting on interactive map
SJER_plot_locations_json = SJER_plot_locations.to_crs(epsg=4326).to_json()

# Create interactive map and add SJER plot locations
SJER_map = folium.Map([-37.840935, 144.946457],
                  zoom_start=14)

points = folium.features.GeoJson(SJER_plot_locations_json)

SJER_map.add_child(points)
SJER_map


# In[ ]:


# Create interactive map with different basemap
SJER_map = folium.Map([-37.840935, 144.946457],
                  zoom_start=14,
                  tiles='Stamen Terrain')

points = folium.features.GeoJson(SJER_plot_locations_json)

SJER_map.add_child(points)
SJER_map

