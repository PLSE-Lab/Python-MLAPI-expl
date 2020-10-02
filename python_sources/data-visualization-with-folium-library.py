#!/usr/bin/env python
# coding: utf-8

# To create maps for different objectives using ***Python visualization library, namely Folium***. What is nice about Folium is that it was developed for the sole purpose of visualizing geospatial data. While other libraries are available to visualize geospatial data, such as plotly, they might have a cap on how many API calls you can make within a defined time frame. Folium, on the other hand, is completely free.

# **Folium** is a powerful Python library that helps you create several types of Leaflet maps. The fact that the Folium results are interactive makes this library very useful for dashboard building.
# 
# 

# From the official Folium documentation page:
# 
# * Folium builds on the data wrangling strengths of the Python ecosystem and the mapping strengths of the Leaflet.js library. Manipulate your data in Python, then visualize it in on a Leaflet map via Folium.
# 
# * Folium makes it easy to visualize data that's been manipulated in Python on an interactive Leaflet map. It enables both the binding of data to a map for choropleth visualizations as well as passing Vincent/Vega   visualizations as markers on the map.
# 
# * The library has a number of built-in tilesets from OpenStreetMap, Mapbox, and Stamen, and supports custom tilesets with Mapbox or Cloudmade API keys. Folium supports both GeoJSON and To

# **Folium** is not available by default. So, we first need to install it before we are able to import it.

# By default tiles is **OpenStreetMap**. Here is an example of OpenStreetMap tiles.

# In[ ]:


# import folium package 
import folium


# Example of **Stamen Toner** Tiles.

# In[ ]:


a = folium.Map(
    location=[23.8859, 45.0792],
    tiles='Stamen Toner',
    zoom_start=6
)
# folium CircleMarker
folium.CircleMarker(
    # radius of CircleMarker
    radius=15,
    # Coordinates of location Masjid-al-Haram
    location=[21.4225, 39.8262],
    popup='Masjid-al-Haram',
    # color of CircleMarker
    color='crimson',
    fill=False,
).add_to(a)

folium.CircleMarker(
    # Coordinates of Masjid Nabwi
    location=[24.4672, 39.6111],
    # radius of CircleMarker
    radius=15,
    popup='',
    # color of CircleMarker
    color='#3186cc',
    fill=True,
    fill_color='#3186cc'
).add_to(a)

a


# Example of Stamen Terrain.

# To display **Golden Quadriateral** using **Stamen Terrain tiles** along with Parachute style Mark. 

# In[ ]:


l = folium.Map(
    location=[20.5937, 78.9629],
    zoom_start=4.5,
    tiles='Stamen Terrain'
)

folium.Marker(
    location=[19.0760, 72.8777],
    popup=folium.Popup(max_width=450).add_child
).add_to(l)

folium.Marker(
    location=[28.7041, 77.1025],
    popup=folium.Popup(max_width=450).add_child
).add_to(l)

folium.Marker(
    location=[22.5726, 88.3639],
    popup=folium.Popup(max_width=450).add_child
).add_to(l)

folium.Marker(
    location=[13.0827, 80.2707],
    popup=folium.Popup(max_width=450).add_child
).add_to(l)

# Add line to map
folium.PolyLine(locations = [(19.0760, 72.8777), (28.7041, 77.1025), (22.5726, 88.3639), (13.0827, 80.2707), (19.0760, 72.8777)], 
                line_opacity = 0.5).add_to(l) 

l


# In[ ]:


# Kaaba View
m=folium.Map(
    location=[21.4225, 39.8262],
    zoom_start=16.5
)
m


# In[ ]:




