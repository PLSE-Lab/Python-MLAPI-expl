#!/usr/bin/env python
# coding: utf-8

# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# ---
# 

# # Introduction
# 
# You are an urban safety planner in Japan, and you are analyzing which areas of Japan need extra earthquake reinforcement.  Which areas are both high in population density and prone to earthquakes?
# 
# <center>
# <img src="https://i.imgur.com/Kuh9gPj.png" width="450"><br/>
# </center>
# 
# Before you get started, run the code cell below to set everything up.

# In[ ]:


import pandas as pd
import geopandas as gpd

import folium
from folium import Choropleth
from folium.plugins import HeatMap

from learntools.core import binder
binder.bind(globals())
from learntools.geospatial.ex3 import *


# You'll use the `embed_map()` function from the tutorial to visualize your maps.

# In[ ]:


def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


# # Exercises
# 
# ### 1) Do earthquakes coincide with plate boundaries?
# 
# Run the code cell below to create a DataFrame `plate_boundaries` that shows global plate boundaries.  The "coordinates" column is a list of (latitude, longitude) locations along the boundaries.

# In[ ]:


plate_boundaries = gpd.read_file("../input/geospatial-learn-course-data/Plate_Boundaries/Plate_Boundaries/Plate_Boundaries.shp")
plate_boundaries['coordinates'] = plate_boundaries.apply(lambda x: [(b,a) for (a,b) in list(x.geometry.coords)], axis='columns')
plate_boundaries.drop('geometry', axis=1, inplace=True)

plate_boundaries.head()


# Next, run the code cell below without changes to load the historical earthquake data into a DataFrame `earthquakes`.

# In[ ]:


# Load the data and print the first 5 rows
earthquakes = pd.read_csv("../input/geospatial-learn-course-data/earthquakes1970-2014.csv", parse_dates=["DateTime"])
earthquakes.head()


# The code cell below visualizes the plate boundaries on a map.  Use all of the earthquake data to add a heatmap to the same map, to determine whether earthquakes coincide with plate boundaries.  

# In[ ]:


# Create a base map with plate boundaries
m_1 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)
for i in range(len(plate_boundaries)):
    folium.PolyLine(locations=plate_boundaries.coordinates.iloc[i], weight=2, color='black').add_to(m_1)

# Your code here: Add a heatmap to the map
HeatMap(data=earthquakes[['Latitude','Longitude']],radius=13).add_to(m_1)

# Uncomment to see a hint
#q_1.a.hint()

# Show the map
embed_map(m_1, 'q_1.html')


# In[ ]:


# Get credit for your work after you have created a map
q_1.a.check()

# Uncomment to see our solution (your code may look different!)
#q_1.a.solution()


# So, given the map above, do earthquakes coincide with plate boundaries?

# In[ ]:


# View the solution
q_1.b.solution()


# ### 2) Is there a relationship between earthquake depth and proximity to a plate boundary in Japan?
# 
# You recently read that the depth of earthquakes tells us [important information](https://www.usgs.gov/faqs/what-depth-do-earthquakes-occur-what-significance-depth?qt-news_science_products=0#qt-news_science_products) about the structure of the earth.  You're interested to see if there are any intereresting global patterns, and you'd also like to understand how depth varies in Japan.
# 
# 

# In[ ]:


# Create a base map with plate boundaries
m_2 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)
for i in range(len(plate_boundaries)):
    folium.PolyLine(locations=plate_boundaries.coordinates.iloc[i], weight=2, color='black').add_to(m_2)
    
# Custom function to assign a color to each circle
def color_producer(val):
    if val < 50:
        return 'forestgreen'
    elif val < 100:
        return 'darkorange'
    else:
        return 'darkred'

# Add a map to visualize earthquake depth
for i in range(0,len(earthquakes)):
    folium.Circle(
        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],
        radius=2000,
        color=color_producer(earthquakes.iloc[i]['Depth'])).add_to(m_2)
# Uncomment to see a hint
q_2.a.hint()

# View the map
embed_map(m_2, 'q_2.html')


# In[ ]:


# Get credit for your work after you have created a map
q_2.a.check()

# Uncomment to see our solution (your code may look different!)
#q_2.a.solution()


# Can you detect a relationship between proximity to a plate boundary and earthquake depth?  Does this pattern hold globally?  In Japan?

# In[ ]:


# View the solution
q_2.b.solution()


# ### 3) Which prefectures have high population density?
# 
# Run the next code cell (without changes) to create a GeoDataFrame `prefectures` that contains the geographical boundaries of Japanese prefectures.

# In[ ]:


# GeoDataFrame with prefecture boundaries
prefectures = gpd.read_file("../input/geospatial-learn-course-data/japan-prefecture-boundaries/japan-prefecture-boundaries/japan-prefecture-boundaries.shp")
prefectures.set_index('prefecture', inplace=True)
prefectures.head()


# The next code cell creates a DataFrame `stats` containing the population, area (in square kilometers), and population density (per square kilometer) for each Japanese prefecture.  Run the code cell without changes.

# In[ ]:


# DataFrame containing population of each prefecture
population = pd.read_csv("../input/geospatial-learn-course-data/japan-prefecture-population.csv")
population.set_index('prefecture', inplace=True)

# Calculate area (in square kilometers) of each prefecture
area_sqkm = pd.Series(prefectures.geometry.to_crs(epsg=32654).area / 10**6, name='area_sqkm')
stats = population.join(area_sqkm)

# Add density (per square kilometer) of each prefecture
stats['density'] = stats["population"] / stats["area_sqkm"]
stats.head()


# Use the next code cell to create a choropleth map to visualize population density.

# In[ ]:


# Create a base map
m_3 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

# Your code here: create a choropleth map to visualize population density


Choropleth(geo_data= prefectures.__geo_interface__,
        data= stats['density'],
        key_on="feature.id", 
        fill_color='YlGnBu', 
        legend_name= "Population Density"
          ).add_to(m_3)

          

# Uncomment to see a hint
#q_3.a.hint()

# View the map
embed_map(m_3, 'q_3.html')


# In[ ]:


q_3.a.hint()


# In[ ]:


# Get credit for your work after you have created a map
q_3.a.check()

# Uncomment to see our solution (your code may look different!)
#q_3.a.solution()


# Which three prefectures have relatively higher density than the others?  Are they spread throughout the country, or all located in roughly the same geographical region?  (*If you're unfamiliar with Japanese geography, you might find [this map](https://en.wikipedia.org/wiki/Prefectures_of_Japan) useful to answer the questions.)*

# In[ ]:


# View the solution
q_3.b.solution()


# ### 4) Which high-density prefecture is prone to high-magnitude earthquakes?
# 
# Create a map to suggest one prefecture that might benefit from earthquake reinforcement.  Your map should visualize both density and earthquake magnitude.

# In[ ]:


# Create a base map
m_4 = folium.Map(location=[35,136], tiles='cartodbpositron', zoom_start=5)

# Your code here: create a map
Choropleth(geo_data= prefectures.__geo_interface__,
        data= stats['density'],
        key_on="feature.id", 
        fill_color='YlGnBu', 
        legend_name= "Population Density"
          ).add_to(m_4)

def color_producer(magnitude):
    if magnitude > 6.5:
        return 'red'
    else:
        return 'green'
for i in range(0,len(earthquakes)):
    folium.Circle(
        location=[earthquakes.iloc[i]['Latitude'], earthquakes.iloc[i]['Longitude']],
         popup=("{} ({})").format(
            earthquakes.iloc[i]['Magnitude'],
            earthquakes.iloc[i]['DateTime'].year),
        radius=earthquakes.iloc[i]['Magnitude']**5.5,
        color=color_producer(earthquakes.iloc[i]['Magnitude'])).add_to(m_4)

# Uncomment to see a hint
#q_4.a.hint()

# View the map
embed_map(m_4, 'q_4.html')


# In[ ]:


# Get credit for your work after you have created a map
q_4.a.check()

# Uncomment to see our solution (your code may look different!)
#q_4.a.solution()


# Which prefecture do you recommend for extra earthquake reinforcement?

# In[ ]:


# View the solution
q_4.b.solution()


# # Keep going
# 
# Learn how to convert names of places to geographic coordinates with **[geocoding](https://www.kaggle.com/alexisbcook/manipulating-geospatial-data)**.  You'll also explore special ways to join information from multiple GeoDataFrames.

# ---
# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
