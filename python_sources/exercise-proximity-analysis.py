#!/usr/bin/env python
# coding: utf-8

# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# ---
# 

# # Introduction 
# 
# You are part of a crisis response team, and you want to identify how hospitals have been responding to crash collisions in New York City.
# 
# <center>
# <img src="https://i.imgur.com/wamd0n7.png" width="450"><br/>
# </center>
# 
# Before you get started, run the code cell below to set everything up.

# In[ ]:


import math
import geopandas as gpd
import pandas as pd
from shapely.geometry import MultiPolygon

import folium
from folium import Choropleth, Marker
from folium.plugins import HeatMap, MarkerCluster

from learntools.core import binder
binder.bind(globals())
from learntools.geospatial.ex5 import *


# You'll use the `embed_map()` function to visualize your maps.

# In[ ]:


def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


# # Exercises
# 
# ### 1) Visualize the collision data.
# 
# Run the code cell below to load a GeoDataFrame `collisions` tracking major motor vehicle collisions in 2013-2018.

# In[ ]:


collisions = gpd.read_file("../input/geospatial-learn-course-data/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions/NYPD_Motor_Vehicle_Collisions.shp")
collisions.head()


# Use the "LATITUDE" and "LONGITUDE" columns to create an interactive map to visualize the collision data.  What type of map do you think is most effective?

# In[ ]:


m_1 = folium.Map(location=[40.7, -74], zoom_start=11) 

# Your code here: Visualize the collision data
____

# Uncomment to see a hint
#q_1.hint()

# Show the map
embed_map(m_1, "q_1.html")


# In[ ]:


# Get credit for your work after you have created a map
q_1.check()

# Uncomment to see our solution (your code may look different!)
#q_1.solution()


# ### 2) Understand hospital coverage.
# 
# Run the next code cell to load the hospital data.

# In[ ]:


hospitals = gpd.read_file("../input/geospatial-learn-course-data/nyu_2451_34494/nyu_2451_34494/nyu_2451_34494.shp")
hospitals.head()


# Use the "latitude" and "longitude" columns to visualize the hospital locations. 

# In[ ]:


m_2 = folium.Map(location=[40.7, -74], zoom_start=11) 

# Your code here: Visualize the hospital locations
____

# Uncomment to see a hint
#q_2.hint()
        
# Show the map
embed_map(m_2, "q_2.html")


# In[ ]:


# Get credit for your work after you have created a map
q_2.check()

# Uncomment to see our solution (your code may look different!)
#q_2.solution()


# ### 3) When was the closest hospital more than 10 kilometers away?
# 
# Create a DataFrame `outside_range` containing all rows from `collisions` with crashes that occurred more than 10 kilometers from the closest hospital.
# 
# Note that both `hospitals` and `collisions` have EPSG 2263 as the coordinate reference system, and EPSG 2263 has units of meters.

# In[ ]:


# Your code here
outside_range = ____

# Check your answer
q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()


# The next code cell calculates the percentage of collisions that occurred more than 10 kilometers away from the closest hospital.

# In[ ]:


percentage = round(100*len(outside_range)/len(collisions), 2)
print("Percentage of collisions more than 10 km away from the closest hospital: {}%".format(percentage))


# ### 4) Make a recommender.
# 
# When collisions occur in distant locations, it becomes even more vital that injured persons are transported to the nearest available hospital.
# 
# With this in mind, you decide to create a recommender that:
# - takes the location of the crash (in EPSG 2263) as input,
# - finds the closest hospital (where distance calculations are done in EPSG 2263), and 
# - returns the name of the closest hospital. 

# In[ ]:


def best_hospital(collision_location):
    # Your code here
    name = ____
    return name

# Test your function: this should suggest CALVARY HOSPITAL INC
print(best_hospital(outside_range.geometry.iloc[0]))

# Check your answer
q_4.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_4.hint()
#q_4.solution()


# ### 5) Which hospital is under the highest demand?
# 
# Considering only collisions in the `outside_range` DataFrame, which hospital is most recommended?  
# 
# Your answer should be a Python string that exactly matches the name of the hospital returned by the function you created in **4)**.

# In[ ]:


# Your code here
highest_demand = ____

# Check your answer
q_5.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_5.hint()
#q_5.solution()


# ### 6) Where should the city construct new hospitals?
# 
# Run the next code cell (without changes) to visualize hospital locations, in addition to collisions that occurred more than 10 kilometers away from the closest hospital. 

# In[ ]:


m_6 = folium.Map(location=[40.7, -74], zoom_start=11) 

coverage = gpd.GeoDataFrame(geometry=hospitals.geometry).buffer(10000)
folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m_6)
HeatMap(data=outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m_6)
folium.LatLngPopup().add_to(m_6)

embed_map(m_6, 'm_6.html')


# Click anywhere on the map to see a pop-up with the corresponding location in latitude and longitude.
# 
# The city of New York reaches out to you for help with deciding locations for two brand new hospitals.  They specifically want your help with identifying locations to bring the calculated percentage from step **3)** to less than ten percent.  Using the map (and without worrying about zoning laws or what potential buildings would have to be removed in order to build the hospitals), can you identify two locations that would help the city accomplish this goal?  
# 
# Put the proposed latitude and longitude for hospital 1 in `lat_1` and `long_1`, respectively.  (Likewise for hospital 2.)
# 
# Then, run the rest of the cell as-is to see the effect of the new hospitals.  Your answer will be marked correct, if the two new hospitals bring the percentage to less than ten percent.

# In[ ]:


# Your answer here: proposed location of hospital 1
lat_1 = ____
long_1 = ____

# Your answer here: proposed location of hospital 2
lat_2 = ____
long_2 = ____


# Do not modify the code below this line
try:
    new_df = pd.DataFrame(
        {'Latitude': [lat_1, lat_2],
         'Longitude': [long_1, long_2]})
    new_gdf = gpd.GeoDataFrame(new_df, geometry=gpd.points_from_xy(new_df.Longitude, new_df.Latitude))
    new_gdf.crs = {'init' :'epsg:4326'}
    new_gdf = new_gdf.to_crs(epsg=2263)
    # get new percentage
    new_coverage = gpd.GeoDataFrame(geometry=new_gdf.geometry).buffer(10000)
    new_my_union = new_coverage.geometry.unary_union
    new_outside_range = outside_range.loc[~outside_range["geometry"].apply(lambda x: new_my_union.contains(x))]
    new_percentage = round(100*len(new_outside_range)/len(collisions), 2)
    print("(NEW) Percentage of collisions more than 10 km away from the closest hospital: {}%".format(new_percentage))
    # Did you help the city to meet its goal?
    q_6.check()
    # make the map
    m = folium.Map(location=[40.7, -74], zoom_start=11) 
    folium.GeoJson(coverage.geometry.to_crs(epsg=4326)).add_to(m)
    folium.GeoJson(new_coverage.geometry.to_crs(epsg=4326)).add_to(m)
    for idx, row in new_gdf.iterrows():
        Marker([row['Latitude'], row['Longitude']]).add_to(m)
    HeatMap(data=new_outside_range[['LATITUDE', 'LONGITUDE']], radius=9).add_to(m)
    folium.LatLngPopup().add_to(m)
    display(embed_map(m, 'q_6.html'))
except:
    q_6.hint()


# In[ ]:


# Uncomment to see one potential answer 
#q_6.solution()


# # Congratulations!
# 
# You have just completed the Geospatial Analysis micro-course!  Great job!

# ---
# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum/161464) to chat with other Learners.*
