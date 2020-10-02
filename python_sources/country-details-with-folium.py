#!/usr/bin/env python
# coding: utf-8

# # Folium Basics
# ### In this short kernel I would like to introduce Folium and how to plot different types of World Maps and markers. 

# #### Folium provides and easy way of creating world maps. Go through their docs [here](http://python-visualization.github.io/folium/docs-v0.5.0/modules.html)

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# #### The world bank data set has three csv files

# In[ ]:


import os
print(os.listdir("../input"))


# We will first choose country-population.csv for world maps and then go on to others to plot each. Lets explore the csv and see what all columns exist. 

# In[ ]:


df = pd.read_csv("../input/country_population.csv")
df.head()


# We will take the 2016 population and make it display on our worldmap. But first lets obtain the latitudes and longitudes of each of our countries. Since we are plotting the whole of the country we dont have to worry about the states. For this lets use **geopy**

# In[ ]:


from geopy.geocoders import Nominatim
geolocator = Nominatim(user_agent="Worldmap for countries population in 2016")
latitude = []
long = []
for i in df["Country Name"]:
    if i != None:
        location = geolocator.geocode(i)
        if location!=None:
            latitude.append(location.latitude)#, location.longitude)
            long.append(location.longitude)
        else:
            latitude.append(float("Nan"))#, location.longitude)
            long.append(float("Nan"))
    else:
        latitude.append(float("Nan"))#, location.longitude)
        long.append(float("Nan"))


# In[ ]:


df["Latitude"] = latitude
df.head()


# In[ ]:


df["Longitude"] = long
df.head()


# In[ ]:


type(df.iloc[1]["Longitude"])


# In[ ]:


df = df.dropna(axis=0)


# In[ ]:


import folium
world_map = folium.Map(location=[20, 0], tiles="Mapbox Bright", zoom_start=4)
for i in range(0,len(df)):
    folium.Marker([df.iloc[i]['Latitude'], df.iloc[i]['Longitude']], popup = "Population - " + str(df.iloc[i]['2016'])).add_to(world_map)
world_map


# In[ ]:


df[df["Country Code"] == "SVN"]


# In[ ]:


world_map.choropleth(
geo_data = df,
 name='choropleth',
 data=df,
 columns=['Country Name', '2016'],
 key_on='2016',
 fill_color='YlGn',
 fill_opacity=0.7,
 line_opacity=0.2,
 legend_name='Unemployment Rate (%)'
)

