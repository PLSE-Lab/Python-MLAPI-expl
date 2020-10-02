#!/usr/bin/env python
# coding: utf-8

# # How to get data from OpenStreetMap
# 
# In this notebook I'd like to show how to query data from OpenStreetMap from the browser with [Overpass Turbo](https://overpass-turbo.eu/) <br>
# A bunch of different data can be queried, from highways to districts boundaries or amenities like pubs, benches, restaurants. <br>
# In this example I'll use data of pubs in the Great Britain to show how to deal with Points, and then, data about streets of London to show how to deal with LineStrings.

# # Great Britain pubs
# First of all we need to build the query processed by Overpass Turbo.<br>
# On the site we can see that there is an editor on the left that allows you to write your query or generate it with the wizard.<br>
# 
# 

# ![](https://imgur.com/dWexWAt.png)
# <br>
# <br>
# In the query we can see the country specification **area["ISO3166-1"="GB"]** that stands for Great Britain<br>
# Alternatively we can manually select a rectangular region (bbox), in this case the query changes a bit.<br>
# *The region can be selected clicking the fifth small button on the upper-left of the map.*
# ![](https://imgur.com/xJ0AsxS.png)

# We can see from the map all the nodes selected by the query.<br>
# ![](https://imgur.com/TcDoRfS.png)

# If we zoom in we can select a node to see all the data:<br> 
# ![](https://imgur.com/J4ZxWUd.png)<br>
# For our analisys, the coordinates of the points are enough.

# So we can export the data as a GEOJSON file by clicking on **Export** in the upper bar and **download as GeoJSON** on the pop-up window that appears.
# ![](https://imgur.com/Ggeswzf.png)

# Now we have tha GeoJSON file so we can start coding.

# In[ ]:


import geojson #used to handle geojson files
import matplotlib.pyplot as plt #used to plot the final graphs
from shapely.geometry import LineString #used to handle shape and path objects
import pandas as pd 
import seaborn as sns
import numpy as np

from datashader.utils import lnglat_to_meters as webm


# Loading file with geojson library

# In[ ]:


with open("../input/osm-overpass-tutorial-dataset/pubs.geojson") as file:
    data = geojson.load(file)


# Let's see how data is saved inside the file.

# In[ ]:


data.keys()


# We are interested in ***features***.<br>
# Every feature is composed by:
# * a **type** (in this case, always "feature"), 
# * a unique **id**,
# * a **geometry** that specify if the object is a point, a shape or a path with the related coordinates,
# * the **properties** that are all the informations we have seen clicking on a node on the map

# In[ ]:


data.features[0].keys()


# As said before we only need the coordinates, so we can collect all of them.<br>
# We cast all geometry coordinates to Point, this is not strictly necessary but is consistent with the method used later to plot paths and shapes.

# In[ ]:


nodes = []
for node in data.features:
    nodes.append(node.geometry.coordinates)


# Now we can plot the points to see the results.

# In[ ]:


from matplotlib.colors import LogNorm

df = pd.DataFrame(nodes)
df.columns = ["Longitude", "Latitude"]
df.Longitude, df.Latitude = webm(df.Longitude, df.Latitude)
df["Lat_binned"] = pd.cut(df.Latitude, 150)
df["Lon_binned"] = pd.cut(df.Longitude, 150)

df = df.pivot_table(
        values='Latitude', 
        index='Lat_binned', 
        columns='Lon_binned', 
        aggfunc=np.size)
df = df[::-1] #reverse latitude values
df = df.fillna(1) #pivoting produces nans that needs to be converted to values to be displayed (I cannot fill with zero because the color scale is logarithmic)

fig, ax = plt.subplots(figsize=(9, 12.24))
log_norm = LogNorm(vmin=df.min().min(), vmax=df.max().max())
sns.heatmap(df, norm = log_norm, ax = ax)
plt.axis("off");


# # London streets
# 
# The data extraction procedure is exactly the same, the only thing that changes is the query (you don't say) <br>
# In  this case I will use a bbox to select the region I need.<br>
# The larger the area, the longer it takes to return the results, so be careful. For this reason, in the query I omitted some streets types.<br>
# All the info about diffentent type of highways and railways can be found on the official OpenStreetMap Wiki [Highway section](https://wiki.openstreetmap.org/wiki/Key:highway) and [Railway section](https://wiki.openstreetmap.org/wiki/Railways)

# When dealing with highways there are two types of object, LineString that represents a line and Polygon that represent a shape.<br>
# Shapes are useful when you are trying to plot squares or buildings.<br>
# ![](https://imgur.com/6Aa23fz.png)<br>
# As you can see, the map is pretty messed up:
# ![](https://imgur.com/VMhKJ3v.png)

# Now we can go back to the code

# In[ ]:


with open("../input/osm-overpass-tutorial-dataset/roads.geojson") as file:
    data = geojson.load(file)

data.keys()


# In the document properties there is also "highway" or "railway" that indicates the type (that can be motorway, trunk, primary etc. for highways)

# In[ ]:


data.features[0].properties.keys()


# In this case there are only LineStrings and not Polygons because we are not selecting *pedestrians* or other type of highway that can be represented by areas, but in general they can be saved in the same manner, importing Polygons (the needed cose is left commented)<br>

# In[ ]:


roads = {}
rails = {}
for path in data.features:
        # street data
    if "highway" in path.properties.keys():
        if path["properties"]["highway"] not in roads.keys():
            roads[path["properties"]["highway"]] = []
        
        '''
        if path["geometry"]["type"]=="Polygon":
            cart = [[x, y] for x, y in path["geometry"]["coordinates"][0]]
            roads[path["properties"]["highway"]].append(Polygon(cart))
                     
        '''    
        if path["geometry"]["type"]=="LineString":
            cart = [[x, y] for x, y in path["geometry"]["coordinates"]]
            roads[path["properties"]["highway"]].append(LineString(cart))
                      
    # railway data
    elif "railway" in path["properties"].keys():
        if path["properties"]["railway"] not in rails.keys():
            rails[path["properties"]["railway"]] = []
         
        '''
        if path["geometry"]["type"]=="Polygon":
            cart = [[x, y] for x, y in path["geometry"]["coordinates"][0]]
            rails[path["properties"]["railway"]].append(Polygon(cart))
        '''
        
        if path["geometry"]["type"]=="LineString":
            cart = [[x, y] for x, y in path["geometry"]["coordinates"]]
            rails[path["properties"]["railway"]].append(LineString(cart))


# LineString are composed by two couples of coordinates indicating starting and ending point.

# In[ ]:


str(roads["primary"][0])


# We can then plot each LineString, extracting the coordinates.<br>
# White lines represents streets while red lines represents railways<br>
# As before, the case of Polygons is left commented.

# In[ ]:


def plot_paths(ax, paths, color, width, linestyle):
    for path in paths:
        '''
        if isinstance(path, Polygon):
            x, y = path.exterior.xy   
        '''
        
        if isinstance(path, LineString):
            x, y = path.xy
        else:
            continue
        
        mercator = webm(list(x), list(y))
        
        
        ax.plot(mercator[0], mercator[1], color=color, linewidth=width, linestyle=linestyle, solid_capstyle='round')


# In[ ]:


roads.keys()


# In[ ]:


rails.keys()


# In[ ]:


fig, ax = plt.subplots(figsize=(10, 10))

bg = '#011654'
rail_color = '#CF142B'
street_color = '#FFFFFF'

plot_paths(ax, roads["motorway"], street_color, 0.8, "-")
plot_paths(ax, roads["motorway_link"], street_color, 0.8, "-")
plot_paths(ax, roads["trunk"], street_color, 0.8, "-")
plot_paths(ax, roads["trunk_link"], street_color, 0.8, "-")

plot_paths(ax, roads["primary"], street_color, 0.6, "-")
plot_paths(ax, roads["primary_link"], street_color, 0.6, "-")

plot_paths(ax, roads["secondary"], street_color, 0.4, "-")
plot_paths(ax, roads["secondary_link"], street_color, 0.4, "-")

plot_paths(ax, rails["rail"], rail_color, 0.6, '-.')

ax.set_facecolor(bg)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.get_xaxis().set_ticks([])
ax.get_yaxis().set_ticks([])

plt.show()


# I made another [kernel](https://www.kaggle.com/lorenzodenisi/street-map-visualization-from-open-street-map-data) where I show how to plot the highways of the city of Turin.<br>
# In that case, considering all type of highways, there are some polygons although its difficult to notice them with a low scale.
