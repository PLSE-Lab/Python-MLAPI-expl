#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

import folium
from folium.map import Icon


# The aim of this notebook will be to provide you a good overview of how you can use folium to draw nice maps.
# You can also find the official documentation [there](https://python-visualization.github.io/folium/).
# 
# Creating a folium map is a two-steps process:
# First, you have to setup your map, the *backgound*. Then, you can add all the elements you want, mainly your data.
# 
# This tuto will be divided as follows:
# * [Maps](https://www.kaggle.com/anthaus/tuto-draw-nice-maps-with-folium#Maps)
#     * [Zoom](https://www.kaggle.com/anthaus/tuto-draw-nice-maps-with-folium#Zoom)
#     * [Tiles](https://www.kaggle.com/anthaus/tuto-draw-nice-maps-with-folium#Tiles)
# * [Points of interest](https://www.kaggle.com/anthaus/tuto-draw-nice-maps-with-folium#Points%20of%20interest)
#     * [Markers](https://www.kaggle.com/anthaus/tuto-draw-nice-maps-with-folium#Markers)
#     * [Circles](https://www.kaggle.com/anthaus/tuto-draw-nice-maps-with-folium#Circles)
#     * [PolyLines](https://www.kaggle.com/anthaus/tuto-draw-nice-maps-with-folium#PolyLines)
#     
# Folium also provides ways to draw choropleth maps, but it is quite different, and harder to master, so I decided to let it apart for a further tutorial.

# ## Maps
# 
# The map object will contain your draw. It is basically a map, with some specifities, where points of interest will be added.
# 
# Useful parameters for your maps:
# - **location**: latitude and longitude of the center of the map
# - **zoom_start**: level of zoom of your map (details below)
# - **tiles**: graphic style of the map (details below)
# - **width, height** width and height of your map
# - **zoom_min, zoom_max**: if you want to put restriction on the zoom possibilities
# 
# ### Zoom ###
# 
# It is not always obvious where to center your map and how to focus your zoom. It is usually done by testing and fixing until the result is fine. The following table gives an idea of the area covered depending on the level of zoom.
# 
# | Zoom level | Covered area | Example |
# | ---------- | ------------ | ------- |
# |3|World| |
# |4|Continent| |
# |5|Big country|China, USA...|
# |7|Medium country|Spain, Poland...|
# |10|Small country, megapolis|Luxembourg, Tokyo...|
# |12|Extended city|Los Angeles...|
# |13|Big city|Paris, Bruxelles...|
# |15|Medium city| |
# |18|Neighbourhood|Maximum value|
# 
# 
# If you use float values for your round levels, they will be rounded to the closest integer. It is also possible to zoom and unzoom the map once it is generated.
# 
# ### Tiles ###
# 
# Tiles are the graphic style of your map. It is a way to embellish your maps, but also to give more information. First, let's take a look at the freely avaiblable tiles.
# 
# | Tile's name | Description |
# | ----------- | ----------- |
# |OpenStreetMap|Same maps as the one in www.openstreetmap.org and default value.|
# |Stamen Terrain|Color varies depending on height and type of terrain.|
# |Stamen Toner|Black & White. Water and frontiers are in black, terrains in white.|
# |Stamen Watercolor|Terrains in red tones, water in blue tones. Looks like if it was drawn on carton.|
# |CartoDB positron|Light map with light colors.|
# |CartoDB dark_matter|Light map with dark colors.|
# 
# The tiles' choice depends on your sensibility, but also in what you want to show. If the kind of terrain is important, then a *Stamen Terrain* will be an interesting choice. If you have a lot of elements to add, with many colors, a lighter map, with *Stamen Toner* or *CartoDB positron*  will probably be easier to read. We will use various tiles as examples in this tutorial.
# You can find more tiles using the good API.

# ## Points of Interest
# 
# There are different way to build points of interest (*PoI*). We will present some of them below. The process is always the same: we build our *PoI* one by one (usually by looping over the data) and add them to the map with the *add_to(map)* function.
# 
# ### Markers
# 
# Markers are probably the most basic way to indicate some point of interest on a folium map.
# We will illustrate it with a dataset of 3-stars Michelin restaurants of 2019.
# 
# 
# Useful parameters for your Markers:
# - **location** : latitude and longitude of the elements. It is an essential parameter.
# - **popup** : label for your Marker, that you can read by clicking on the Marker on the map.
# - **icon** : *pictogram* of the Marker (more details below)

# In[ ]:


# Loading the data
df_resto = pd.read_csv("../input/michelin-restaurants/three-stars-michelin-restaurants.csv")

# Loading the background map
resto_map = folium.Map(location=[37.6, 4.96], zoom_start=3)

# Adding a marker for each restaurant
for idx, restaurant in df_resto.iterrows():
    folium.Marker(location=(restaurant['latitude'],restaurant['longitude']), popup=restaurant['name']).add_to(resto_map)

# Display
resto_map


# As you can see, we obtained a simple map of the world, with markers to indicate the locations of our 3-stars restaurants.
# But the dataset gives more information than just location, e.g. the type of cuisine served in these restaurants. So it could be interesting to show such an information in the map as well. A simple way to do that is to color the markers depending on the cuisine of their restaurants.
# 
# So, we will draw a new map, giving the markers different colours depending on the cuisine of the restaurant:
# 
# - French in blue 
# - Italian in green 
# - Asian in beige 
# - American in red 
# - Others in black 
# 
# To this end, we will need to specify our Icon items. An Icon describes how the Marker looks like: i.e. its colors and if there is a pictogram inside. When not specified, as in the previous map, a standard Icon is used, i.e. a blue one, with a white circle inside.
# 
# 
# Useful parameters for you Icons:
# - **color** : the background color of your Marker
# - **icon_color** : the color of the pictogram inside the Marker
# - **icon** : the pictogram ( [list of available pictograms](https://fontawesome.com/v4.7.0/icons/) )
# 
# The use of icons suits very well to show qualitative distinctions.

# In[ ]:


# Loading the background map
resto_colored_map = folium.Map(location=[37.6, 4.96], zoom_start=3, tiles="CartoDB positron")

# Adding a marker for each restaurant
for idx, restaurant in df_resto.iterrows():
    if restaurant['cuisine'] in ('Classic French, French, French contemporary'):
        folium.Marker(location=(restaurant['latitude'],restaurant['longitude']), icon=Icon(color='blue', icon_color='darkblue', icon='cutlery')).add_to(resto_colored_map)
    elif restaurant['cuisine'] == 'Italian':
        folium.Marker(location=(restaurant['latitude'],restaurant['longitude']), icon=Icon(color='green', icon='cutlery')).add_to(resto_colored_map)
    elif restaurant['cuisine'] in ('Asian', 'Cantonese', 'Chinese', 'Japanese', 'Korean', 'Sushi'):
        folium.Marker(location=(restaurant['latitude'],restaurant['longitude']), icon=Icon(color='beige', icon='cutlery')).add_to(resto_colored_map)
    elif restaurant['cuisine'] == 'American':
        folium.Marker(location=(restaurant['latitude'],restaurant['longitude']), icon=Icon(color='red', icon_color='darkred', icon='cutlery')).add_to(resto_colored_map)
    else:
        folium.Marker(location=(restaurant['latitude'],restaurant['longitude']), icon=Icon(color='black', icon='cutlery')).add_to(resto_colored_map)

# Display
resto_colored_map


# ### Circles
# 
# Another interesting item is the circle. It does not differ that much from Markers. It exists in two forms: Circle and CircleMarker. The main difference is that the radius is expressed in meters for Circle, whereas it is expressed in pixels for CircleMarkers. Thus, Circles can be zoomed or unzoomed, whereas CircleMarkers keep the same size on screen, no matter the zoom. The main advantage of circles is that we can play with their radius. This is much appreciated to express quantitative values.
# 
# 
# Useful parameters for you Circles and CircleMarkers:
# 
# - **location** : never forget it ;-)
# - **popup** : same as for Markers
# - **radius** : radius of the circle
# - **color** : color of the circle
# - **fill** : True if you also want color inside the circle
# 
# As an illustration, we will use a dataset about Italian earthquakes.

# In[ ]:


# Loading the data
df = pd.read_csv("../input/earthquakes-in-japan/Japan earthquakes 2001 - 2018.csv")
df_eq = df[df['mag'] > 6.3]

# Loading the background map
eq_map = folium.Map(location=[37.973265, 142.597713], zoom_start=5, tiles='Stamen Terrain', width=1250)

# Adding a circle for each earthquake
for idx, eq in df_eq.iterrows():
    folium.Circle(location=(eq['latitude'], eq['longitude']), radius=20000 * (eq['mag']-6), color='brown', fill=True).add_to(eq_map)
    
eq_map


# ### PolyLines
# 
# With folium, it is also possible to draw lines between points. This can be used to draw something, but especially for locations tracking. 
# 
# Useful parameters:
# - **locations**: please note the plurals. Here, locations should be provided as a sorted list of tuples(latitude, longitude).
# - **popup**: label displayed while clicking.
# - **no_clip**: boolean (default is False) to enable or disable polyline clipping
# - **smooth_factor**: How much to simplify the polyline on each zoom level. More means better performance and smoother look, and less means more accurate representation.
# 
# 
# As an example, we will use a dataset on birds migration. The map will show the trajectories of two birds' round trips between Europe and Africa.
# 

# In[ ]:


# Loading the data
df_animal = pd.read_csv("../input/movebank-animal-tracking/migration_original.csv").sort_values('timestamp')

# Loading the background map
animal_map = folium.Map(location=[31.958247, 31.091527], zoom_start=3, width=800, tiles='Stamen Terrain')

# Building the coordinates lists for two birds
loc_1 = []
loc_2 = []
for idx, point in df_animal.iterrows():
    if point['individual-local-identifier'] == '91732A':
        loc_1.append((point['location-lat'], point['location-long']))
    elif point['individual-local-identifier'] == '91752A':
        loc_2.append((point['location-lat'], point['location-long']))

# Adding trajectories to the map
folium.PolyLine(locations=loc_1, color='darkred', no_clip=True).add_to(animal_map)
folium.PolyLine(locations=loc_2, color='darkorange', no_clip=True).add_to(animal_map)

animal_map


# ## If you are still here 
# 
# Thank you for reading. I hope you enjoined this tutorial, and learned from it. If something is unclear, unaccurate, if there are some mispellings, or if you want me to go deeper on some points, please tell me in the comment section.
