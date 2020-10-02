#!/usr/bin/env python
# coding: utf-8

# # Intro
# I've always like the idea that excelling in one area can be better than being average in a lot, so I wanted to walk through my favourite functionality to Matplotlib's Basemap.
# 
# This Kernel walks through basic design styles, a few colour schemes and methods of representing data to get people started.
# 
# I first learned to use Basemap on DataQuest.io and I highly recommend the site to any aspiring datascientist (no value in the promotion for me, I'm just a fan).
# 
# ### Before you begin
# One notable fact about this data set is that it uses Easting and Northing, i.e. not your normal Longitude and Latitude. How this works is explained below. You can also grab the lat and long from the data set so have no fear if you (for good reason) hate other coordinate systems.
# 
# Basemap is a good tool for taking varied types of coordinates so this factors well into the walkthrough.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import matplotlib.cm

from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize

import matplotlib.pyplot as plt
import matplotlib.cm
 
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
df = pd.read_csv('../input/ukTrafficAADF.csv')


# ### Latitude and longitude basics
# Latitude (aka vertical, or North-South) can range from -90 to 90, which is effectively the south pole (-90) and north pole (90). The equator represents 0.
# 
# They are in the 'lat' and 'lon' columns of the dataset.
# 
# Longitude (aka horizontal, or East-West) can range from -180 to 180. Imagine the world is split down the middle from top to bottom. On the west side you have 0 to 180 (south to north pole). On the east side you have 0 to -180.
# 
# There's a simple visual for that here, http://www.learner.org/jnorth/tm/LongitudeIntro.html
# 
# ### The other systems (including the British one that this data uses)
# There are a lot of systems for map coordinates outside the international lat & long system.
# 
# Why have multiple systems? They allow for more precision. Local systems can be accurate down the cm (potentially smaller). When a local government is doing work that can be valuable. This dataset uses a British system and the method for calling this is displayed in the first setup.
# 
# If you need to call one of these systems in your map then we add a parameter called 'epsg' (explained later). epsg values can be found here, http://spatialreference.org
# 
# ** Two small examples are catchy titles **
# - OSGB (ordanance Survey of Great Britain) = Easting and Northing
# - WGS84 (World Geodetic System) = Latitude and Longitude
# 
# 

# # Using Basemap (mpl_toolkits.basemap module)
# 
# This is the fundamental mapping tool for Python. You could master this and not need another mapping tool.
# 
# The first code block creates a figure, sets the area for the map, adds basics like boundaries and water colour 

# #A basic map
# 
# I've added notes below for more details. Essentially the process here is 
# 1. Create a figure
# 1. Create a basic map to plot things on to our map (see 'm = Basemap' below)
# 1. Add features like mapboundaries, rivers, etc
# 1. I created tuples to store my coords in one place
# 1. Run a 'for' function to grab each set of coords and plot them on the map - If you removed the for loop then you would just have a nice plain map.
# 
# ### Calling the British system
# - First, we call 'epsg=27700' when creating the map.
# - Seond, when plotting points, we sat 'latlon=False' because we aren't providing latitude or longitude we're providing OSGB/espg=27700 coordinates
# - For projection we use 'tmerc'
# 
# ### If you want to use the normal lat and long
# - Use projection = 'merc', 
# - Delete 'latlon=False' (it defaults to True)
# - Delete 'epsg=27700'.
# - Change 'x,y=i' to 'x,y = m(lat, lon)' (for reasons I don't fully grasp the basemap does a further translation on the lat and long with this.
# 
# 

# **N.B. This example just uses the year 2000 so it was faster to run and clean to view.**

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='i', # Set using letters, e.g. c is a crude drawing, f is a full detailed drawing
            projection='tmerc', # The projection style is what gives us a 2D view of the world for this
            lon_0=-4.36,lat_0=54.7, # Setting the central point of the image
            epsg=27700) # Setting the coordinate system we're using

m.drawmapboundary(fill_color='#46bcec') # Make your map into any style you like
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like
m.drawcoastlines()
m.drawrivers() # Default colour is black but it can be customised
m.drawcountries()

df['lat_lon'] = list(zip(df.Easting, df.Northing)) # Creating tuples
df_2000 = df[df['AADFYear']==2000]

for i in df_2000[0:500]['lat_lon']:
    x,y = i
    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)

plt.show()


# # Choose your background
# It's more exciting to start with a good looking map. You can stick with the simple colour tones we've got above, set your own colours or choose from some pre-made styles.
# 
# Personally I prefer using colour tones. I'll be playing with them throughout. They're mostly set with drawmapboundary(), fillcontintinents(), and drarrivers()

# ## Using NASA's Blue Marble
# 
# It's a simple and basic tool in the toolkit but it looks good. Turn off other drawings. This essentially places a background image instead of a map and plots on top of it. I used a subset of the data for faster loading. 

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='f',
            projection='cass',
            lon_0=-4.36,lat_0=54.7,
            epsg=27700)

m.bluemarble() # The key change in this cell

m.drawcoastlines() # You can run without this and it removes the black line

df['lat_lon'] = list(zip(df.Easting, df.Northing))
df_2000 = df[df['AADFYear']==2000]

for i in df_2000[0:1000]['lat_lon']:
    x,y = i
    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)

plt.show()


# ### Setting the background to m.shadedrelief

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='f',
            projection='cass',
            lon_0=-4.36,lat_0=54.7,
            epsg=27700)

m.shadedrelief()

m.drawcoastlines() # You can run without this and it removes the black line

df['lat_lon'] = list(zip(df.Easting, df.Northing))
df_2000 = df[df['AADFYear']==2000]

for i in df_2000[0:1000]['lat_lon']:
    x,y = i
    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)

plt.show()


# ### Setting the background to m.etopo

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))

m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='f',
            projection='cass',
            lon_0=-4.36,lat_0=54.7,
            epsg=27700)

m.etopo()

m.drawcoastlines() # You can run without this and it removes the black line

df['lat_lon'] = list(zip(df.Easting, df.Northing))
df_2000 = df[df['AADFYear']==2000]

for i in df_2000[0:1000]['lat_lon']:
    x,y = i
    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)

plt.show()


# # Start representing the data
# I'll run through some approaches and keep it simple. The idea is to set you up with the principles so that you can experiment.

# ### Colour coding to see regions
# You can use more advanced functions called shapefiles to upload boundary areas and map them on. For our purposes though we can use the files pre-defined areas. Users in the US can also employ 'map.drawcounties()'.
# 
# To see a shapefile in action you can use this link, http://www.datadependence.com/2016/06/creating-map-visualisations-in-python/

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='i',
            projection='tmerc',
            lon_0=-4.36,lat_0=54.7,
            epsg=27700)

m.drawmapboundary(fill_color='#e2e2d7')
m.fillcontinents(color='#007fff',lake_color='#ffffff') #zorder=0
#m.drawcoastlines()
m.drawrivers(color = '#ffffff', linewidth=2)
m.drawcountries(linewidth=2)

df['lat_lon'] = list(zip(df.Easting, df.Northing))
df_2000 = df[df['AADFYear']==2000]

region_set = set(df['Region']) # Creates a list of all the unique regions
colour_set = ['#f9ebea','#d5d8dc','#c39bd3','#BA4A00','#17A589','#1E8449','#e2df5d','#2E4053','#F1c40F','#A9DFBF','#F0B27A'] # A list of random colour codes
region_colour_dict = dict(zip(region_set, colour_set)) # Creates a dictionary so each region has a colour code

df_2000 = df_2000.reset_index()

for i, r in df_2000.iterrows(): # Runs over rows returning each one as a series, so we can still use the values to se colour
    x,y = r['lat_lon']
    m.plot(x, y, marker = 'o', c=region_colour_dict[r['Region']], markersize=1, alpha=0.8, latlon=False)
# The main change in this block is using the region dictionary to change the colour code for each marker

plt.show()


# In[ ]:


print(region_set)


# ### Where are pedal bikes most popular? (By size)
# 
# We can represent popularity by changing the marker sizes. In this case I added 4 lines.
# - 3 lines get the min and max vales and subtract them from one another. 
# - The fourth line is in the for loop and creates a size value
# - We then editer the marker size value.
# 
# ** What's happening **
# We use the min and max values to determine the normalised size of the biking population (ranging from 0 to 1, with 1 being the highest possible value).Then  I multiply the marker size by the normalisex value and then we have markers relatively sized to reflect the popularity of biking.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='i',
            projection='tmerc',
            lon_0=-4.36,lat_0=54.7,
            epsg=27700)

m.drawmapboundary(fill_color='#232b2b') # Make your map into any style you like #46bcec
m.fillcontinents(color='#A9A9A9',lake_color='#46bcec') # Make your map into any style you like
m.drawcoastlines()
m.drawrivers(linewidth=2, color='#46BCEC') # Default colour is black but it can be customised
m.drawcountries(linewidth=2, color='#ffffff')

df['lat_lon'] = list(zip(df.Easting, df.Northing))
df_2000 = df[df['AADFYear']==2000]

region_set = set(df['Region']) # Creates a list of all the unique regions
colour_set = ['#f9ebea','#d5d8dc','#c39bd3','#BA4A00','#17A589','#1E8449','#e2df5d','#2E4053','#F1c40F','#A9DFBF','#F0B27A'] # A list of random colour codes
region_colour_dict = dict(zip(region_set, colour_set)) # Creates a dictionary so each region has a colour code

df_2000 = df_2000.reset_index()

min_PedalCycles = min(df_2000['PedalCycles'])
max_PedalCycles = max(df_2000['PedalCycles'])
denom = max_PedalCycles - min_PedalCycles

for i, r in df_2000.iterrows(): # Runs over rows returning each one as a series, so we can still use the values to se colour
    x,y = r['lat_lon']
    size = (r['PedalCycles']-min_PedalCycles)/denom
    m.plot(x, y, marker = 'o', c=region_colour_dict[r['Region']], markersize=40*size, alpha=0.8, latlon=False)
# The main change in this block is using the region dictionary to change the colour code for each marker

plt.show()


# ### Where are motor vehickes most popular? (By colour hue)
# 
# We switch out the colour list with different hues and then alter the for loop to select based on the position in the normalised values.
# 
# I would have used pedal or motor cycles but the skew of the data was very extreme (see box plot below). Instead I used AllMotorVehicles but still had to account for a large skew.

# In[ ]:


fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='i',
            projection='tmerc',
            lon_0=-4.36,lat_0=54.7,
            epsg=27700)

m.drawmapboundary(fill_color='#46bcec')
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') #zorder=0
m.drawcoastlines()
m.drawrivers(linewidth=2, color='#46bcec')
m.drawcountries()

#m.drawmapboundary(fill_color='#232b2b') # Make your map into any style you like #46bcec
#m.fillcontinents(color='#A9A9A9',lake_color='#46bcec') # Make your map into any style you like
#m.drawcoastlines()
#m.drawrivers(linewidth=2, color='#46BCEC') # Default colour is black but it can be customised
#m.drawcountries(linewidth=2, color='#ffffff')


df['lat_lon'] = list(zip(df.Easting, df.Northing))
df_2000 = df[df['AADFYear']==2000]

region_set = set(df['Region']) # Creates a list of all the unique regions
colour_set = ['#382E2E','#471F1F','#521414','#8A0F0F','#990000'] # A list of random colour codes
region_colour_dict = dict(zip(region_set, colour_set)) # Creates a dictionary so each region has a colour code

df_2000 = df_2000.reset_index()

min_AllMotorVehicles = min(df_2000['AllMotorVehicles'])
max_AllMotorVehicles = max(df_2000['AllMotorVehicles'])
denom = max_AllMotorVehicles - min_AllMotorVehicles

for i, r in df_2000.iterrows(): # Runs over rows returning each one as a series, so we can still use the values to se colour
    x,y = r['lat_lon']
    size = (r['AllMotorVehicles']-min_AllMotorVehicles)/denom
    if size < 0.1:
        colour_depth = 0
    elif size <0.06:
        colour_depth = 1
    elif size <0.2:
        colour_depth = 2
    elif size <0.3:
        colour_depth = 3
    elif size <1:
        colour_depth = 4
    m.plot(x, y, marker = 'o', c=colour_set[colour_depth], markersize=0.7, alpha=0.8, latlon=False)
# The main change in this block is using the region dictionary to change the colour code for each marker

plt.show()


# The representation by hue seemed strange so a quick box plot reveals why. The data is very skewed. Then I recalled that this is because we are looking at main road data so probaby shouldn't expect many bikes.

# In[ ]:


df_2000['AllMotorVehicles'].plot(kind="box")


# In[ ]:





# # Nightshade
# Shade the regions of the map that are in darkness at the time specifed by date. date is a datetime instance, assumed to be UTC.
# 
# Notice that this tutorial calls the time as 'now' so you may see different shading depending on when it runs.

# In[ ]:


import datetime

fig, ax = plt.subplots(figsize=(10,10))
m = Basemap(llcrnrlon=-7.5600,llcrnrlat=49.7600,
            urcrnrlon=2.7800,urcrnrlat=60.840,
            resolution='i', # Set using letters, e.g. c is a crude drawing, f is a full detailed drawing
            projection='tmerc', # The projection style is what gives us a 2D view of the world for this
            lon_0=-4.36,lat_0=54.7, # Setting the central point of the image
            epsg=27700) # Setting the coordinate system we're using

m.drawmapboundary(fill_color='#46bcec') # Make your map into any style you like
m.fillcontinents(color='#f2f2f2',lake_color='#46bcec') # Make your map into any style you like
m.drawcoastlines()
m.drawrivers() # Default colour is black but it can be customised
m.drawcountries()

m.nightshade(datetime.datetime.now())


df['lat_lon'] = list(zip(df.Easting, df.Northing)) # Creating tuples
df_2000 = df[df['AADFYear']==2000]

for i in df_2000[0:500]['lat_lon']:
    x,y = i
    m.plot(x, y, marker = 'o', c='r', markersize=1, alpha=0.8, latlon=False)

plt.show()


# # Still to use
# * wmsimage
# * warpimage
# 
# * imshow(*args, **kwargs)
# 

# In[ ]:




