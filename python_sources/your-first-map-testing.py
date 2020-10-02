#!/usr/bin/env python
# coding: utf-8

# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial)**
# 
# ---
# 

# # Introduction
# 
# In this micro-course, you'll learn about different methods to wrangle and visualize **geospatial data**, or data with a geographic location.
# 
# Along the way, you'll solve hands-on exercises that address several questions:
# - Where should a global non-profit expand its reach in remote areas of the Philippines?
# - How do purple martins, a threatened bird species, travel between North and South America?  Are the birds travelling to conservation areas?
# - Which areas of Japan could potentially benefit from extra earthquake reinforcement?
# - Which Starbucks stores in California are strong candidates for the next [Starbucks Reserve Roastery](https://www.forbes.com/sites/garystern/2019/01/22/starbucks-reserve-roastery-its-spacious-and-trendy-but-why-is-starbucks-slowing-down-expansion/#6cb80d4a1bc6) location?
# - Does New York City have sufficient hospitals to respond to motor vehicle collisions?  Which areas of the city have gaps in coverage?
# 
# You'll also visualize crime in the city of Chicago, examine health facilities in Ghana, explore top universities in Europe, and track releases of toxic chemicals in the United States.
# 
# This micro-course assumes that you have previous experience with the pandas library.  In this first tutorial, we'll quickly cover the pre-requisites that you'll need to complete this micro-course.  And, if you'd like to review more deeply, we strongly recommend the **[Pandas micro-course](https://www.kaggle.com/learn/pandas)**.  
# 
# We'll also get started with visualizing our first geospatial dataset!
# 
# # Reading and plotting data
# 
# The first step is to read in some geospatial data!  To do this, we'll use the [GeoPandas](http://geopandas.org/) library.

# In[ ]:


import geopandas as gpd


# GeoPandas is an extension of the [Pandas](https://pandas.pydata.org/) library with added functionality for geospatial data.
# 
# With GeoPandas, we can read data from a variety of common geospatial file formats, such as [shapefile](https://en.wikipedia.org/wiki/Shapefile), [GeoJSON](https://en.wikipedia.org/wiki/GeoJSON), [KML](https://en.wikipedia.org/wiki/Keyhole_Markup_Language), and [GPKG](https://en.wikipedia.org/wiki/GeoPackage).  You can also work with [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) files.  
# 
# For now, we'll load a shapefile containing information about lands under the care of the [Department of Environmental Conservation](https://www.dec.ny.gov/index.html) in the state of New York.

# In[ ]:


# Read in the data
full_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_lands/DEC_lands/DEC_lands.shp")
full_data.head()


# The command above reads the data into a (GeoPandas) **GeoDataFrame** object that has all of the capabilities of a (Pandas) DataFrame object: so, everything you learned in the [Pandas micro-course](https://www.kaggle.com/learn/pandas) can be used to work with the data. 

# In[ ]:


type(full_data)


# For instance, if we don't plan to use all of the columns, we can select a subset of them.  
# > To review other methods for selecting data, check out [this tutorial](https://www.kaggle.com/residentmario/indexing-selecting-assigning/) from the Pandas micro-course.

# In[ ]:


data = full_data.loc[:, ["CLASS", "COUNTY", "geometry"]].copy()


# We use the `value_counts()` method to see a list of different land types, along with how many times they appear in the dataset. 
# > To review this (and related methods), check out [this tutorial](https://www.kaggle.com/residentmario/summary-functions-and-maps) from the Pandas micro-course.

# In[ ]:


# How many lands of each type are there?
data.CLASS.value_counts()


# You can also use `loc` (and `iloc`) and `isin` to select subsets of the data.  
# > To review this, check out [this tutorial](https://www.kaggle.com/residentmario/indexing-selecting-assigning/) from the Pandas micro-course.

# In[ ]:


# Select lands that fall under the "WILD FOREST" or "WILDERNESS" category
wild_lands = data.loc[data.CLASS.isin(['WILD FOREST', 'WILDERNESS'])].copy()
wild_lands.head()


# GeoDataFrames also have some added methods and attributes (that don't apply to DataFrames).  For instance, we can quickly visualize the data with the `plot()` method.  

# In[ ]:


wild_lands.plot()


# # The "geometry" column
# 
# Every GeoDataFrame contains a special "geometry" column.  It contains all of the geometric objects that are displayed when we call the `plot()` method.

# In[ ]:


wild_lands.geometry.head()


# While this column can contain a variety of different datatypes, each entry will typically be a **Point**, **LineString**, or **Polygon**.
# 
# ![](https://i.imgur.com/N1llefr.png)
# 
# The "geometry" column in the dataset that we've just loaded contains 2983 different Polygon objects, each corresponding to a different shape in the plot above.
# 
# In the code cell below, we create three more GeoDataFrames, containing campsite locations (Point), foot trails (LineString), and county boundaries (Polygon).

# In[ ]:


# Campsites in New York state (Point)
POI_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_pointsinterest/DEC_pointsinterest/Decptsofinterest.shp")
campsites = POI_data.loc[POI_data.ASSET=='PRIMITIVE CAMPSITE'].copy()

# Foot trails in New York state (LineString)
roads_trails = gpd.read_file("../input/geospatial-learn-course-data/DEC_roadstrails/DEC_roadstrails/Decroadstrails.shp")
trails = roads_trails.loc[roads_trails.ASSET=='FOOT TRAIL'].copy()

# County boundaries in New York state (Polygon)
counties = gpd.read_file("../input/geospatial-learn-course-data/NY_county_boundaries/NY_county_boundaries/NY_county_boundaries.shp")


# We can quickly plot the information from all four GeoDataFrames.  Note that the `plot()` method takes as (optional) input several parameters that can be used to customize the appearance of your plots.

# In[ ]:


ax = counties.plot(figsize=(10,10), color='none', edgecolor='gainsboro', zorder=3)
wild_lands.plot(color='lightgreen', ax=ax)
campsites.plot(color='maroon', markersize=2, ax=ax)
trails.plot(color='black', markersize=1, ax=ax)


# It looks like the northeastern part of the state would be a great option for a camping trip!
# 
# # Your turn
# 
# **[Identify remote areas](https://www.kaggle.com/kernels/fork/5822773)** of the Philippines where a non-profit can expand its operations.

# ---
# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
