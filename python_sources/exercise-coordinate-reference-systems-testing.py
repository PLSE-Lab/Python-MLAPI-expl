#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install -U -t /kaggle/working/ git+https://github.com/Kaggle/learntools.git@geospatial_edits')


# In[ ]:


import sys
sys.path.append('/kaggle/working')


# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial)**
# 
# ---
# 

# # Introduction
# 
# You are a bird conservation expert and want to understand migration patterns of purple martins.  In your research, you discover that these birds typically spend the summer breeding season in the eastern United States, and then migrate to South America for the winter.  But since this bird is under threat of endangerment, you'd like to take a closer look at the locations that these birds are more likely to visit.
# 
# <center>
# <img src="https://i.imgur.com/qQcS0KM.png" width="1000"><br/>
# </center>
# 
# There are several [protected areas](https://www.iucn.org/theme/protected-areas/about) in South America, which operate under special regulations to ensure that species that migrate (or live) there have the best opportunity to thrive.  You'd like to know if purple martins tend to visit these areas.  To answer this question, you'll use some recently collected data that tracks the year-round location of eleven different birds.
# 
# Before you get started, run the code cell below to set everything up.

# In[ ]:


import pandas as pd
import geopandas as gpd

from shapely.geometry import LineString

from learntools.core import binder
binder.bind(globals())
from learntools.geospatial.ex2 import *


# # Exercises
# 
# ### 1) Load the data.
# 
# Run the next code cell (without changes) to load the GPS data into a pandas DataFrame `birds_df`.  

# In[ ]:


# Load the data and print the first 5 rows
birds_df = pd.read_csv("../input/geospatial-learn-course-data/purple_martin.csv", parse_dates=['timestamp'])
print("There are {} different birds in the dataset.".format(birds_df["tag-local-identifier"].nunique()))
birds_df.head()


# There are 11 birds in the dataset, where each bird is identified by a unique value in the "tag-local-identifier" column.  Each bird has several measurements, collected at different times of the year.
# 
# Use the next code cell to create a GeoDataFrame `birds`.  
# - `birds` should have all of the columns from `birds_df`, along with a "geometry" column that contains Point objects with (longitude, latitude) locations.  
# - Set the CRS of `birds` to `{'init': 'epsg:4326'}`.

# In[ ]:


# Your code here: Create the GeoDataFrame
birds = ____

# Your code here: Set the CRS to {'init': 'epsg:4326'}
birds.crs = ____

# Check your answer
q_1.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()


# ### 2) Plot the data.
# 
# Next, we load in the `'naturalearth_lowres'` dataset from GeoPandas, and set `americas` to a GeoDataFrame containing the boundaries of all countries in the Americas (both North and South America).  Run the next code cell without changes.

# In[ ]:


# Load a GeoDataFrame with country boundaries in North/South America, print the first 5 rows
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
americas = world.loc[world['continent'].isin(['North America', 'South America'])]
americas.head()


# Use the next code cell to create a single plot that shows both: (1) the country boundaries in the `americas` GeoDataFrame, and (2) all of the points in the `birds_gdf` GeoDataFrame.  
# 
# Don't worry about any special styling here; just create a preliminary plot, as a quick sanity check that all of the data was loaded properly.  In particular, you don't have to worry about color-coding the points to differentiate between birds, and you don't have to differentiate starting points from ending points.  We'll do that in the next part of the exercise.

# In[ ]:


# Your code here
____

# Uncomment to see a hint
#q_2.hint()


# In[ ]:


# Get credit for your work after you have created a map
q_2.check()

# Uncomment to see our solution (your code may look different!)
##q_2.solution()


# ### 3) Where does each bird start and end its journey? (Part 1)
# 
# Now, we're ready to look more closely at each bird's path.  Run the next code cell to create two GeoDataFrames:
# - `path_gdf` contains LineString objects that show the path of each bird.  It uses the `LineString()` method to create a LineString object from a list of Point objects.
# - `start_gdf` contains the starting points for each bird.

# In[ ]:


# GeoDataFrame showing path for each bird
path_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: LineString(x)).reset_index()
path_gdf = gpd.GeoDataFrame(path_df, geometry=path_df.geometry)
path_gdf.crs = {'init' :'epsg:4326'}

# GeoDataFrame showing starting point for each bird
start_df = birds.groupby("tag-local-identifier")['geometry'].apply(list).apply(lambda x: x[0]).reset_index()
start_gdf = gpd.GeoDataFrame(start_df, geometry=start_df.geometry)
start_gdf.crs = {'init' :'epsg:4326'}

# Show first five rows of GeoDataFrame
start_gdf.head()


# Use the next code cell to create a GeoDataFrame `end_gdf` containing the final location of each bird.  
# - The format should be identical to that of `start_gdf`, with two columns ("tag-local-identifier" and "geometry"), where the "geometry" column contains Point objects.
# - Set the CRS of `end_gdf` to `{'init': 'epsg:4326'}`.

# In[ ]:


# Your code here
end_gdf = ____

# Check your answer
q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()


# ### 4) Where does each bird start and end its journey? (Part 2)
# 
# Use the GeoDataFrames from the question above (`path_gdf`, `start_gdf`, and `end_gdf`) to visualize the paths of all birds on a single map.  You may also want to use the `americas` GeoDataFrame.

# In[ ]:


# Your code here
____

# Uncomment to see a hint
#q_4.hint()


# In[ ]:


# Get credit for your work after you have created a map
q_4.check()

# Uncomment to see our solution (your code may look different!)
#q_4.solution()


# ### 5) Where are the protected areas in South America? (Part 1)
# 
# It looks like all of the birds end up somewhere in South America.  But are they going to protected areas?
# 
# In the next code cell, you'll create a GeoDataFrame `protected_areas` containing the locations of all of the protected areas in South America.  The corresponding shapefile is located at filepath `protected_filepath`.

# In[ ]:


# Path of the shapefile to load
protected_filepath = "../input/geospatial-learn-course-data/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile/SAPA_Aug2019-shapefile-polygons.shp"

# Your code here
protected_areas = ____

# Check your answer
q_5.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_5.hint()
#q_5.solution()


# ### 6) Where are the protected areas in South America? (Part 2)
# 
# Create a plot that uses the `protected_areas` GeoDataFrame to show the locations of the protected areas in South America.

# In[ ]:


# Country boundaries in South America
south_america = americas.loc[americas['continent']=='South America']

# Your code here: plot protected areas in South America
____

# Uncomment to see a hint
#q_6.hint()


# In[ ]:


# Get credit for your work after you have created a map
q_6.check()

# Uncomment to see our solution (your code may look different!)
#q_6.solution()


# ### 7) What percentage of South America is protected?
# 
# You're interested in determining what percentage of South America is protected, so that you know how much of South America is suitable for the birds.  
# 
# As a first step, you calculate the total area of all protected lands in South America (not including marine area).  To do this, you use the "REP_AREA" and "REP_M_AREA" columns, which contain the total area and total marine area, respectively, in square kilometers.
# 
# Run the code cell below without changes.

# In[ ]:


P_Area = sum(protected_areas['REP_AREA']-protected_areas['REP_M_AREA'])
print("South America has {} square kilometers of protected areas.".format(P_Area))


# Then, to finish the calculation, you'll use the `south_america` GeoDataFrame.  

# In[ ]:


south_america.head()


# Calculate the total area of South America by following these steps:
# - Calculate the area of each country using the `area` attribute of each polygon (with EPSG 3035 as the CRS), and add up the results.  The calculated area will be in units of square meters.
# - Convert your answer to have units of square kilometeters.

# In[ ]:


# Your code here: Calculate the total area of South America (in square kilometers)
totalArea = ____

# Check your answer
q_7.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_7.hint()
#q_7.solution()


# Run the code cell below to calculate the percentage of South America that is protected.

# In[ ]:


# What percentage of South America is protected?
percentage_protected = P_Area/totalArea
print('Approximately {}% of South America is protected.'.format(round(percentage_protected*100, 2)))


# ### 8) Where are the birds in South America?
# 
# So, are the birds in protected areas?  
# 
# Create a plot that shows for all birds, all of the locations where they were discovered in South America.  Also plot the locations of all protected areas in South America.
# 
# To exclude protected areas that are purely marine areas (with no land component), you can use the "MARINE" column (and plot only the rows in `protected_areas[protected_areas['MARINE']!='2']`, instead of every row in the `protected_areas` GeoDataFrame).

# In[ ]:


# Your code here
____

# Uncomment to see a hint
#q_8.hint()


# In[ ]:


# Get credit for your work after you have created a map
q_8.check()

# Uncomment to see our solution (your code may look different!)
#q_8.solution()


# # Keep going
# 
# Create stunning **[interactive maps](https://www.kaggle.com/alexisbcook/interactive-maps-testing)** with your geospatial data.

# ---
# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
