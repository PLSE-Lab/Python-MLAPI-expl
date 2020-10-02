#!/usr/bin/env python
# coding: utf-8

# This notebook is following the course : https://www.kaggle.com/alexisbcook/your-first-map
# 
# 
# The notes and comments are personals. 

# # Goals and steps
# 
# In this notebook as a first step towards manipulating maps with data science, we want to know where are we going to pass our summer holiday in NY in the wildness. 
# 
# For this purpose, we import some availables data about where the wild places are located in NY as well as the campsites. 
# 
# In the following parts, we will import a dataset where we have the location of the wild places in NY then plotted them. 
# 
# Next step will be knowing the campsities are located for that we import the data suitable. 
# 
# In the End, we will plot the campsities and wild lands in one map to see where we will go to enjoy our summer :) 

# ### Importing the usual packages: + Geopandas that allows us manipulating the maps.

# In[ ]:




import geopandas as gpd


# ### Importing data set where the wild places are located in NY. 

# In[ ]:


full_data = gpd.read_file('../input/my-first-map-dec-lands/DEC_lands/DEClands.shp' )
full_data.head()


# We will take a look into the data type of our data. 

# In[ ]:


type(full_data)


# ### As in this project we are intersted in type of the wild lands and which county is located as well as the geomtry of this wild lands

# In[ ]:


data= full_data.loc[:, ["CLASS", "COUNTY","geometry"]].copy()
data.head()


# ### Checking the type of lands that we have in our data set 

# In[ ]:


data.CLASS.value_counts()


# ### Here, we get only the classes of open landscape as well as: wild forest, wilderness

# In[ ]:


wild_lands = data.loc[data.CLASS.isin(['WILD FOREST', 'WILDERNESS'])].copy()
wild_lands.head()


# ### Create my  first map: Plotting

# We move to plot the wild lands where is located using our data wild_lands which contains the county as well as the geomtry

# In[ ]:


wild_lands.plot()


# We are more intersted in the wild places where a campsite is there. in order to get the campsites we import a data containing this information. 

# In[ ]:


# Campsites in New York state (Point)
POI_data = gpd.read_file("../input/geospatial-learn-course-data/DEC_pointsinterest/DEC_pointsinterest/Decptsofinterest.shp")
campsites = POI_data.loc[POI_data.ASSET=='PRIMITIVE CAMPSITE'].copy()
campsites.head()


# Here, we import two data set, where people go as a data foot trail, and the other one the shape of the counties of NY.

# In[ ]:


# Foot trails in New York state (LineString)
roads_trails = gpd.read_file("../input/geospatial-learn-course-data/DEC_roadstrails/DEC_roadstrails/Decroadstrails.shp")
trails = roads_trails.loc[roads_trails.ASSET=='FOOT TRAIL'].copy()

# County boundaries in New York state (Polygon)
counties = gpd.read_file("../input/geospatial-learn-course-data/NY_county_boundaries/NY_county_boundaries/NY_county_boundaries.shp")


# In[ ]:


#county precise the boundaries of a base map 


# We plot our datasets together, withe the function plot. and we get the plot as following: 

# In[ ]:


ax = counties.plot(figsize= (10,10), color= 'none', edgecolor='gainsboro', zorder=3)
wild_lands.plot(color='lightgreen', ax= ax)
campsites.plot(color= 'maroon', markersize=2, ax= ax)
trails.plot(color='red', markersize=1, ax=ax)


# ## Conclusion
# From this map, we can see where the campisties are located and where people go most . and it is the nothern region of NY.
