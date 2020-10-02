#!/usr/bin/env python
# coding: utf-8

# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# ---
# 

# In[ ]:


import geopandas as gpd

from learntools.core import binder
binder.bind(globals())
from learntools.geospatial.ex1 import *


# ### 1) Get the data.
# 
# Use the next cell to load the shapefile located at `loans_filepath` to create a GeoDataFrame `world_loans`.  

# In[ ]:


loans_filepath = "../input/geospatial-learn-course-data/kiva_loans/kiva_loans/kiva_loans.shp"

# Your code here: Load the data
world_loans = gpd.read_file(loans_filepath)

# Check your answer
q_1.check()

# Uncomment to view the first five rows of the data
#world_loans.head()


# In[ ]:


world_loans.head()


# ### 2) Plot the data.
# 
# Run the next code cell without changes to load a GeoDataFrame `world` containing country boundaries.

# In[ ]:


# This dataset is provided in GeoPandas
world_filepath = gpd.datasets.get_path('naturalearth_lowres')
world = gpd.read_file(world_filepath)
world.head()


# Use the `world` and `world_loans` GeoDataFrames to visualize Kiva loan locations across the world.

# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


# Your code here


fig, ax = plt.subplots(figsize=(16, 16))
world.plot(ax=ax,color='green')
world_loans.plot(ax=ax, color="C1")
fig.tight_layout()
# Uncomment to see a hint
#q_2.hint()


# In[ ]:


# Get credit for your work after you have created a map
q_2.check()

# Uncomment to see our solution (your code may look different!)
#q_2.solution()


# ### 3) Select loans based in the Philippines.
# 
# Next, you'll focus on loans that are based in the Philippines.  Use the next code cell to create a GeoDataFrame `PHL_loans` which contains all rows from `world_loans` with loans that are based in the Philippines.

# In[ ]:


# Your code here
PHL_loans = world_loans.loc[world_loans.country=="Philippines"]

# Check your answer
q_3.check()


# ### 4) Understand loans in the Philippines.
# 
# Run the next code cell without changes to load a GeoDataFrame `PHL` containing boundaries for all islands in the Philippines.

# In[ ]:


# Load a KML file containing island boundaries
gpd.io.file.fiona.drvsupport.supported_drivers['KML'] = 'rw'
PHL = gpd.read_file("../input/geospatial-learn-course-data/Philippines_AL258.kml", driver='KML')
PHL.head()


# Use the `PHL` and `PHL_loans` GeoDataFrames to visualize loans in the Philippines.

# In[ ]:


# Your code here

fig, ax = plt.subplots(figsize=(16, 16))
PHL.plot(ax=ax,color='green')
PHL_loans.plot(ax=ax, color="C1")
fig.tight_layout()
# Uncomment to see a hint
#q_4.a.hint()


# In[ ]:


# Get credit for your work after you have created a map
q_4.a.check()

# Uncomment to see our solution (your code may look different!)
#q_4.a.solution()


# Can you identify any islands where it might be useful to recruit new Field Partners?  Do any islands currently look outside of Kiva's reach?
# 
# You might find [this map](https://bit.ly/2U2G7x7) useful to answer the question.

# In[ ]:


# View the solution (Run this code cell to receive credit!)
q_4.b.solution()

