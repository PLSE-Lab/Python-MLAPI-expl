#!/usr/bin/env python
# coding: utf-8

# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# ---
# 

# # Introduction
# 
# [Kiva.org](https://www.kiva.org/) is an online crowdfunding platform extending financial services to poor people around the world. Kiva lenders have provided over $1 billion dollars in loans to over 2 million people.
# 
# <center>
# <img src="https://i.imgur.com/2G8C53X.png" width="500"><br/>
# </center>
# 
# Kiva reaches some of the most remote places in the world through their global network of "Field Partners". These partners are local organizations working in communities to vet borrowers, provide services, and administer loans.
# 
# In this exercise, you'll investigate Kiva loans in the Philippines.  Can you identify regions that might be outside of Kiva's current network, in order to identify opportunities for recruiting new Field Partners?
# 
# To get started, run the code cell below to set up our feedback system.

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
world_loans = ____

# Check your answer
q_1.check()

# Uncomment to view the first five rows of the data
#world_loans.head()


# In[ ]:


# Lines below will give you a hint or solution code
#q_1.hint()
#q_1.solution()


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


# Your code here
____

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
PHL_loans = ____

# Check your answer
q_3.check()


# In[ ]:


# Lines below will give you a hint or solution code
#q_3.hint()
#q_3.solution()


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
____

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


# View the solution
q_4.b.solution()


# # Keep going
# 
# Continue to learn about **[coordinate reference systems](https://www.kaggle.com/alexisbcook/coordinate-reference-systems)**.

# ---
# **[Geospatial Analysis Home Page](https://www.kaggle.com/learn/geospatial-analysis)**
# 
# 
# 
# 
# 
# *Have questions or comments? Visit the [Learn Discussion forum](https://www.kaggle.com/learn-forum) to chat with other Learners.*
