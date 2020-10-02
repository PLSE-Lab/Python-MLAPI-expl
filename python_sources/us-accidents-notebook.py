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


# **Importing the data file to do the tango with...**

# In[ ]:


import pandas as pd
accidents = pd.read_csv("../input/us-accidents/US_Accidents_Dec19.csv")
accidents.head()


# **Top five rows give us idea of exactly what monstrosity of a data are we looking at. 
# Notice the bottom left info about data. 49 columns describing where, what, when, how, and at what condition an accident occured in USA. Now lets get to business. Plotting stuff. Firstly importing required library to plot stuff**

# In[ ]:


pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# **1. Which is the bloodiest state? which state has the highest acccident count**

# In[ ]:


state_wise= accidents.State.value_counts()
plt.figure(figsize=(30,6))
sns.barplot(x=state_wise.index[:15,], y=state_wise[:15,])


# **California guilty as charged**

# **Next up- 2. What god forsaken time do motorist are on road to meet accidents with! For that the time from data needs to be readable. Rescue 'Time'**

# In[ ]:


import datetime


# In[ ]:


times = pd.DatetimeIndex(accidents.Start_Time)
t = times.hour.value_counts()
t= t.sort_values(ascending=False)
plt.figure(figsize=(30,6))
sns.barplot(x=t.index, y=t)


# **Peak hours are risky then after all! **

# **3. Now to map the locations and see the result on map. We need maps for that. To the import counter we head out**

# In[ ]:


import geopandas as gpd
from shapely.geometry import LineString
import folium
from folium import Choropleth, Circle, Marker
from folium.plugins import HeatMap, MarkerCluster


# **A routine function to view the maps in a notebook window. Just need it here is all i would say:-**

# In[ ]:


# Function for displaying the map
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


# In[ ]:


#Getting the world map for base
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
americas = world.loc[world['continent'].isin(['North America', 'South America'])]

USA = world.loc[world['name'].isin(['United States of America'])]
USA.head()


# **This map will show you accidents on a heat map**

# In[ ]:


# Create a base map
m_1 = folium.Map(location=[42.32,-71.0589], tiles='cartodbpositron', zoom_start=4)

# Add a heatmap to the base map
HeatMap(data=accidents[['Start_Lat', 'Start_Lng']], radius=10).add_to(m_1)

# Display the map
embed_map(m_1, 'm_1.html')

