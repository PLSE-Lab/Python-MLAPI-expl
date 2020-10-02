#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import folium 
from folium import plugins

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# #Read in the Store Directory File
# 

# In[ ]:


SB_Df = pd.read_csv('../input/directory.csv')
SB_Df= pd.DataFrame(SB_Df)
SB_Df.isnull().sum()


# In[ ]:


SB_Df['Latitude'].fillna(0, inplace=True)
SB_Df['Longitude'].fillna(0, inplace=True)
SB_Df.isnull().sum()


# In[ ]:


location = SB_Df[['Latitude','Longitude']]
location.fillna(0, inplace=True)
location.isnull().sum()
locationList = location.values.tolist()
len(locationList)
locationList[223]


# In[ ]:


Evansville_Coordinates = (-87.47, 37.98)
# create empty map zoomed in on Evansville
map = folium.Map(location= Evansville_Coordinates, tiles='CartoDB dark_matter', zoom_start=11)
marker_cluster = folium.plugins.MarkerCluster().add_to(map)

for loc in range(0, len(locationList)):
    folium.Marker(locationList[loc], popup=SB_Df['Store Number'][loc]).add_to(marker_cluster)
map

