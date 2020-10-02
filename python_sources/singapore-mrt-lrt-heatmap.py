#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# This kernel visualizes a heatmap of Singapore MRT (Mass Rapid Transit) and LRT (Light Rapid Transit) stations
# 
# This [kernel](https://www.kaggle.com/heesoo37/boston-crimes-starter-eda) by [rgriffin](https://www.kaggle.com/heesoo37) helped and inspired me to use the folium heatmap

# In[ ]:


import pandas as pd

raw_data = pd.read_csv('/kaggle/input/singapore-train-station-coordinates/mrt_lrt_data.csv')

print('Total stations amount:', len(raw_data))
print(raw_data.describe())
raw_data.head()


# Dataframe description indicates no NaN or unusual values, but it is better to check for any NaN's

# In[ ]:


print('NaN values amount:')
print(raw_data.isna().sum())


# it looks like there are 0 columns containing NaN's, so that's good
# 
# Here I separated MRT stations from LRT ones

# In[ ]:


mrt_stations = raw_data.loc[raw_data['type'] == 'MRT']
lrt_stations = raw_data.loc[raw_data['type'] == 'LRT']
print('MRT stations amount:', len(mrt_stations))
print('LRT stations amount:', len(lrt_stations))


# # Drawing heatmap
# 
# The map object centered on Singapore coordinates was created. Then I got dataframes of coordinates for each station type. These dataframes were later used to create heamaps for each station type, with different colors for each type (red for MRT, blue for LRT). Heatmaps were then added to previously created map of Singapore

# In[ ]:


import seaborn as sns
import folium
from folium.plugins import HeatMap
import matplotlib.pyplot as plt

singapore_map = folium.Map(location=[1.3521, 103.8198],
                           zoom_start = 12)

mrt_coordinates = mrt_stations[['lat', 'lng']]
lrt_coordinates = lrt_stations[['lat', 'lng']]

mrt_heatmap = HeatMap(mrt_coordinates, radius=20, gradient={0: 'white', 1:'red'})
lrt_heatmap = HeatMap(lrt_coordinates, radius=20, gradient={0: 'white', 1:'blue'})

mrt_heatmap.add_to(singapore_map)
lrt_heatmap.add_to(singapore_map)
singapore_map


# # Conclusion
# 
# From the created heatmap, we can conclude that MRT stations cover most of the city-state, being concentrated in downtown and stretching to the East, West and North areas. They also provide access to the Changi Ariport. LRT stations are concentrated in 2 different areas where MRT lines are less prevelant. It can be assumed that those 2 areas are too highly poplulated for sparce MRT in the region, therefore LRT serves as a feeder line for the MRT
