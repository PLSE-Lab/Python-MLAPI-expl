#!/usr/bin/env python
# coding: utf-8

# I initially scraped Twitter data during the Australia elections with the idea of doing semtiment analysis, however the results of that are less promising. Maybe when they improve I will share another kernel with the results but for now I decided to have some fun and make a pretty map showing where all the tweets came from. 

# In[ ]:


# Install latest folium because popup doesn't seem to work with current Kaggle folium 0.5.0
get_ipython().system('pip install folium==0.9.0')


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import folium
from folium import IFrame

TWEET_FILE = 'auspol2019.csv'
GEOCODE_FILE = 'location_geocode.csv'
INPUT_PATH = '../input/'
LOCATION_GEOCODE_DATA = '../data/location_geocode.csv'


# In[ ]:


# Read in the tweet file
df = pd.read_csv(INPUT_PATH + TWEET_FILE)
df.shape


# In[ ]:


# Get a count of tweets grouped by location
loc_grouped_df = df[['user_location', 'id']].groupby('user_location', sort=False).count().sort_values(['id'], ascending=False)
loc_grouped_df.columns=['count']
loc_grouped_df.reset_index(inplace=True)
loc_grouped_df.head(10)


# In[ ]:


# Read in the geocode file
geocode_df = pd.read_csv(INPUT_PATH + GEOCODE_FILE)
geocode_df.shape


# In[ ]:


# Add the count column to the geocode dataframe
geocode_df = geocode_df.merge(loc_grouped_df, how='left', left_on='name', right_on='user_location')
geocode_df.shape


# In[ ]:


# Number of locations that were not found
geocode_df[geocode_df['lat'].isnull()].shape


# In[ ]:


# Remove nulls
geocode_df.dropna(inplace=True)
geocode_df.shape


# In[ ]:


# Remove non-alphanum characters from names to avoid breaking folium
geocode_df['name'] = geocode_df['name'].str.replace('[^a-zA-Z\s,]', '')


# In[ ]:


# Make an empty Folium map

m = folium.Map(location=[20,0], tiles="Mapbox Bright", zoom_start=2)

# Add markers one by one
for i in range(0,len(geocode_df)):
    folium.Circle(
      # .item() to avoid json errors by converting numpy.int64 to regular ints.
      location=[geocode_df.iloc[i]['lat'].item(), geocode_df.iloc[i]['long'].item()], 
      #popup=geocode_df.iloc[i]['name'] + ' ' +str(geocode_df.iloc[i]['count'].item()),
      radius=geocode_df.iloc[i]['count'].item()*100,
      color='crimson',
      fill=True,
      fill_color='crimson'
    ).add_to(m)

# Save and display it
m.save('ausvotes_map.html')
m


# You can zoom and pan on the map. Clicking on points for popups showing the location name and count of tweets in that area is commented out because it doens't seem to work when I commit the kernel. The popups work fine when running the kernel interactively in edit mode. (If anyone knows why please let me know and I can fix it.) It also works fine locally.
# 
# The results are not unexpected. Apart from Australia, many tweets come from the UK and US. Also quite a few from Europe, India and South East Asia. Overall there is actually more interest that I would have predicted coming from all over the world.
