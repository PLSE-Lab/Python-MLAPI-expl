#!/usr/bin/env python
# coding: utf-8

# Thanks to BeyondBeneath for the initial Geolocation viz and the code examples [here](https://www.kaggle.com/beyondbeneath/talkingdata-mobile-user-demographics/geolocation-visualisations/comments)
# 
# This notebook is an experiment to determine exactly what the timestamp means in the data, by identifying samples within a few major cities and comparing the hourly distribution of events against the daylight hours in that city. My guess is that the timestamps are in UTC, but they could plausibly be Chinese Standard Time or device local time.
# 
# Let's find out!
# 
# ### Start with the basic imports

# In[ ]:


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
from mpl_toolkits.basemap import Basemap
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Define and plot cities on world map
# 
# Just to make sure the coordinates are right... The huge box around NY is basically the entire Eastern Seaboard of the US, but since they all share EST/EDT, I reckon that'll be fine.

# In[ ]:


# Cities (lat, lon)
beijing = [40, 116.5]
london = [51.5, -0.25]
newyork = [40.75, -74]

# Set up Mercator projection plot for the whole world
plt.figure(1, figsize=(12,6))
m = Basemap(projection='merc',
            llcrnrlat=-60,
            urcrnrlat=65,
            llcrnrlon=-180,
            urcrnrlon=180,
            lat_ts=0,
            resolution='c')
#m.fillcontinents(color='#191919',lake_color='#000000') # dark grey land, black lakes
m.drawmapboundary(fill_color='#000000')                # black background
m.drawcountries(linewidth=0.1, color="w")              # thin white line for country borders

# Plot the cities
m.plot([beijing[1]-0.5,beijing[1]-0.5,beijing[1]+0.5,beijing[1]+0.5,beijing[1]-0.5],
       [beijing[0]-0.5,beijing[0]+0.5,beijing[0]+0.5,beijing[0]-0.5,beijing[0]-0.5],
       latlon=True)
m.plot([london[1]-3,london[1]-3,london[1]+3,london[1]+3,london[1]-3],
       [london[0]-3,london[0]+3,london[0]+3,london[0]-3,london[0]-3],
       latlon=True)
m.plot([newyork[1]-5,newyork[1]-5,newyork[1]+5,newyork[1]+5,newyork[1]-5],
       [newyork[0]-5,newyork[0]+5,newyork[0]+5,newyork[0]-5,newyork[0]-5],
       latlon=True)

plt.title("City locations")
plt.show()


# ### Load the input events and filter by city

# In[ ]:


df_events = pd.read_csv("../input/events.csv", dtype={'device_id': np.str})

idx_beijing = (df_events["longitude"]>beijing[1]-0.5) &              (df_events["longitude"]<beijing[1]+0.5) &              (df_events["latitude"]>beijing[0]-0.5) &              (df_events["latitude"]<beijing[0]+0.5)
df_events_beijing = df_events[idx_beijing]

idx_london =  (df_events["longitude"]>london[1]-3) &              (df_events["longitude"]<london[1]+3) &              (df_events["latitude"]>london[0]-3) &              (df_events["latitude"]<london[0]+3)
df_events_london = df_events[idx_london]

idx_newyork = (df_events["longitude"]>newyork[1]-5) &              (df_events["longitude"]<newyork[1]+5) &              (df_events["latitude"]>newyork[0]-5) &              (df_events["latitude"]<newyork[0]+5)
df_events_newyork = df_events[idx_newyork]

print("Total # events:", len(df_events))
print("Total # Beijing events:", len(df_events_beijing))
print("Total # London events:", len(df_events_london))
print("Total # New York events:", len(df_events_newyork))


# ### Plot by hour of day

# In[ ]:


plt.figure(1, figsize=(12,18))
plt.subplot(311)
plt.hist(df_events_newyork['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("New York")
plt.subplot(312)
plt.hist(df_events_london['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("London")
plt.subplot(313)
plt.hist(df_events_beijing['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("Beijing")
plt.show()


# Hmm, OK, so the 'City That Never Sleeps' seems to have nodded off at dinner, and London is most active between midnight and 6am? So I guess it's not even close to UTC. Beijing looks reasonable, so I guess we're on Chinese Standard Time?
# 
# Let's try again with the (+8 - -5) = 12 hour difference for NY and (+8 - +1) = 7 hour difference for London.

# In[ ]:


plt.figure(1, figsize=(12,18))
plt.subplot(311)
plt.hist(df_events_newyork['timestamp'].map( lambda x: (pd.to_datetime(x).hour-12)%24 ), bins=24)
plt.title("New York")
plt.subplot(312)
plt.hist(df_events_london['timestamp'].map( lambda x: (pd.to_datetime(x).hour-7)%24 ), bins=24)
plt.title("London")
plt.subplot(313)
plt.hist(df_events_beijing['timestamp'].map( lambda x: pd.to_datetime(x).hour ), bins=24)
plt.title("Beijing")
plt.show()


# That looks much more sensible. There's a definite minimum around 3-4am, and a significant few night owls out to a after midnight.
# 
# So the timestamps appear to be Chinese Standard Time (GMT+8). I know this is a bit trivial, but it's my first Kernel Notebook.
