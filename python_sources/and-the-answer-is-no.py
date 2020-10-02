#!/usr/bin/env python
# coding: utf-8

# ## Astrology refuted
# 
# If we stay within the realm of science rather than superstition, I could think of two major hypotheses to test: that earthquakes correlate to the time of day (the sun's relative longitude) or to the tidal phase (the moon's relative longitude). Such a relation would show up as a diagonal on the two graphs below. Instead, the only pattern seen corresponds to the earths geology, with no visible influences from cosmic forces.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from mpl_toolkits.basemap import Basemap, cm


# In[ ]:


df = pd.read_csv('../input/SolarSystemAndEarthquakes.csv')
for x in df.columns:
    print(x)


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 8))
df.plot.scatter('earthquake.longitude', 'Moon.longitude', c='earthquake.mag', cmap='hot', ax=ax)


# In[ ]:


fig, ax = plt.subplots(figsize=(12, 8))
df.plot.scatter('earthquake.longitude', 'Sun.longitude', c='earthquake.mag', cmap='hot', ax=ax)


# All the structure that can be seen in these two graphs is vertical lines, corresponding to areas on the earth with high seismic activity. There is nothing diagonal seen, that would correspond to sun or moon influence.
# 
# And, just because we can, let's plot the quakes on a map. (We shift everything 10 degrees to the right to keep Alaska in one piece.)

# In[ ]:


fig, ax = plt.subplots(figsize=(12, 8))
m = Basemap(llcrnrlon=-170, llcrnrlat=-80,
            urcrnrlon=190, urcrnrlat=80, 
            projection='mill')
m.ax = ax
x, y = m(df['earthquake.longitude'].values, 
         df['earthquake.latitude'].values)
z = df['earthquake.mag'].values
m.scatter(x, y, c=z, cmap='hot')
m.drawcoastlines(linewidth=1.)


# Thanks for a great dataset!
