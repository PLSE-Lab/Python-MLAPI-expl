#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Visualization scripts
# Maps
from mpl_toolkits.basemap import Basemap
# Plots
import matplotlib.pyplot as plt
from matplotlib import cm

# Cluster analysis
from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
df = pd.read_csv('../input/SLC_Police_Cases_2016_cleaned_geocoded.csv')


# ## Some initial wrangling

# In[ ]:


df.dtypes


# In[ ]:


df['reported'] = pd.to_datetime(df['reported'], errors='coerce')
df['occurred'] = pd.to_datetime(df['occurred'], errors='coerce')


# In[ ]:


df.head()


# ## Exploration with initial visualizations

# In[ ]:


df.description.unique()


# In[ ]:


fig = plt.figure()
plt.suptitle("Crime Categories")
ax = fig.add_subplot(111)
df.description.value_counts().plot(ax=ax, kind='bar', title='Category Count')


# ## Some location mapping

# In[ ]:


# SLC Lon lat manual
#west = df.x_gps_coords.min()
#east = df.x_gps_coords.max()
#north = df.y_gps_coords.max()
#south = df.y_gps_coords.min()

north = 41.0
south = 40.0
east = -111.5
west = -112.5


#     fig = plt.figure(figsize=(14,10))
#     ax = fig.add_subplot(111)
#     #m = Basemap(
#     #    projection='merc',
#     #    llcrnrlat=south,
#     #    urcrnrlat=north,
#     #    llcrnrlon=west,
#     #    urcrnrlon=east,
#     #    lat_ts=south,
#     #    resolution='i')
#     m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
#                 llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')
#     x, y = m(df.x_gps_coords.values, df.y_gps_coords.values)
#     m.hexbin(x, y, gridsize=5000, bins='log', cmap=cm.YlOrRd)

# ## Cluster Analysis

# In[ ]:


north = 41.0
south = 40.5
east = -111.7
west = -112.3

df['x_gps_coords'] = pd.to_numeric(df.x_gps_coords, errors='coerce')
df['y_gps_coords'] = pd.to_numeric(df.y_gps_coords, errors='coerce')
latlon = df[['x_gps_coords', 'y_gps_coords']]
latlon = latlon.dropna(axis=0)
kmeans = KMeans(n_clusters=10)
kmodel = kmeans.fit(latlon)
centroids = kmodel.cluster_centers_
clons, clats = zip(*centroids)
fig = plt.figure(figsize=(14,10))
ax = fig.add_subplot(111)
m = Basemap(projection='merc', llcrnrlat=south, urcrnrlat=north,
            llcrnrlon=west, urcrnrlon=east, lat_ts=south, resolution='i')
x, y = m(df.x_gps_coords.values, df.y_gps_coords.values)
#m.scatter(x, y, 1, color='orange')
m.hexbin(x, y, bins)
cx, cy = m(clons, clats)
m.scatter(cx, cy, 3, color='red')
m.drawcounties()


# In[ ]:




