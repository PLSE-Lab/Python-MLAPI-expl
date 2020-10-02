#!/usr/bin/env python
# coding: utf-8

# # Intro
# 
# The aim of this notebook is to use density based clustering (DBSCAN) to identify accident hotspots in central London. DBSCAN groups points that are closely packed together and marks points outside of these groups as noise. Therefore, using this algorithm, locations in which a high density of accidents take place will be highlighted as clusters. We can then plot the location of these clusters using folium. First we need to load in the data:

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import folium
from geopy.distance import great_circle
from sklearn.cluster import DBSCAN as dbscan
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
        
df = pd.read_csv('../input/2000-16-traffic-flow-england-scotland-wales/accidents_2012_to_2014.csv', usecols=['Latitude','Longitude','Number_of_Vehicles',
                                                                   'Time','Local_Authority_(Highway)','Year'])

df = df[df['Year'] == 2014] # Focus on accidents that took place in 2014
City = df[df['Local_Authority_(Highway)'] == 'E09000001'] # Investigate City and Westminster boroughs
Westminster = df[df['Local_Authority_(Highway)'] == 'E09000033']

df = pd.concat([City, Westminster], axis = 0)
df['Time'] = pd.to_datetime(df['Time'], format = '%H:%M').dt.hour # convert time to the nearest hour, we shall make use of this later


# DBSCAN requires a metric to use when calculating the distance between points. To do this we shall create a function that takes the latitude and longitude of two points and calculates the distance between them in meters. The great_circle function from geopy is used to ensure that the curvature of the Earth is taken into account when calculating these distances. 

# In[ ]:


def greatcircle(x,y):
    lat1, long1 = x[0], x[1]
    lat2, long2 = y[0], y[1]
    dist = great_circle((lat1,long1),(lat2,long2)).meters
    return dist


# Now we can find the clusters. The eps parameter determines the maximum distance between two samples for one to be considered as in the neighborhood of the other. This and the min_samples parameter can be adjusted to change the size and number of points contained in each cluster. Points not located within clusters are given the label -1.

# In[ ]:


eps = 100 #distance in meters
min_samples = 10

df_dbc = df

loc = df_dbc[['Latitude','Longitude']]

dbc = dbscan(eps = eps, min_samples = min_samples, metric=greatcircle).fit(loc)

labels = dbc.labels_
unique_labels = np.unique(dbc.labels_)

print(unique_labels)

df_dbc['Cluster'] = labels


# Now that we have found the clusters we can plot them on a map using folium. We can assign colours to each cluster so that they can be more easily identified.

# In[ ]:


location = df_dbc['Latitude'].mean(), df_dbc['Longitude'].mean()

m = folium.Map(location=location,zoom_start=13)

folium.TileLayer('cartodbpositron').add_to(m)

clust_colours = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99','#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928']

for i in range(0,len(df_dbc)):
    colouridx = df_dbc['Cluster'].iloc[i]
    if colouridx == -1:
        pass
    else:
        col = clust_colours[colouridx%len(clust_colours)]
        folium.CircleMarker([df_dbc['Latitude'].iloc[i],df_dbc['Longitude'].iloc[i]], radius = 10, color = col, fill = col).add_to(m)

m


# We have found the spacial clusters, but what if we want to find the regions *and* times that accidents are most likely to occur?  To achieve this we need to perform spaciotemporal clustering. This can be done by creating a new distance metric that takes both space and time into account:

# In[ ]:


# First set the limits for the distances used in space and time for clustering
spatial_dist_max = 500 # meters
temporal_dist_max = 5 # hours

def GreatCircle(lat1,long1,lat2,long2):
    dist = great_circle((lat1,long1),(lat2,long2)).meters
    return dist

def SpaceTimeDistance(x,y):
    diff_days = math.fabs(x[2] - y[2])
    if (np.isnan(diff_days) or diff_days > temporal_dist_max):
        return np.Infinity
    
    try:
        gc_dist = GreatCircle(x[1],x[0],y[1],y[0])
    except ValueError:
        #print(x[1],x[0],y[1],y[0])
        gc_dist = np.Infinity
    
    if (gc_dist>spatial_dist_max):
        return np.Infinity
    
    ratio_t=diff_days/temporal_dist_max
    ratio_d=gc_dist/spatial_dist_max
    if (ratio_d>ratio_t):
        return gc_dist
    else:
        return ratio_t * spatial_dist_max


# Use this metric to determine the clusters:

# In[ ]:


eps = 100
min_no_samples = 5

clustered = dbscan(eps = eps, metric=SpaceTimeDistance, min_samples=min_no_samples).fit(df[['Latitude','Longitude','Time']])
labels=clustered.labels_
unique_labels=np.unique(clustered.labels_)

df_sptc = df
df_sptc['Cluster'] = labels

print(unique_labels)


# Once again we can plot these clusters onto a map using folium. In this case I thought it would be interesting to colour the clusters according to the number of vehicles involved in the accident.

# In[ ]:


df_sptc['colour'] = df['Number_of_Vehicles']

df_sptc.loc[df_sptc['Number_of_Vehicles'] == 1, 'colour'] = '#e31a1c' #red
df_sptc.loc[df_sptc['Number_of_Vehicles'] == 2, 'colour'] = '#1f78b4' #blue
df_sptc.loc[df_sptc['Number_of_Vehicles'] == 3, 'colour'] = '#b2df8a' #green
df_sptc.loc[df_sptc['Number_of_Vehicles'] == 4, 'colour'] = '#ff7f00' #orange

location = df_sptc['Latitude'].mean(), df_sptc['Longitude'].mean()
m = folium.Map(location=location,zoom_start=13)
folium.TileLayer('cartodbpositron').add_to(m)

for i in range(0,len(df_sptc)): 
    if df_sptc['Cluster'].iloc[i] == -1:
        pass
    else:
        col = df_sptc['colour'].iloc[i]
        folium.CircleMarker([df_sptc['Latitude'].iloc[i],df_sptc['Longitude'].iloc[i]], radius = 10, color = col, fill = col).add_to(m)

m


# This shows that accidents involving two or more vehicles are more likely to take place at busy junctions such as those near bridges. Conversly, accidents involving one vehicle are generally found in areas containing large numbers of pedestrians such as Oxford Street.
# 
# We can try and visualise the hours in which these accidents take place using a space-time cube:

# In[ ]:


plotpoint_colour = 'colour'

fig = plt.figure(1)
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=20, azim=30) # change parameters here for experimenting

df_sptc_nn = df_sptc[df_sptc['Cluster'] != -1] # remove noise
ax.scatter(df_sptc_nn['Longitude'], df_sptc_nn['Latitude'], df_sptc_nn['Time'], c = df_sptc_nn[plotpoint_colour], alpha=0.5)

plt.show()


# Unfortunately this representation is fairly confusing but it does provide a general idea of the cluster locations in spacetime. 
# 
# These are just some basic ideas for analysing accident hotspot location, but I hope it provides some inspiration for a more indepth analysis. For example, the DBSCAN parameters in each instance can be adjusted to find clusters of different sizes, and the clusters can be coloured according to different accident conditions.
