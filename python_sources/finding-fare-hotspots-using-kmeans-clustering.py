#!/usr/bin/env python
# coding: utf-8

# # Finding fare hotspots using KMeans Clustering
# 
# This aim of this kernel is to use KMeans to divide up the coordinate information present in the data set into clusters. By comparing the clusters generated against fare information, we will show this will add useful features that as input to ML models.

# In[ ]:


import math
import pandas as pd
import numpy as np


# # Preparation of data

# ## Distance Measurements
# 
# - Haversine distance https://en.wikipedia.org/wiki/Haversine_formula (note the use of half angle formulas)
# 
# \begin{align}
# d = 2r\arcsin{\sqrt{\frac{1-\cos{(x_2 - x_1)}}{2} + \cos{x_1}\cos{x_2}\frac{1-\cos{(y_2 - y_1)}}{2}}}
# \end{align}
# 
# where x1 and x2 are the latitude points and y1 and y2 are the longitude points

# In[ ]:


from math import cos, asin, sqrt, radians
def sin_sq_half_angle(r):
    return (1 - cos(r)) / 2

def haversine_formula(d_lat, p_lat, d_lon, p_lon):
    earth_radius = 6371 #km
    
    # Application of the Haversine formula using
    # https://en.wikipedia.org/wiki/Haversine_formula
    # Calculating for d
    
    a = sin_sq_half_angle(d_lat - p_lat) + (cos(p_lat) * cos(d_lat) * sin_sq_half_angle(d_lon - p_lon))
    return earth_radius * 2 * asin(sqrt(a))

def haversine_distance(row):
    p_lon = radians(row['pickup_longitude'])
    d_lon = radians(row['dropoff_longitude'])
    p_lat = radians(row['pickup_latitude'])
    d_lat = radians(row['dropoff_latitude'])
    return haversine_formula(d_lat, p_lat, d_lon, p_lon)


# ## Initial data processsing
# - Filter any rows with missing values
# - Use a bounding box to discard any outliers (Thanks to https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration/notebook)
# - Filter any negative fare amounts or anything beyond a certain threshold ($200)
# - Filter any journeys with no passengers.
# 
# We split the data into two:
# - One contains solely coordinate data for further manipulation
# - One contains data that would be used for training a model.

# In[ ]:


# https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration/notebook
# Use bounding box to discard any outliers
# define bounding box
BB = (-75, -73, 40, 41.5)

# this function will be used with the test set below
def select_within_boundingbox(df, BB):
    return (df.pickup_longitude >= BB[0]) & (df.pickup_longitude <= BB[1]) &            (df.pickup_latitude >= BB[2]) & (df.pickup_latitude <= BB[3]) &            (df.dropoff_longitude >= BB[0]) & (df.dropoff_longitude <= BB[1]) &            (df.dropoff_latitude >= BB[2]) & (df.dropoff_latitude <= BB[3])

def process_data(filename, num_rows=100):
    df = pd.read_csv(filename, nrows=num_rows, parse_dates=["pickup_datetime"]).dropna()
    df = df[select_within_boundingbox(df, BB)]
    
    # Filter any negative fare amounts or anything huge ( > $200 USD)
    df = df[(df.fare_amount > 0) & (df.fare_amount < 200) ]
    
    #Filter journeys with no passengers
    df = df[df.passenger_count > 0]

    distances = df.filter(regex='longitude|latitude')
    df['distance'] = distances.apply(haversine_distance, axis=1)
    df['hour'] = df.pickup_datetime.apply(lambda t: pd.to_datetime(t).hour)
    df['year'] = df.pickup_datetime.apply(lambda t: pd.to_datetime(t).year)
    df = df[['distance', 'hour', 'year', 'fare_amount', 'passenger_count']]
    return df, distances

df, df_coordinates = process_data('../input/new-york-city-taxi-fare-prediction/train.csv', num_rows=50000)
df.describe()


# In[ ]:


df_coordinates.describe()


# # KMeans Clustering
# 
# The number of clusters is generated is completely arbitary. This is a parameter that could be optimised in future updates.

# In[ ]:


from sklearn.cluster import KMeans
pickup_clusters = KMeans(n_clusters=30, random_state=0).fit(df_coordinates.filter(regex='pickup'))
dropoff_clusters = KMeans(n_clusters=30, random_state=0).fit(df_coordinates.filter(regex='dropoff'))

df['pickup_cluster'] = pickup_clusters.labels_
df['dropoff_cluster'] = dropoff_clusters.labels_


# ## Map plotting
# 
# Once we have generated the clusters we can then plot this against a NYC map defined by `BB` boundaries (the image itself is obtained from img export in open street map)
# 
# Interestingly, there seem to be clusters for the dropoff coordinates that are centered around Brooklyn and Queens that are not present in the pickup coordinates. Assuming this isn't a bias in the way the data has been collected, it suggests that people are more likely to call a taxi to get them back to the suburbs, but are less likely to start a journey from there.

# In[ ]:


import matplotlib.pyplot as plt

nyc_map = plt.imread('../input/map-png/map.png')
# Adapted from https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration/notebook
# this function will be used more often to plot data on the NYC map
def plot_on_map(df_coord, BB, nyc_map, figsize=(15, 10), clusters='pickup', colors=True, autoScale=True, title='Unnamed figure'):
    if(clusters == 'pickup'):
        if(colors):
            colors = (df.pickup_cluster-df.pickup_cluster.mean())/df.pickup_cluster.std()
        else:
            colors = 'r'
        longitude = df_coord.pickup_longitude
        latitude = df_coord.pickup_latitude
    elif(clusters == 'dropoff'):
        if(colors):
            colors = (df.dropoff_cluster-df.dropoff_cluster.mean())/df.dropoff_cluster.std()
        else:
            colors = 'r'
        longitude = df_coord.dropoff_longitude
        latitude = df_coord.dropoff_latitude
        
    plt.figure(figsize=figsize)
    plt.scatter(longitude, latitude, zorder=1, alpha=0.4, c=colors, s=2)
    plt.title(title)
    axes = plt.gca()
    if(autoScale):
        if(abs(longitude.max() - longitude.min()) < 0.1):
            xmin = longitude.min() - 0.2
            xmax = longitude.max() + 0.2
        else:
            xmin = longitude.min()
            xmax = longitude.max()
        
        if(abs(latitude.max() - latitude.min()) < 0.1):
            ymin = latitude.min() - 0.2
            ymax = latitude.max() + 0.2
        else:
            ymin = latitude.min()
            ymax = latitude.max()
        axes.set_xlim(xmin, xmax)
        axes.set_ylim(ymin, ymax)
    plt.imshow(nyc_map, zorder=0, extent=[-75, -73, 40, 41.5]);
    
plot_on_map(df_coordinates, BB, nyc_map, clusters='pickup', title='Pickup locations in NYC, divided into clusters by KMeans clustering')
plot_on_map(df_coordinates, BB, nyc_map, clusters='dropoff', title='Dropoff locations in NYC, divided into clusters by KMeans clustering')


# ## Combining cluster and fare information.
# 
# We would like to see if there are certain pickup or dropoff locations that are potentially more expensive. To do this, we can start with a heatmap, plotting pickup against dropoff clusters.

# In[ ]:


df_clusters = df.pivot_table('fare_amount', 'pickup_cluster', 'dropoff_cluster', fill_value=0)
import seaborn as sns

sns.heatmap(df_clusters)


# We can clearly see from the heatmap that there are strips of rows and columns that are more prominent than others. This suggests that certain locations on the map result in more expensive taxi journeys either if you start from these clusters, or finish in these clusters.
# 
# Let's take the 4 most expensive cluster areas on average in which to take a taxi journey.

# In[ ]:


df_clusters.T.mean().nlargest(4)


# If we plot it on a map then:
# - Cluster 1 corresponds approximately to the upper east side area of NY
# - Cluster 2 corresponds to JFK airport
# - Cluster 3 corresponds approximately to Midtown in NY
# - Cluster 5 corresponds to LaGuardia airport

# In[ ]:


plot_on_map(df_coordinates[df.pickup_cluster == 1], BB, nyc_map, colors=False, title='Cluster1: ~Upper east side')
plot_on_map(df_coordinates[df.pickup_cluster == 2], BB, nyc_map, colors=False, title='Cluster2: JFK')
plot_on_map(df_coordinates[df.pickup_cluster == 3], BB, nyc_map, colors=False, title='Cluster3: ~Midtown')
plot_on_map(df_coordinates[df.pickup_cluster == 5], BB, nyc_map, colors=False, title='Cluster5: LaGuardia')


# We can also do the same with dropoff areas:
# 
# - Cluster 1 corresponds approximately La Guardia airport and the surrounding brooklyn area
# - Cluster 3 corresponds to JFK airport
# - Cluster 8 corresponds approximately to Newark airport
# - Cluster 12 corresponds to south brooklyn (around the NY aquarium area) [see here](https://www.google.com/maps/place/Upper+East+Side,+New+York,+NY,+USA/@40.6023071,-74.001599,13z/data=!4m5!3m4!1s0x89c258bf08488f6b:0x618706a9142daa0d!8m2!3d40.7735649!4d-73.9565551)

# In[ ]:


df_clusters.mean().nlargest(4)


# In[ ]:


plot_on_map(df_coordinates[df.dropoff_cluster == 1], BB, nyc_map, clusters='dropoff', colors=False, title='Cluster1: LaGuardia and surrounding area')
plot_on_map(df_coordinates[df.dropoff_cluster == 3], BB, nyc_map, clusters='dropoff', colors=False, title='Cluster3: JFK')
plot_on_map(df_coordinates[df.dropoff_cluster == 8], BB, nyc_map, clusters='dropoff', colors=False, title='Cluster8: Newark')
plot_on_map(df_coordinates[df.dropoff_cluster == 12], BB, nyc_map, clusters='dropoff', colors=False, title='South Brooklyn')


# # Discussion
# 
# From a qualitative perspective, what's interesting is how simple KMeans clustering can be compared to fare data to give interesting information. We can immediately see that:
# 
# - Going to/from airports results in on-average more expensive fares. (also see https://www.kaggle.com/breemen/nyc-taxi-fare-data-exploration for an excellent discussion on fixed fare airport taxi fares)
# - Going from the main city centre areas of NY (Midtown, Upper East Side) are generally more expensive.
# - Taxis are far more likely to go to Brooklyn/Queens then going from it.
# 
# # Future work
# 
# - Demonstrate that adding cluster labels results in improved models. 
# - Provide visual data for cheapest taxi fares as well as the most expensive.
