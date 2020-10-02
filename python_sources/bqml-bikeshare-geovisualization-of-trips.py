#!/usr/bin/env python
# coding: utf-8

# # Introduction: Deeper Insights on Austin Bike Share
# 
# ![bike rentals](https://upload.wikimedia.org/wikipedia/commons/thumb/a/a0/Bay_Area_Bike_Share_launch_in_San_Jose_CA.jpg/640px-Bay_Area_Bike_Share_launch_in_San_Jose_CA.jpg)
# 
# This is the 2nd kernel on the Austin bike share dataset.
# https://www.kaggle.com/dcstang/bigquery-ml-austin-bike-share-2013-2019
# 
# We will have a deeper look at one aspect of the dataset
# >  Geographical impact on station hire
# 

# # 1. Setup Bigquery Environment, Libraries & Loading the Data
# 
# Loading up all the Python libraries for data analysis, and loading the data from SQL queries.
# All this was done previously in the first kernel - so all the code is masked.

# In[ ]:


import pandas as pd
import numpy as np
import os

#import data visualization libraries
import missingno
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd


PROJECT_ID = 'kaggle-tutorial-249207'

from google.cloud import bigquery
client = bigquery.Client(project=PROJECT_ID, location="US")
dataset = client.create_dataset('model_dataset', exists_ok=True)

from google.cloud.bigquery import magics
from kaggle.gcp import KaggleKernelCredentials
magics.context.credentials = KaggleKernelCredentials()
magics.context.project = PROJECT_ID

get_ipython().run_line_magic('load_ext', 'google.cloud.bigquery')


# # 2. Getting 2017 & 2018 Data, and making Geopandas Points

# In[ ]:


get_ipython().run_cell_magic('bigquery', 'location_rides_2018', "SELECT \n    t.start_station_name as station_name,\n    s.status as station_status,\n    s.latitude as latitude, \n    s.longitude as longitude,\n    COUNT(bikeid) as num_rides\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips` as t\nINNER JOIN `bigquery-public-data.austin_bikeshare.bikeshare_stations` as s\n    ON t.start_station_name = s.name\nWHERE DATE(start_time) BETWEEN '2018-01-01' AND '2019-01-01'\nGROUP BY station_name, station_status, latitude, longitude")


# In[ ]:


location_rides_2018.tail()


# In[ ]:


gdf = gpd.GeoDataFrame(
    location_rides_2018, geometry=gpd.points_from_xy(location_rides_2018.longitude, location_rides_2018.latitude))


# In[ ]:


gdf.head()


# In[ ]:


gdf.max()


# In[ ]:


get_ipython().run_cell_magic('bigquery', 'location_rides_2017', "SELECT \n    t.start_station_name as station_name,\n    s.status as station_status,\n    s.latitude as latitude, \n    s.longitude as longitude,\n    COUNT(bikeid) as num_rides\nFROM `bigquery-public-data.austin_bikeshare.bikeshare_trips` as t\nINNER JOIN `bigquery-public-data.austin_bikeshare.bikeshare_stations` as s\n    ON t.start_station_name = s.name\nWHERE DATE(start_time) BETWEEN '2017-01-01' AND '2018-01-01'\nGROUP BY station_name, station_status, latitude, longitude")


# In[ ]:


gdf_2017 = gpd.GeoDataFrame(
    location_rides_2017, geometry=gpd.points_from_xy(location_rides_2017.longitude, location_rides_2017.latitude))


# In[ ]:


gdf_2017.head()


# In[ ]:


# Use Geopandas to read the Shapefile
austin_shp_gdf = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.dbf')
#austin_shp_gdf = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.sbn')
#austin_shp_gdf = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.sbx')
austin_shp_gdf = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shp')
austin_shp_gdf = gpd.read_file('../input/data-science-for-good/cpe-data/Dept_37-00027/37-00027_Shapefiles/APD_DIST.shx')

austin_shp_gdf.crs = {'init' :'esri:102739'}
austin_shp_gdf = austin_shp_gdf.to_crs(epsg='4326')


# In[ ]:


location_rides_2017_grouped = location_rides_2017.groupby("station_name").sum() #station names as index
stations_2017 = location_rides_2017_grouped.index #get all station names

location_rides_2018_grouped = location_rides_2018.groupby("station_name").sum() #station names as index
stations_2018 = location_rides_2018_grouped.index #get all station names

stations_new_2018 = sorted(list(set(stations_2018) - set(stations_2017)))

gdf_new = (gdf.loc[gdf['station_name'].isin(stations_new_2018)])
#get new stations list


# # 3. Plotting Data

# In[ ]:


fig1,ax1 = plt.subplots(nrows=1, ncols=2, figsize=(25, 25))
austin_shp_gdf.plot(ax=ax1[1],color='#556B2F')
gdf.plot(ax=ax1[1],marker='.',color='#FFE67C',markersize=(gdf['num_rides'])/27)
gdf_new.plot(ax=ax1[1],marker='.',color='#F76D82',markersize=(gdf_new['num_rides'])/27,label='new stations')
ax1[1].set_xlim([-97.8,-97.7])
ax1[1].set_ylim([30.23,30.31])
ax1[1].title.set_text('Bike Share 2018 in Austin,Tx')
ax1[1].axis("off")
ax1[1].legend(loc='lower center', framealpha=0.1, borderpad = 1)

austin_shp_gdf.plot(ax=ax1[0],color='#556B2F')
gdf_2017.plot(ax=ax1[0],marker='.',color='#FFE67C',markersize=(gdf_2017['num_rides'])/27)
ax1[0].set_xlim([-97.8,-97.7])
ax1[0].set_ylim([30.23,30.31])
ax1[0].title.set_text('Bike Share 2017')
ax1[0].axis("off")

l1 = plt.scatter([],[], s=10, c='#FFE67C', edgecolors='none')
l2 = plt.scatter([],[], s=80, c='#FFE67C',edgecolors='none')
l3 = plt.scatter([],[], s=250, c='#FFE67C', edgecolors='none')
l4 = plt.scatter([],[], s=600, c='#FFE67C', edgecolors='none')

labels = ["1k", "5k", "20k", "70k"]

ax1[0].legend([l1, l2, l3, l4], labels, ncol=4, frameon=True, fontsize=10,
handlelength=2, loc = 8, borderpad = 1, framealpha=0.1,
handletextpad=1, title='Total Number of Trips started', scatterpoints = 1)
#plt.setp(leg.get_texts(), color='w')


# # 4. Insights from geographical data
# 
# **Few insights from this new data:**
# - Explosion of new stations on the northern side of the city
# - sprinkling of new stations in between old routes, mainly along large roads
# - slight decrease of number of rides in the southern side (~10-20%)
# - only increase of rides for *old stations* is within the northern part
# 
# **Business insights:**
# - addition of new stations on northern site is profitable and increases the number of rides for the older stations in proximity.
# - growth on the southern / western side is not as pronounced, customers maybe flocking to northern side

# # 5. What's next
# 
# I would like to examine the seasonality of the data as well with BigQuery. Will post a link here once it's ready.

# In[ ]:



# %bigquery test
# SELECT 
#     EXTRACT(MONTH from start_time) as month_2017,
#     COUNT(*) as monthly_rides_2017
# FROM `bigquery-public-data.austin_bikeshare.bikeshare_trips`
# WHERE
#     DATE(start_time) BETWEEN '2017-01-01' AND '2018-01-01'
# GROUP BY 
#     monthly_rides_2017, month_2017
    

