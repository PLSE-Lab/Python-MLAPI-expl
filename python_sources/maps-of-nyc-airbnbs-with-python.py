#!/usr/bin/env python
# coding: utf-8

# <h1 align="center"> Spatial Analysis of Airbnb listings in NYC</h1>
# 
#  <h2> Before we Begin: </h2> 
#  To reproduce this work, you will need to the following datases from the NYC Open Data site:<br>
#  1.   https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas/cpf4-rkhq<br>
#  2.   https://data.cityofnewyork.us/Transportation/Subway-Stations/arq3-7z49 <br><br>
#  **This project also takes advantage of GeoPandas for spatial analysis/plotting of maps: http://geopandas.org/** <br><br>
#  
#  <h2> Introduction </h2>In this kernel I explore the NYC Airbnb listings dataset in a spatial context. I also conduct an analysis to find a listing that meets specific criteria for an upcoming trip where I theoretically attend a New York Yankees baseball game.
#  
#  <br><br><br>
#  <h2> Outline: </h2>
#  
#  I. <b> Understanding our data</b> <br>
#  a) [Explore Data](#explore)<br>
#  
#  II. <b> Borough/Neighborhood Charts and Maps</b> <br>
#  a) [Borough Plots](#cbbmap)<br>
#  b) [Neighborhood Map](#nbmap)<br>
#  c) [Fixed Neighborhood Map](#rnbmap)<br>
#  
#  III. <b> Yankee Stadium Analysis</b><br>
#  a) [Criteria](#yankeestadium)<br>
#  b) [Mapping Yankee Stadium](#yankeesmap)<br>
#  c) [Subway Analysis](#subways)<br>
#  d) [Criteria Analysis](#criteria)<br>
#  e) [Airbnb's near Subways](#abnearsubways)<br>
#  
#  IV. <b> Final Results! </b> <br>
#  a) [Airbnbs that meet all criteria](#finalresults)<br>
# 

# In[ ]:


#Import various python packages
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely import wkt

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

plt.style.use('fivethirtyeight')


# # Understanding our Data
#  <a id="explore"></a>First we will bring in the data, look at its structure, data types, and do make some simple plots

# In[ ]:


#Create a pandas dataframe of the Airbnb data
data = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

data.head(5)


# In[ ]:


#Review the data types
data.dtypes


# In[ ]:


#Review the columns 
data.columns


# In[ ]:


#Rename a column to accurately reflect Boroughs
data.rename(columns={'neighbourhood_group':'boroname'}, inplace=True)


# In[ ]:


#Review the listings by boroname
plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude', y='latitude', hue='boroname',s=20, data=data)


# # Borough/Neighborhood Charts & Maps:
#  <a id="cbbmap"></a>
# ### Lets do an aggregation by borough, look at a bar chart, then use geopandas to read in a basemap

# In[ ]:


#Get a count by borough
borough_count = data.groupby('boroname').agg('count').reset_index()


# In[ ]:


#Plot the count by borough
fig, ax1 = plt.subplots(1,1, figsize=(6,6)
                       )
sns.barplot(x='boroname', y='id', data=borough_count, ax=ax1)

ax1.set_title('Number of Listings by Borough', fontsize=15)
ax1.set_xlabel('Borough', fontsize=12)
ax1.set_ylabel('Count', fontsize=12)
ax1.tick_params(axis='both', labelsize=10)


# In[ ]:


#Here we are using geopandas to bring in a base layer of NYC boroughs
nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
nyc.head(5)


# In[ ]:


#Rename the column to boroname, so that we can join the data to it on a common field
nyc.rename(columns={'BoroName':'boroname'}, inplace=True)
bc_geo = nyc.merge(borough_count, on='boroname')


# In[ ]:


#Plot the count by borough into a map
fig,ax = plt.subplots(1,1, figsize=(10,10))
bc_geo.plot(column='id', cmap='viridis_r', alpha=.5, ax=ax, legend=True)
bc_geo.apply(lambda x: ax.annotate(s=x.boroname, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)
plt.title("Number of Airbnb Listings by NYC Borough")
plt.axis('off')


# ### Lets explore neighborhoods:
#  <a id="nbmap"></a>
# ### We do not have the geometries of neighborhoods, so lets bring in a CSV file from the NYC Open Data Site. It has wkt in a geometry column. 
# ### We can convert that to a GeoPandas Data Frame

# In[ ]:


#Now,lets take a look at the count by neighborhood. Use the file downloaded from https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas/cpf4-rkhq
nbhoods = pd.read_csv('../input/nbhoods/nynta.csv')
nbhoods.head(5)


# In[ ]:


#There is a lot going on here... first rename the column
nbhoods.rename(columns={'NTAName':'neighbourhood'}, inplace=True)

#Then, since this is a csv file, convert the geometry column text into well known text, this will allow you to plot its geometry correctly
nbhoods['geom'] = nbhoods['the_geom'].apply(wkt.loads)

#Now convert the pandas dataframe into a Geopandas GeoDataFrame
nbhoods = gpd.GeoDataFrame(nbhoods, geometry='geom')


# In[ ]:



#Lets take a look at what the neighborhoods look like
fig,ax = plt.subplots(1,1, figsize=(8,8))
nbhoods.plot(ax=ax)


# In[ ]:


#Lets get a count by neighborhood
nbhood_count = data.groupby('neighbourhood').agg('count').reset_index()


# In[ ]:


#Lets merge the spatial GeoPandas Dataframe (with geometry), with the nbhood_count layer that is aggregated
nb_count_geo = nbhoods.merge(nbhood_count, on='neighbourhood')
nb_count_geo.head(3)


# In[ ]:


#Lets take a look at the count by neighborhood
fig,ax = plt.subplots(1,1, figsize=(10,10))

base = nbhoods.plot(color='white', edgecolor='black', ax=ax)

nb_count_geo.plot(column='id', cmap='plasma_r', ax=base, legend=True)

plt.title("Number of Airbnb Listings by Neighborhood")
ax.text(0.5, 0.01,'White = No Data',
       verticalalignment='bottom', horizontalalignment='left',
       transform=ax.transAxes,
       color='blue', fontsize=12)
plt.axis('off')


# ### WOAH! See a problem? This is no bueno.
# ### Why are there so many areas with missing data?
# ### My guess is that the neighborhood names do not exactly align**

# ### The neighborhood names do not exactly align:
# # <a id="rnbmap"></a>
# # **Lets instead do some spatial operations to find exactly where each Airbnb is located**

# In[ ]:


#Create a point of each Airbnb location, and enable the "data" dataframe into a geopandas dataframe
data = gpd.GeoDataFrame(data, geometry=gpd.points_from_xy(data.longitude, data.latitude))

#Now, do a spatial join... This code here runs an intersect analysis to find which neighborhood the Airbnb location is in
joined = gpd.sjoin(nbhoods, data, how='inner', op='intersects')


# In[ ]:


#Lets take a look 
joined.head(3)


# In[ ]:


#Drop the second geometry column
joined.drop(columns='geom', inplace=True)


# In[ ]:


#Rename the column. 
joined.rename(columns={'neighbourhood_left':'neighbourhood'}, inplace=True)

#Create a count of each neighborhood
nb_join_count = joined.groupby('neighbourhood').agg('count').reset_index()


# In[ ]:


#Get the "true count". Join this data to the original neighborhoods geometry 
true_count = nbhoods.merge(nb_join_count, on='neighbourhood')


# In[ ]:


#Lets plot the data
fig,ax = plt.subplots(1,1, figsize=(10,10))

base = nbhoods.plot(color='white', edgecolor='black', ax=ax)

true_count.plot(column='id',cmap='plasma_r', ax=base, legend=True)
plt.title('Number of Airbnb listings by Neighborhood in NYC')


# ### **BEAUTIFUL! **

# ### **I now want to do some additional spatial analysis.**
# #### **Lets say that I am visiting NYC to go check out a Yankee game. ** <br>
#  <a id="yankeestadium"></a>
#   **Here are my criteria:**
#   <ol>
#      <li> I want to stay within 2 miles of the stadium</li>
#      <li> I want to stay within 1/4 mile of a subway station that services the D or 4 line</li>
#      <li> I want to stay in a place that is less than $250 a night </li>
#      <li> The host must have more than 10 reviews</li>
#      <li> The minimum night max can not be more than 3 days. We are only coming for an extended weekend</li>
#      <li> I want to stay in a place where I can have the entire apartment/home. I have friends coming in</li>
#      </ol>
# 

# # Yankee Stadium Maps & Analysis:
#  <a id="yankeesmap"></a>
# 

# In[ ]:


#Create a data frame, and add data for Yankee stadium to it
yankee_stadium = pd.DataFrame()
yankee_stadium['name'] = ["Yankee Stadium"]
yankee_stadium['lon'] = -73.926186
yankee_stadium['lat'] = 40.829659
yankee_stadium


# In[ ]:


#Create a geodataframe of Yankee Stadium
yankee_stadium= gpd.GeoDataFrame(yankee_stadium, geometry=gpd.points_from_xy(yankee_stadium.lon, yankee_stadium.lat))


# In[ ]:


#Lets plot the data
fig,ax1 = plt.subplots(1,1, figsize=(10,10))
base = nbhoods.plot(color='orange',alpha=0.5, edgecolor='black', ax=ax1)
yankee_stadium.plot(markersize=300,ax=base)
plt.title('Yankee Stadium and NYC')


# In[ ]:


#Lets filter the neighborhoods down to Manahattan and the Bronx
man_bronx_geo = nbhoods.loc[(nbhoods['BoroName'] == 'Manhattan') | (nbhoods['BoroName'] == 'Bronx')]


# In[ ]:


#Plot Yankee Stadium with the Bronx and Manhattan
fig,ax = plt.subplots(1,1, figsize=(10,10))
yankee_stadium.plot(markersize=300,color='red',ax=ax)
man_bronx_geo.plot(column='BoroName', cmap = 'tab20b',alpha=.5, ax=ax, legend=True)
plt.title("Bronx, Manhattan, and Yankee Stadium")


# #### Looking at Subways:
#  <a id="subways"></a>
# Lets add subway data

# In[ ]:


#Create a pandas dataframe of the Airbnb data
subways = pd.read_csv('../input/nyc-subway-stations/DOITT_SUBWAY_STATION_01_13SEPT2010.csv')
subways.head(5)


# In[ ]:


#Then, since this is a csv file, convert the geometry column text into well known text, this will allow you to plot its geometry correctly
subways['geom'] = subways['the_geom'].apply(wkt.loads)

#Now convert the pandas dataframe into a Geopandas GeoDataFrame
subways = gpd.GeoDataFrame(subways, geometry='geom')


# In[ ]:


#Lets take a look at what the neighborhoods look like
fig,ax = plt.subplots(1,1, figsize=(8,8))
subways.plot(ax=ax)
yankee_stadium.plot(markersize=100,ax=ax)
plt.title('NYC Subway Stations and Yankee Stadium', fontsize=12)


# In[ ]:


subways = subways[subways['LINE'].str.contains('4') | (subways['LINE'].str.contains('D'))]


# In[ ]:


#Lets take a look at what the neighborhoods look like
fig,ax = plt.subplots(1,1, figsize=(8,8))
subways.plot(ax=ax)
yankee_stadium.plot(markersize=100,ax=ax)
plt.title('NYC Subway Stations Servicing 4 and D lines, with Yankee Stadium', fontsize=10)


# In[ ]:


#Plot the count by borough into a map
fig,ax = plt.subplots(1,1, figsize=(10,10))

yankee_stadium.plot(markersize=300,color='red',ax=ax, label='Yankee Stadium')

subways.plot(markersize=50, color='green',ax=ax, label='Subways')

man_bronx_geo.plot(column='BoroName', cmap = 'tab20b',alpha=.5, ax=ax, legend=True)

plt.title("Bronx, Manhattan, and Yankee Stadium")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')


# In[ ]:


#2 miles in feet is .001 * 32.195122 
yankee_stadium.crs = {'init' :'epsg:2263'}
stadium_buff = yankee_stadium.buffer(.001 * 32.195122)


# In[ ]:


stadium_buff = gpd.GeoDataFrame({'geometry': stadium_buff})


# In[ ]:


stadium_buff


# In[ ]:


stadium_buff.crs = {'init' :'epsg:2263'}
data.crs = {'init' :'epsg:2263'}


# In[ ]:


airbnbs_within_2m_of_ys = gpd.sjoin(data,stadium_buff, how='inner', op='intersects')


# In[ ]:


len(airbnbs_within_2m_of_ys)


# In[ ]:


#Plot the airbnbs within 2 miles of Yankee stadium
fig,ax = plt.subplots(1,1, figsize=(10,10))
airbnbs_within_2m_of_ys.plot(markersize=50,ax=ax, label="Airbnbs")
yankee_stadium.plot(markersize=300,color='red',ax=ax, label="Yankee Stadium")
#man_bronx_geo.plot(column='BoroName', cmap = 'tab20b',alpha=.5, ax=ax, legend=True)
plt.title("Airbnbs within 2 miles of Yankee Stadium")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')


# ### We now have all airbnbs within 2 miles of Yankee Stadium
# ##### We should now select all airbnbs that meet our other crtieria
#  <a id="criteria"></a>
# 

# In[ ]:


#Lets add our crieria, one by one, and see how many listings are left after each
print("Starting number of airbnbs: {0}".format(len(airbnbs_within_2m_of_ys)))

airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['price'] < 250]
print("Number of airbnbs after cutting price to less than $250: {0} ".format(len(airbnbs_within_2m_of_ys)))

airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['availability_365'] > 240]
print("Number of airbnbs after selecting those that are available at least 6 months out of the year: {0} ".format(len(airbnbs_within_2m_of_ys)))

airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['number_of_reviews'] >= 10]
print("Number of airbnbs after selecting those with at least 10 reviews: {0} ".format(len(airbnbs_within_2m_of_ys)))

airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['room_type'] == 'Entire home/apt']
print("Number of airbnbs after selecting those that offer the entire home/apt: {0} ".format(len(airbnbs_within_2m_of_ys)))

airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.loc[airbnbs_within_2m_of_ys['minimum_nights'] <= 3]
print("Number of airbnbs left after selecting airbnbs that have a minimum night stay of 3 or less: {0} ".format(len(airbnbs_within_2m_of_ys)))


# In[ ]:


airbnbs_within_2m_of_ys.head()


# In[ ]:


#Plot the airbnbs within 2 miles of Yankee stadium
fig,ax = plt.subplots(1,1, figsize=(10,10))
airbnbs_within_2m_of_ys.plot(markersize=100,ax=ax, legend=True, label="Airbnbs")
yankee_stadium.plot(markersize=300,color='red',ax=ax, label="Yankee Stadium")
plt.title("Airbnbs' near Yankee Stadium that meet our criteria so far")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')


# ### We now need to verify airbnbs within 1/4 mile of subway stations that serve the D or 4 lines
# # <a id="abnearsubways"></a>

# In[ ]:


#1/4 mile in feet is .001 * 3.657 
subways.crs = {'init' :'epsg:2263'}
subways_buff = subways.buffer(.001 * 3.657)


# In[ ]:


#Create a geodataframe for the subways buffer. Set the crs to 2263
subways_buff = gpd.GeoDataFrame({'geometry': subways_buff})
subways_buff.crs = {'init' :'epsg:2263'}


# In[ ]:


#Rename the index_right column. It can not be in our final spatial join
airbnbs_within_2m_of_ys = airbnbs_within_2m_of_ys.rename(columns={'index_right': 'other_name'})


# # Final Results
# ##### Here we will do our final analysis. We will identify the airbnbs that meet all of our conditions
#  <a id="finalresults"></a>

# In[ ]:


#Lets find the airbnbs that intersect our subway buffers
final_abs = gpd.sjoin(airbnbs_within_2m_of_ys,subways_buff, how='inner', op='intersects')


# In[ ]:


#How many airbnbs meet all of our conditions?
print("There are {0} airbnbs that meet all of the conditions".format(len(final_abs)))


# In[ ]:


#Plot the final results
fig,ax = plt.subplots(1,1, figsize=(10,10))
final_abs.plot(markersize=100,ax=ax, label="Airbnbs")
yankee_stadium.plot(markersize=300,color='red',ax=ax, label="Yankee Stadium")

plt.title("Airbnbs that meet all conditions")

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')


# In[ ]:


#Lets look at the final results
final_abs


# <h1 align="center"> The End!</h1>
# 
#  <h2 align="center"> Thank you for exploring this notebook </h2> 
# 
# 

# In[ ]:




