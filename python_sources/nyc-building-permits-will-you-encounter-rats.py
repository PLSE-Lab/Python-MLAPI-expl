#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import folium
from folium.plugins import FastMarkerCluster
from folium.plugins import MarkerCluster
from sklearn.cluster import DBSCAN
from folium.plugins import HeatMap
# Any results you write to the current directory are saved as output.


# While starting off with the Exploratory analysis of NY building permit dait was an interesting situation to what other dataset could better contextualize the findings from the preliminary dataset. There were many ideas that floated in my head like analyzing building permits from other cities, or map the location of construction to Traffic adccidents, crime data but further into the analysis, we come to fact that majority of the permits are for residential construction and Rodent problem is very common in New York City.

# In[ ]:


dataframe = pd.read_csv("../input/ny-permit-issuance/DOB_Permit_Issuance.csv", nrows=100000) #Trying to maintain the computational sanity of my device with nrows. 
                                                                                            #please increase limits if you are using more powerful device
ratdata = pd.read_csv("../input/nyc-rat-sightings/Rat_Sightings.csv")


# In[ ]:


dataframe.head(5)


# In[ ]:


dataframe.dtypes


# **After some sneak peak into our dataset,the next step was trying to understand some of the basic trends in the dataset**.

# In[ ]:


ax = dataframe.groupby('BOROUGH')['BOROUGH'].count().sort_values().plot(
    kind='bar', figsize=(10,6), title="Permit count by Borough")
for p in ax.patches:
    ax.annotate(str(p.get_height()), xy=(p.get_x(), p.get_height()))


# From this visualization we can see that Manhattan (are we even surprised) have the most number of building permits compared to the other Borough.

# In[ ]:


ax = dataframe.groupby('Permit Status')['Permit Status'].count().sort_values().plot(
    kind='barh',  figsize=(10,8), title="Number of Permits Issued")


# Here we tried to look at the **Status of the Permit** which gives us a sense that the dataset has almost 98% of issued permits with Re-issued and inprocess making up the rest of the records.

# In[ ]:


dataframe.groupby('Permit Type')['Permit Type'].count().sort_values().plot(kind='bar', figsize=(15,8),
                                                                          title="Comparing types of Permits")


# The netx up was trying to visualize and get the sense of what types of permits were issued predominantly. As we can see **Equipment Work (*EW*)** is the massively issued permit followed by **Plumbing (PL)**

# On our next assesment, I introuduced an asumption. If we look at our dataset the feature residential has two values; **YES** or **NA**. I went ahead with the asumption that all the NA are non-residential permits irrespective of their type and hence came up with following observation

# In[ ]:


res = dataframe[dataframe['Residential']=='YES']['Residential'].count()
nres = dataframe['Residential'].isna().sum()
data = {'residential':res,'Non-Residential':nres}
# print(data)
typedf = pd.DataFrame(data = data,index=['Counts'])
# typedf.head()
typedf.plot(kind='barh', title="Residential Vs Non Residential Permits")


# From here we can see, a large number of the permits were issued for the residential construction which was another major reason to select the other dataset to be rat citing to compare how it can be a problem for residential construction in due time.

# The next exploration was about the time frame for the permits as we look to compare the permits based on the year along with the months to figure out what are the most frequent time for construction.

# In[ ]:


dataframe['Issuance Date'] = pd.to_datetime(dataframe['Issuance Date'])


# In[ ]:


dataframe['issued year'] = dataframe['Issuance Date'].dt.year


# In[ ]:


dataframe['issued month'] = dataframe['Issuance Date'].dt.month


# In[ ]:


dataframe.head()


# In[ ]:


monthDF = dataframe[dataframe['Permit Status']=='ISSUED'].groupby('issued month')['issued month'].count()
monthDF.plot(kind='bar', title="Number of permits by Month", figsize=(8,6))


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2)
monthDF2019 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2019)].groupby('issued month')['issued month'].count()
monthDF2018 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2018)].groupby('issued month')['issued month'].count()

# .plot(
# kind='bar', ax=axes[0,0])
monthDF2019.plot(kind='bar', title="Number of permits by Month 2019", ax=axes[0,0], figsize=(10,8))
monthDF2018.plot(kind='bar', title="Number of permits by Month 2018", ax=axes[0,1], figsize=(10,8))
fig.delaxes(axes[1][1])
fig.delaxes(axes[1][0])


# In[ ]:


borough2018 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2019)]['BOROUGH']
borough2019 = dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2018)]['BOROUGH']


fig = plt.figure() # Create matplotlib figure
ax = fig.add_subplot(111) # Create matplotlib axes
width = 0.4

dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2019)]['BOROUGH'].value_counts().plot(kind='bar', color='blue', ax=ax, width=width, position=1)
dataframe[(dataframe['Permit Status']=='ISSUED') & (dataframe['issued year']== 2018)]['BOROUGH'].value_counts().plot(kind='bar', color='orange', ax=ax, width=width, position=0)
ax.set_ylabel('Counts')
plt.legend(['2018', '2019'], loc='upper right')
ax.set_title('2018 Vs 2019 Borough Building Permit Counts')


# Next up, we use the interactive maps provided by the Folium Library to combine. Building permits into  interactive clusters. We first create the map of New York and then add the clusters on it. Before that we have to deal with our missing data in lat long. In this case we discard the missing data and then plot the clusters.

# In[ ]:


print("There are ",dataframe['LATITUDE'].isna().sum(), "Missing location data")
dataframe = dataframe.dropna(subset=['LATITUDE','LONGITUDE'])
print("Number of missing data:", dataframe['LATITUDE'].isna().sum())


# In[ ]:


NYmap = folium.Map(location=[40.7128,-74.0060],zoom_start=10)
NYmap


# In[ ]:


mc = MarkerCluster()
for row in dataframe[0:10000].itertuples():
    mc.add_child(folium.Marker(location =[row.LATITUDE,row.LONGITUDE],popup = row.BOROUGH))


# In[ ]:


NYmap.add_child(mc)
NYmap


# From a top view, the clusters are few but if you click on individual clusters, it breaks down into smaller clusters all the way down to the street level.

# We bring our rat data into the fold and compare some of its feature in context with our permit data.

# In[ ]:


ratdata.head()


# In[ ]:


ratdata = ratdata[ratdata['Borough'] != 'Unspecified']


# In[ ]:


fig, axes = plt.subplots(nrows=2, ncols=2)
ratdata.groupby('Borough')['Borough'].count().plot(kind='bar', title="Rat Sightings by Borough", ax = axes[0,1], figsize=(12,10))
dataframe.groupby('BOROUGH')['BOROUGH'].count().plot(kind='bar', title='Building Permit By Borough', ax=axes[0,0], figsize=(12,10))
fig.delaxes(axes[1][1])
fig.delaxes(axes[1][0])


# Interesting observation as the two places that were approved for the most number of builing permits were the ones that are mostly affected by Rodent problems. By this we can imagine residential construction in the area will be heavily affected by Rodent issues resulting in a lot of money spent on Exterminators with a possibility of diseases.

# In[ ]:


ratdata = ratdata.dropna(subset=['Latitude', 'Longitude'])


# In order to better contextualize the data, what we do is, we plot the rat sightings as a heatmap over the the map of new york city having the clusters of the building permit data, which would help us realize how many construction will occur in a place which are massively effected by rat problems.

# In[ ]:


heat_data = [[row['Latitude'],row['Longitude']] for index, row in ratdata.iterrows()]

# Plot it on the map
HeatMap(heat_data).add_to(NYmap)
NYmap

