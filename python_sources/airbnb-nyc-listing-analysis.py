#!/usr/bin/env python
# coding: utf-8

# In this exploratory analysis, I focus on room types, location and pricing.

# ## import packages and get data

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import geopandas as gpd
import descartes
from shapely.geometry import Point, Polygon
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
plt.style.use('Solarize_Light2')

get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## first look

# Get a basic idea of what the data looks like

# In[ ]:


ABdata = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data_head = ABdata.head()
data_head


# Understand the data types.
# 

# In[ ]:


ABdata.dtypes


# Check for null values.

# In[ ]:


ABdata.isnull().sum()


# I can see that there are a few null values for listing name and host name. Going back to listing 3647 in the dataframe head, I notice that it has 0 reviews, and NaN values for last_review and number_of_reviews. I infer that the 10052 missing values for the two fields are because the listing is brand new and has not had reviews. 

# In[ ]:


(ABdata.number_of_reviews == 0).sum()


# Here I verify that this is the case, and there are 10052 listing with no reviews yet. 

# In[ ]:


print("Data Dimensions:", ABdata.shape, '\n')
print(len(ABdata.id.unique()), 'unique listing ids.')
print(len(ABdata.host_id.unique()), "unique host ids.")
print(len(ABdata.neighbourhood.unique()), "unique neighbourhoods.")

print("\nRoom Types:")
print(ABdata.room_type.value_counts())


# The number of entire homes and private rooms are in similar range, but the number of shared rooms is much smaller. It is also obvious from these numbers that some hosts are operating multiple listings. 

# In[ ]:


zero_price = ABdata[ABdata.price == 0]
print(zero_price.shape)
zero_price


# An anomaly in the data is that there are 11 listing for which the price is 0. With the exception of 1 listing which has no reviews, the rest have all been reviewed which means they are probably not new listings. It is unlikely that the price is actually 0 (airbnb probably still requires some kind of service fee?) but it may be a reflection of some special promotion or discount, as many of them share the same host. 

# ## which neighborhoods are the most popular?

# In[ ]:


neighbourhood_group_freq = ABdata.neighbourhood_group.value_counts()
print(neighbourhood_group_freq)
ng_freq_plot = neighbourhood_group_freq.plot.bar(title="Neighbourhood group frequency", color = ['#003f5c', '#58508d', '#bc5090', '#ff6361', '#ffa600'])


# The majority of listings are located in Manhattan or Brooklyn. As Manhattan has many offerings for tourists, the proximity to popular tourist destinations may play a prominent factor. Some other factors to consider would be whether pricing differs among neighbourhoods, or the number of reviews. 

# ## does location affect the types of rooms offered?

# Now I create a new dataframe so that I can map the coordinates of the listings.

# In[ ]:


crs= {'init': 'epsg:4326'}
street_map = gpd.read_file('/kaggle/input/nyu-shapefile/nyu_2451_34490.shp')
geometry = [Point(xy) for xy in zip(ABdata.longitude, ABdata.latitude)]
geometry[0:5]
geo_df = gpd.GeoDataFrame(ABdata, crs=crs, geometry=geometry)
geo_df.drop(columns=['latitude','longitude'])
geo_df.head(2)


# In[ ]:


#plot the locations of the listings, colored by the room type
fig, ax = plt.subplots(figsize=(10,10))
street_map.plot(ax = ax, alpha = 0.4, color = "grey")

geo_df[geo_df['room_type'] == 'Private room'].plot(ax=ax, markersize = 5, color = "red", marker = 'o', label = "private room")
geo_df[geo_df['room_type'] == 'Entire home/apt'].plot(ax=ax, markersize = 5, color = "blue", marker = '^', label = "entire apartment")
geo_df[geo_df['room_type'] == 'Shared room'].plot(ax=ax, markersize = 5, color = "yellow", marker = '*', label = "shared room")
plt.legend(prop={'size': 15})


# From a previous cell, we found that there are a couple thousand more entire apartment homes available than private rooms, however the entire epartment homes seem to be more densely clustered, while private rooms are slightly more dispersed. We can notice that shared room listings appear to cluster around popular tourist destinations like Times Square, Metropolitan Meseum and Brooklyn Museum. 

# In[ ]:


data = [['Timesquare', 40.757918, -73.985489], ['Metropolitan Museum of Art', 40.779242, -73.962665], ['Empire State', 40.748432, -73.985557], ['Chinatown', 40.717717, -73.995995]]
pop_dest = df = pd.DataFrame( data, columns = ['name', 'latitude', 'longitude']) 
print(pop_dest)
fig, ax = plt.subplots(figsize=(10,10))
street_map.plot(ax = ax, alpha = 0.4, color = "grey")
geo_df[geo_df['room_type'] == 'Shared room'].plot(ax=ax, markersize = 5, color = "yellow", marker = '*', label = "shared room")
plt.legend(prop={'size': 15})
ax.scatter(x=pop_dest.longitude, y = pop_dest.latitude)


# When we compare shared home listings with the most popular tourist destinations, it seems that the clusters are very close to some of the most popular places: Empire State Building, Times Square, Metropolitan Museum and Chinatown. Next take a look at the pricing to see if there is a relation between budgets and popular listing locations.

# > ## room-type vs pricing

# First I calculate the outliers. First I find the interquartile range (between 25-75 percentiles), multiply it by 1.5 and add the result to the 75% and subtract from the 25%. 

# In[ ]:


# Find the outliers for price and reviews per month (no output displayed)

quart25 = geo_df.price.quantile(0.25) 
quart75 = geo_df.price.quantile(0.75)
inter_quart = quart75 - quart25

min_val = quart25 - 1.5*inter_quart
max_val = quart75 + 1.5*inter_quart

outliers_gone = geo_df[(geo_df.price > min_val) & (geo_df.price < max_val)]
outliers_gone.head()
outliers_gone.shape

outliers = geo_df[geo_df.price > max_val]

quart999 = geo_df.price.quantile(0.999)
top1percent = geo_df[geo_df.price >= quart999]


# After removing the outliers, I plot the price distribution by room-type

# In[ ]:


sns.set_palette('husl')
plt.figure(figsize=(15,10))
scatter = sns.violinplot(x='room_type', y='price', data=outliers_gone)


# Entire home/apt style listings have higher price range, whereas shared rooms have the lowest range. Unlike private rooms and shared rooms which have a larger proportion of listings on the cheaper end, entire home/apt style listings are more evenly distributed (every $50 benchmark has a lump though, which suggest that hosts prefer having more rounded pricing) 
# 
# Now that we know that shared rooms are typically less expensive and clustered around popular tourist attractions, an explanation is that certain tourists are not looking for a comfortable, expensive place to stay, just somewhere convenient and cheap that can afford them access to the places they wish to visit. Having used airbnb myself, I know that many shared room listings operate in a hostel business model, and are not really 'homes' of the host. These listings are more commercial, and they provide less customized/cozy homes, but more convenient and cheap rooms like hostels. 

# ## location vs pricing

# Now plot the listings to see if location also affects pricing. I plotted the outliers separately from the rest of the data to get a better look.

# In[ ]:


# plot listings colored by price 
f = plt.figure(figsize=(28, 10))

ax1 = f.add_subplot(121)
ax2 = f.add_subplot(122)

street_map.plot(ax = ax1, alpha = 0.4, color = "grey")
street_map.plot(ax = ax2, alpha = 0.4, color = "grey")
#street_map.plot(ax = ax3, alpha = 0.4, color = "grey")

pts = ax1.scatter(outliers_gone.longitude, outliers_gone.latitude, marker='o', s = 3, c=outliers_gone.price)
cbar1 = f.colorbar(pts, ax=ax1)
cbar1.set_label('price')
ax1.set_title("Airbnb Pricing Without Outliers")


pts2 = ax2.scatter(outliers.longitude, outliers.latitude, marker='o', s = 3, c=outliers.price)
cbar2 = f.colorbar(pts2, ax=ax2)
cbar2.set_label('price')
ax2.set_title("Airbnb Pricing of Outliers")

'''
pts3 = ax3.scatter(top1percent.longitude, top1percent.latitude, marker='o', s = 3, c=top1percent.price)
cbar3 = f.colorbar(pts3, ax=ax3)
cbar3.set_label('price')
ax3.set_title("Airbnb Pricing For Top 0.01 Percent")
'''


# From these 2 maps, it is clear that Manhattan has the a higher number of expensive listings. For the normal data, the higher priced listings are concentrated around Manhattan's peninsula. In the outlier data, the *most* expensive listings don't seem to be clustered (it would be interesting to understand the listings priced over $8000), but in general the outliers are priced higher than the normal listings, and are also most densely located around manhattan. 

# In[ ]:


# check pricing by borough
sns.set_palette('cubehelix')
plt.figure(figsize=(15,10))
scatter = sns.violinplot(x='neighbourhood_group', y='price', data=outliers_gone)
#scatter2 = sns.violinplot(x='neighbourhood_group', y='price', data=outliers)


# These violin plots also reflect manhattan's higher number of expensive listings, with brooklyn following in second. Compared to the other neighborhood groups, manhattan has a flatter distribution, whereas the other tend to have a larger number of cheaper listings, but a huge range between their most expensive listing price and their average price.

# ## availability and room type

# see if availability distribution changes with room type

# In[ ]:


#availability distribution by room type
sns.set_palette('Paired')
plt.figure(figsize=(15,10))
scatter = sns.violinplot(x='room_type', y='availability_365', data=geo_df)


# whereas private rooms and entire home/apt room types have similar plot, a large proportion of shared room listings require very few minimum nights, whereas another sizable proportion requires many minimum nights (close to a year). This further support the idea that shared rooms listings may be running a different kind of business model than private room and entire home room listings. 
