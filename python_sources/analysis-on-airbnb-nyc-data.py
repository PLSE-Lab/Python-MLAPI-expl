#!/usr/bin/env python
# coding: utf-8

# ![image.png](attachment:image.png)

# **Abstract**
# 
# Airbnb has been around since 2008. It has helped guests see the world and experience local cultures and things that aren't on the normall travel itinerary. It has also help hosts make some extra money and for some it has become a full time job. There are millions of homes and rooms available on Airbnb now. Analysing all of the data that is produced by these millions of homes can help them make better business decisions by understanding the customers and hosts.

# **Imports**
# 
# We will be using pandas to read and manipulate the data. We will also be using the geopandas to help us plot the data on maps and we will be using folium for the maps.

# In[ ]:


import numpy as np
import pandas as pd
import geopandas as gpd

import folium
from folium import Choropleth
from folium.plugins import HeatMap

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Read the data in
df_airbnb = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
# Have a look at the first 5 entries
df_airbnb.head(5)


# In[ ]:


# Print some stats about the data
print('There are {} listings in the dataset'.format(len(df_airbnb)))
print('The dataset has {} columns with the following titles:\n'.format(len(df_airbnb.columns)))
for col in df_airbnb.columns:
    print(col)


# After loading in the dataset we can see that there are 48895 listings and 16 cloumns of data for each. From the preview of the data it can be seen that there are some rows that are missing data. We will next fix the missing data and create a new column from the coordinates so that the listings can be plotted onto a map. 

# **Cleaning Data**

# In[ ]:


# Check which columns have missing data
df_airbnb.isnull().sum()


# We can drop the Airbnb ID as it has no use to us. The host name is redundant in our analysis and we will not be concerned when the last review was submitted.

# In[ ]:


# Drop the 3 columns and have a look at the new dataset
df_airbnb.drop(['id','host_name','last_review'], axis=1, inplace=True)
df_airbnb.head(5)


# In[ ]:


# Fill the missing reviews per month with zero
df_airbnb.fillna({'reviews_per_month': 0}, inplace=True)
df_airbnb.head(5)


# In[ ]:


# One last check to make sure there is no missing data
df_airbnb.isnull().sum()


# **Map Data**
# 
# For the maps to be more useful, I decided to get the shapefiles for all of the neighbourhoods in NYC. NYC Open Data had the files, https://data.cityofnewyork.us/City-Government/Neighborhood-Tabulation-Areas/cpf4-rkhq. Once the data was read in using geopandas, I plotted it on a map to make sure it was correct.

# In[ ]:


gdf_nyc_boundaries = gpd.read_file("../input/nyc-neighbourhood-boundaries/geo_export_9f5919c7-b377-48b3-bc3f-c45b49cba31a.shp")
gdf_nyc_boundaries.set_index('ntacode', inplace=True)
gdf_nyc_boundaries.head(5)


# We need to add the number of listings in each neighbourhood to the neighbourhood in the geodata.

# In[ ]:


count_data_df = df_airbnb[['neighbourhood']].neighbourhood.value_counts()
gdf_nyc_boundaries["listing_count"] = 0

for ii in range(len(count_data_df)):
    key = count_data_df.keys()[ii]
    
    index = gdf_nyc_boundaries.index[gdf_nyc_boundaries['ntaname'] == key]
    if len(index) > 0:
        gdf_nyc_boundaries.loc[index[0], ('listing_count')] = count_data_df[ii]

gdf_nyc_boundaries.head()


# In[ ]:


# This function is used to help plot the maps in the notebook
def embed_map(m, file_name):
    from IPython.display import IFrame
    m.save(file_name)
    return IFrame(file_name, width='100%', height='500px')


# We can now visualize all of the Airbnb listings on the map. We have plotted the number of listings per neighbourhood.

# In[ ]:


listing_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)
Choropleth(geo_data=gdf_nyc_boundaries['geometry'].__geo_interface__,
           key_on="feature.id",
           data=gdf_nyc_boundaries['listing_count'],
           fill_color='YlOrRd',
           legend_name='NYC Neighbourhoods'
          ).add_to(listing_map)
embed_map(listing_map, 'listing_map.html')


# **Listing Prices**

# In[ ]:


df_airbnb.price.describe()


# We can see that the cheapest listing is \$0 and the most expensive is \$10000. The average price for a listing is \$106. This doesn't give us much insight so we will plot the lower 25%, the middle 50% and the upper 25% in different colors to see wat the distribution is.

# In[ ]:


# We will use the 25 and 75 percentile for the plot
def color_producer(val):
    if val < 69:
        return 'forestgreen'
    if val < 175:
        return 'darkorange'
    else:
        return 'darkred'
    
price_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)
    
for i in range(0, len(df_airbnb)):
    folium.Circle(
        location=[df_airbnb.iloc[i]['latitude'], df_airbnb.iloc[i]['longitude']],
        radius=25,
        color=color_producer(df_airbnb.iloc[i]['price'])
    ).add_to(price_map)
embed_map(price_map, 'price_map.html')


# It can be seen from the map that the most expensive listings (top 25%) are in Manhattan. This goes hand in hand with the fact that it is a weathly area. Majority of the middle 50% of listings is in Brooklyn and Queens. Staten Island and The Bronx do't have close to as many listings as the other three boroughs.

# **Top 100 Reviewed listings**

# In[ ]:


df_airbnb_top_reviewed = df_airbnb.nlargest(100, 'number_of_reviews')
df_airbnb_top_reviewed[['number_of_reviews', 'price']].describe()


# We can see that the max number of reviews 629, the min is 323 and the average is 396 out of the top 100 reviewed listings. We will plot the lower 25%, middle 50% and upper 25% on our plot. The average price to stay in one of the top 100 listings is \$95 per night.

# In[ ]:


def color_producer(val):
    if val < 348:
        return 'forestgreen'
    if val < 427:
        return 'darkorange'
    else:
        return 'darkred'
    
review_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)
    
for i in range(0, len(df_airbnb_top_reviewed)):
    folium.Circle(
        location=[df_airbnb_top_reviewed.iloc[i]['latitude'], df_airbnb_top_reviewed.iloc[i]['longitude']],
        radius=25,
        color=color_producer(df_airbnb_top_reviewed.iloc[i]['number_of_reviews'])
    ).add_to(review_map)
embed_map(review_map, 'review_map.html')


# Majority of the top reviewed listings are in Manhattan, Brooklyn and Queens.

# **Listing Types**

# In[ ]:


df_airbnb_rooms = df_airbnb.room_type.unique()

# Print some stats about the types of rooms
print('There are {} types of rooms\n'.format(len(df_airbnb_rooms)))
for ii in range(len(df_airbnb_rooms)):
    print(df_airbnb_rooms[ii])


# In[ ]:


# Get the average price and number of reviews per room type
df_airbnb_rooms_private = df_airbnb.loc[df_airbnb['room_type'] == 'Private room']
df_airbnb_rooms_entire = df_airbnb.loc[df_airbnb['room_type'] == 'Entire home/apt']
df_airbnb_rooms_shared = df_airbnb.loc[df_airbnb['room_type'] == 'Shared room']

print('Private Room Stats')
df_airbnb_rooms_private[['price', 'number_of_reviews']].describe()


# The average price for a private room is \$90 with an average of 24 reviews.

# In[ ]:


print('Entire Home/Apt Stats')
df_airbnb_rooms_entire[['price', 'number_of_reviews']].describe()


# The average price for a entire house or apartment is \$212 with an average of 22 reviews.

# In[ ]:


print('Shared Room Stats')
df_airbnb_rooms_shared[['price', 'number_of_reviews']].describe()


# The average price for a entire house or apartment is \$70 with an average of 16 reviews.

# In[ ]:


def color_producer(val):
    if val < 'Private room':
        return 'forestgreen'
    if val < 'Shared room':
        return 'darkorange'
    else:
        return 'darkred'
    
types_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)
    
for i in range(0, len(df_airbnb)):
    folium.Circle(
        location=[df_airbnb.iloc[i]['latitude'], df_airbnb.iloc[i]['longitude']],
        radius=25,
        color=color_producer(df_airbnb.iloc[i]['room_type'])
    ).add_to(types_map)
embed_map(types_map, 'types_map.html')


# It can be seen from the plot above that most of the room tyoes are either private or shared rooms. With most of them being in Manhattan, Brooklyn and Queens.

# **Listing Availablitity**

# In[ ]:


df_airbnb.availability_365.describe()


# The average listing is available for 112 days in a year. We will again plot the lower 25%, middle 50% and upper 25% on the map.

# In[ ]:


def color_producer(val):
    if val > 227:
        return 'forestgreen'
    if val > 0:
        return 'darkorange'
    else:
        return 'darkred'
    
availability_map = folium.Map(location=[40.730610,-73.935242], tiles='cartodbpositron', zoom_start=10)
    
for i in range(0, len(df_airbnb)):
    folium.Circle(
        location=[df_airbnb.iloc[i]['latitude'], df_airbnb.iloc[i]['longitude']],
        radius=25,
        color=color_producer(df_airbnb.iloc[i]['availability_365'])
    ).add_to(review_map)
embed_map(availability_map, 'availability_map.html')

