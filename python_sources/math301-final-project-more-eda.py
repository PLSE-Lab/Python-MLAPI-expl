#!/usr/bin/env python
# coding: utf-8

# ### Import Libraries, Read in Data and Merge into one big dataset with year indicator variable

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely import wkt
from collections import Counter
plt.style.use('fivethirtyeight')


# In[ ]:


# Read in wrangled NYC airbnb data from 2016-2020
df16 = pd.read_csv("../input/math301-final-project-data/airbnb16.csv", index_col=0)
df17 = pd.read_csv("../input/math301-final-project-data/airbnb17.csv", index_col=0)
df18 = pd.read_csv("../input/math301-final-project-data/airbnb18.csv", index_col=0)
df19 = pd.read_csv("../input/math301-final-project-data/airbnb19.csv", index_col=0)
df20 = pd.read_csv("../input/math301-final-project-data/airbnb20.csv",  index_col=0)

# Read in NYC Shape/Geometry data
nbhoods = pd.read_csv('../input/nyctractshape/nynta.csv')


# In[ ]:


# Add year variable to all the data sources
for dataf,year in zip([df16,df17,df18,df19,df20], ['16','17','18','19','20']):
    dataf['year'] = '20' + year


# In[ ]:


# Merge data sources into one big dataset
df = pd.concat([df16,df17,df18,df19,df20])


# ### Outliers - Which listings and where are they? (e.g. luxurious penthouses)

# ##### 2017

# In[ ]:


df17.price.describe()


# In[ ]:


# Upper limit for outlier detection (Q3 + IQR * 1.5) => Anything beyond this value will be considered an outlier for year of 2017
165 + 1.5 * 100


# In[ ]:


df17[df17.price >=315].neighbourhood_group.value_counts() * 100 / len(df17[df17.price >=315])


# In[ ]:


(df17[df17.price >=315].neighbourhood.value_counts() * 100 / len(df17[df17.price >=315]))[:20]


# ##### 2020

# In[ ]:


df20.price.describe()


# In[ ]:


# Upper limit for outlier detection (Q3 + IQR * 1.5) => Anything beyond this value will be considered an outlier for year of 2020
180 + 1.5 * 110


# In[ ]:


df20[df20.price >=345].neighbourhood_group.value_counts() * 100 / len(df20[df20.price >=345])


# In[ ]:


(df20[df20.price >=345].neighbourhood.value_counts() * 100 / len(df20[df20.price >=345]))[:20]


# ### Number of Listings over time on a tract level (more granular than borough level)

# In[ ]:


# rename the "NTAName" column to "neighborhood" to make the column seem more straightforward to understand
nbhoods.rename(columns={'NTAName':'neighbourhood'}, inplace=True)

# Convert the geometry column text into well known text, allowing me to plot geometry accurately
nbhoods['geom'] = nbhoods['the_geom'].apply(wkt.loads)

# Convert dataframe to "Geopandas GeoDataFrame"
nbhoods = gpd.GeoDataFrame(nbhoods, geometry='geom')


# In[ ]:


# Match each Airbnb location, and enable the "data" dataframe into a geopandas dataframe
df17_geo = gpd.GeoDataFrame(df17, geometry=gpd.points_from_xy(df17.longitude, df17.latitude))

# Spatial join
joined17 = gpd.sjoin(nbhoods, df17_geo, how='inner', op='intersects')


# In[ ]:


# Drop the second geometry column
joined17.drop(columns='geom', inplace=True)

# Rename the neighbourhood_left column to neighbourhood
joined17.rename(columns={'neighbourhood_left':'neighbourhood'}, inplace=True)

# Create a count of listings in each neighborhood (tract)
nb_join_count17 = joined17.groupby('neighbourhood').agg('count').reset_index()

# Attach count of listings onto nbhood map
listings_count17 = nbhoods.merge(nb_join_count17, on='neighbourhood')

# Make an index column for colorbar later
listings_count17['index'] = range(0,len(listings_count17))


# In[ ]:


# Plot 2017 Number of Listings Map Visualization

fig,ax = plt.subplots(1,1, figsize=(10,10))

base = nbhoods.plot(color='white', edgecolor='black', ax=ax)

listings_count17.plot(column='index',cmap='plasma_r', ax=base, legend=True)

plt.title('Number of Airbnb listings by tract in NYC in 2017')


# ##### 2020

# In[ ]:


# Match each Airbnb location, and enable the "data" dataframe into a geopandas dataframe
df20_geo = gpd.GeoDataFrame(df20, geometry=gpd.points_from_xy(df20.longitude, df20.latitude))

# Spatial join
joined20 = gpd.sjoin(nbhoods, df20_geo, how='inner', op='intersects')


# In[ ]:


# Drop the second geometry column
joined20.drop(columns='geom', inplace=True)

# Rename the neighbourhood_left column to neighbourhood
joined20.rename(columns={'neighbourhood_left':'neighbourhood'}, inplace=True)

# Create a count of listings in each neighborhood (tract)
nb_join_count20 = joined20.groupby('neighbourhood').agg('count').reset_index()

# Attach count of listings onto nbhood map
listings_count20 = nbhoods.merge(nb_join_count20, on='neighbourhood')

# Make an index column for colorbar later
listings_count20['index'] = range(0,len(listings_count20))


# In[ ]:


# Plot 2020 Number of Listings Map Visualization

fig,ax = plt.subplots(1,1, figsize=(10,10))

base = nbhoods.plot(color='white', edgecolor='black', ax=ax)

listings_count20.plot(column='index',cmap='plasma_r', ax=base, legend=True)

plt.title('Number of Airbnb listings by tract in NYC in 2020')


# ### Minimum Revenue = price x minimum number of nights x Demand (proxy: Number of Reviews)

# In[ ]:


# Create variable for Minimum Revenue for each listing 
df['revenue'] = df['price'] * df['minimum_nights'] * df['number_of_reviews']


# In[ ]:


df.groupby(['neighbourhood_group','year']).revenue.agg({'sum','mean','median'})


# In[ ]:


df17['revenue'] = df17['price'] * df17['minimum_nights'] * df17['number_of_reviews']
df20['revenue'] = df20['price'] * df20['minimum_nights'] * df20['number_of_reviews']


# In[ ]:


df17.groupby(['neighbourhood']).revenue.agg({'sum','mean','median'}).sort_values('median', ascending=False)[:20]


# In[ ]:


df20.groupby(['neighbourhood']).revenue.agg({'sum','mean','median'}).sort_values('median', ascending=False)[:20]


# In[ ]:


# List of top 20 tracts with highest median min revenues in 2017 and 2020

top17_rev_2017 = df17.groupby(['neighbourhood']).revenue.agg({'sum','mean','median'}).sort_values('median', ascending=False)[:20].index.tolist()
top20_rev_2020 = df20.groupby(['neighbourhood']).revenue.agg({'sum','mean','median'}).sort_values('median', ascending=False)[:20].index.tolist()


# In[ ]:


top17_rev_2017_neighb = []
for tract in top17_rev_2017:
    top17_rev_2017_neighb.append(list(set(df[df.neighbourhood == tract].neighbourhood_group))[0])


# In[ ]:


pd.DataFrame([top17_rev_2017,top17_rev_2017_neighb]).T


# In[ ]:


Counter(top17_rev_2017_neighb)


# In[ ]:


top20_rev_2020_neighb = []
for tract in top20_rev_2020:
    top20_rev_2020_neighb.append(list(set(df[df.neighbourhood == tract].neighbourhood_group))[0])


# In[ ]:


pd.DataFrame([top20_rev_2020,top20_rev_2020_neighb]).T


# In[ ]:


Counter(top20_rev_2020_neighb)

