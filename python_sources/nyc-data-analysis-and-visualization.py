#!/usr/bin/env python
# coding: utf-8

# 
# <h1>NY City Airbnb - Data Analysis and Visualization</h1>
# 
# This notebook focuses on NYC Airbnb data. I mainly studied the price by borough and neighbourhood and the type of room. I tried different Data Visualization techniques such as plot, maps and wordclouds

# <h2>Importing librairies and loading data</h2> 

# In[ ]:


#dealing with data
import numpy as np
import pandas as pd

#plot
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

#map
import geopandas as gpd
from shapely import wkt

#wordcloud
from wordcloud import WordCloud


# In[ ]:


airbnb=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')

airbnb.head(5)

There are 48,895 listings and 16 features. I will used the 8 columns : name, nighbourhood_group, neighbourhood, latitude, longitude, room_type, price and availability_365. 
# In[ ]:


airbnb.shape


# We can see that there are 4 variables with missing values. I could have treated it but I don't need it in my study. 

# In[ ]:


airbnb.isnull().sum()


# <h2>Quick analysis</h2>
# 
# First, I computed the quantile to see the range of values. The median price is 106\$ and 75% of the prices are under 175\$. Also, some prices are 0\$ which is not possible so I decided to delete them. Finally, we can say that most of the prices are between 50\$ and 200\$ and there are extreme values till 10,000\$. I chose to keep the listings with a price under 2000\$.
# 
# I also checked the number of listings by type of room : there are mainly private room and entire home.

# In[ ]:


airbnb['price'].quantile([0,.25, .5,0.75,1])


# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=airbnb,y='price')
plt.ylim(0,1000)


# In[ ]:


#Number of 0$
len(airbnb[airbnb.price==0.0])


# In[ ]:


#Number of 10,000$
len(airbnb[airbnb.price==10000.0])


# In[ ]:


len(airbnb[airbnb.price>2000.0])


# In[ ]:


#Remove 0$ and take only price under 2000$ by night
airbnb = airbnb[airbnb.price != 0.0]
airbnb = airbnb[airbnb.price <= 2000.0]


# In[ ]:


sns.countplot(airbnb['room_type'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Number of listings by type of room ')


# <h2>Price analysis by borough</h2>
# 
# - Number of listings by borough : we see that Manhattan and Brooklyn have more listings than Bronx or Staten Island. 
# - Price by borough : the boroughs with the highest prices are also those with the most listings (Manhattan and Brooklyn). On the violin plot, we can see that Manhattan has a bigger range of value and higher prices, instead of Bronx, Queens and Staten Island distributions are more left-skewed with a price around 70.

# In[ ]:


# Data for maps
nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
nyc.head(5)

# Count the number of listings by borough
borough_count = airbnb.groupby('neighbourhood_group').agg('count').reset_index()

#Rename the column to join the data 
nyc.rename(columns={'BoroName':'neighbourhood_group'}, inplace=True)
bc_geo = nyc.merge(borough_count, on='neighbourhood_group')

#Plot the count by borough into a map
fig,ax = plt.subplots(1,1, figsize=(10,10))
bc_geo.plot(column='id', cmap='viridis_r', alpha=.5, ax=ax, legend=True)
bc_geo.apply(lambda x: ax.annotate(s=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)
plt.title("Number of Airbnb Listings by NYC Borough")
plt.axis('off')


# In[ ]:


sns.countplot(airbnb['neighbourhood_group'], palette="plasma")
fig = plt.gcf()
fig.set_size_inches(10,10)
plt.title('Number of listings by borough ')


# In[ ]:


# Data group by neighbourhood_group (borough)
airbnb_neighbourhood_group = airbnb.groupby(['neighbourhood_group']) 


# In[ ]:


# Quartile by borough
airbnb_neighbourhood_group['price'].quantile([0,.25, .5,0.75,1]).to_frame()


# In[ ]:


viz_price_neighbourhood_group=sns.violinplot(data=airbnb[airbnb.price < 500], x='neighbourhood_group', y='price')
viz_price_neighbourhood_group.set_title('Density and distribution of prices for each borough')


# In[ ]:


#Compute average price by borough and join
borough_price = airbnb.groupby('neighbourhood_group').agg('median').reset_index()[['neighbourhood_group','price']]
bp_geo = nyc.merge(borough_price, on='neighbourhood_group')
bp_geo


# In[ ]:


fig,ax = plt.subplots(1,1, figsize=(10,10))
bp_geo.plot(column='price', cmap='plasma_r', alpha=.5, ax=ax, legend=True) #change cmap for colors
bp_geo.apply(lambda x: ax.annotate(s=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)
plt.title("Median price of Airbnb by NYC Borough")
plt.axis('off')


# <h2>Price analysis by neighbourhood</h2>
# 
# In this section, I wanted to see if each neighbourhood of each borough has the same median price or if there is disparity in a borough. The two barplots show the median prices of each neighbourhood in Manhattan and Bronx. I chose Manhattan and Bronx because there are spatially close but have global prices different. Actually, we see that some neighbourhoods of Bronx have the same price as some neighbourhood in Manhattan. Even, some of them (such as Riverdale) have a higher price than some neighbourhood of Manhattan (e.g Marble Hill). We find this idea on the map.
# Indeed, we can see that the North of Manhattan is very similar with the south of Bronx in terms of price. We've got the same idea between the part of Brooklyn wich is clode to the south of Manhattan or between the North of Brooklyn and the South of Queens. 
# So, the price depends more on the neighbourhood than the borough. Manhattan doesn't mean rich everywhere and Bronx poor. 

# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=airbnb[airbnb.neighbourhood_group == "Manhattan"],y='price')
plt.ylim(0,1000)


# In[ ]:


airbnb_manhattan = airbnb[airbnb.neighbourhood_group == "Manhattan"].groupby(['neighbourhood']) 
airbnb_med_manhattan = airbnb_manhattan['price'].median().to_frame()

airbnb_med_manhattan_sort = airbnb_med_manhattan.sort_values(by='price')
plt.figure(figsize=(40,30))
ax = sns.barplot(x=airbnb_med_manhattan_sort.index, y="price", data=airbnb_med_manhattan_sort)


# In[ ]:


plt.figure(figsize=(10,10))
ax = sns.boxplot(data=airbnb[airbnb.neighbourhood_group == "Bronx"],y='price')
plt.ylim(0,1000)


# In[ ]:


airbnb_bronx = airbnb[airbnb.neighbourhood_group == "Bronx"].groupby(['neighbourhood']) 
airbnb_bronx_med = airbnb_bronx['price'].median().to_frame()


airbnb_bronx_med_sort = airbnb_bronx_med.sort_values(by='price')
plt.figure(figsize=(40,30))
ax = sns.barplot(x=airbnb_bronx_med_sort.index, y="price", data=airbnb_bronx_med_sort)


# In[ ]:


airbnb.groupby(['neighbourhood'])['price'].median().to_frame().sort_values(by='price')


# In[ ]:


# Import data from the website (find it)
nbhoods = pd.read_csv('../input/nyntacsv/nynta.csv')
nbhoods.head(5)


# In[ ]:


#Rename the column
nbhoods.rename(columns={'NTAName':'neighbourhood'}, inplace=True)

#Convert the geometry column text into well known text (librairy shapely)
nbhoods['geom'] = nbhoods['the_geom'].apply(wkt.loads)

#Now convert the pandas dataframe into a Geopandas GeoDataFrame
nbhoods = gpd.GeoDataFrame(nbhoods, geometry='geom')


# In[ ]:


airbnb = gpd.GeoDataFrame(airbnb, geometry=gpd.points_from_xy(airbnb.longitude, airbnb.latitude))

# Spatial join (this code runs an intersect analysis to find which neighborhood the Airbnb location is in)
joined = gpd.sjoin(nbhoods, airbnb, how='inner', op='intersects')
joined.drop(columns='geom', inplace=True)
joined.rename(columns={'neighbourhood_left':'neighbourhood'}, inplace=True)
nb_join_price = joined.groupby('neighbourhood').agg('median').reset_index()[['neighbourhood','price']]
true_count = nbhoods.merge(nb_join_price, on='neighbourhood')


# In[ ]:


fig,ax = plt.subplots(1,1, figsize=(10,10))

base = nbhoods.plot(color='white', edgecolor='black', ax=ax)

true_count.plot(column='price',cmap='plasma_r', ax=base, legend=True)
plt.title('Median Price of listings by Neighborhood in NYC')


# <h2>Word Analysis</h2>
# 
# In the last part, I just show some wordclouds about the name of the listings. Globally, we see that words related to Manhattan and Brooklyn are often used which reflects the number of listings in those boroughs. Also, private room and some words describing entire home are frequent (same idea with the number of listings for this type of room). 
# 
# <h4> Manhattan </h4>
# 
# The second wordcloud focuses only on Manhattan. We see that the places around the location is important (East Village, Times Sqaure, Harlem...) and a good description of the room (Spacious, Charming, Modern, Large...). There are also Private room and apartement which are the most type of room in this area. Finally, there is the word "Luxury" which shows that Airbnb hosts want to highlight that Manhattan is richer.
# 
# <h4> Bronx </h4> 
# For this borough, we see that the Private Room is the most important type of room. In contrast to Manhattan, the hosts write "Bronx" instead of the places around the home. Also, there is "subway" which means that is important for a room in Bronx to have a subway nearby. 

# In[ ]:


# Global wordcloud
name = " ".join(str(w) for w in airbnb.name)
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080,max_words=60
                         ).generate(name)
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('name.png')
plt.show()


# In[ ]:


name_manhattan = " ".join(str(w) for w in airbnb.name[airbnb.neighbourhood_group == "Manhattan"])
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080,max_words=30
                         ).generate(name_manhattan)
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('name.png')
plt.show()


# In[ ]:


name_bronx = " ".join(str(w) for w in airbnb.name[airbnb.neighbourhood_group == "Bronx"])
plt.subplots(figsize=(25,15))
wordcloud = WordCloud(
                          background_color='white',
                          width=1920,
                          height=1080,max_words=30
                         ).generate(name_bronx)
plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('name.png')
plt.show()


# <h2> Sources </h2>
# 
# Those notebooks helped me to do my analysis : 
# 
# - [https://www.kaggle.com/geowiz34/maps-of-nyc-airbnbs-with-python](https://www.kaggle.com/geowiz34/maps-of-nyc-airbnbs-with-python)
# - [https://www.kaggle.com/chirag9073/airbnb-analysis-visualization-and-prediction](https://www.kaggle.com/chirag9073/airbnb-analysis-visualization-and-prediction)
