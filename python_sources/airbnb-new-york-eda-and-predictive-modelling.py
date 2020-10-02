#!/usr/bin/env python
# coding: utf-8

# # Let's work together!

# ### Exploratory Data Analysis on AirBnB New York Dataset

# ![NewYorkSkyline](http://richiewong.co.uk/wp-content/uploads/2019/12/New_York_Skyline-scaled.jpg)

# AirBnB has been a big disruptor in the hotel industry, as well as part of the gig eccnonomy, where it was founded in late 2008. AirBnB is an online market place for arranging or offering lodging, primarily homestays, or tourism experiences. It has scaled and grew fast as the company does not own any of the real estate listings.
# 
# It is very interesting to undertake analysis on this dataset and understaken more about New York and it's geography. Analysis includes learning about the most expensive neighbourhoods, what is the market price for renting an entire apartment or private room?
# 
# Feel free to fork for your own learning and edit the code or use in your own submissions. If you found this enriched your learning in the slightest please **upvote** this notebook as an encouragement for me to continue writing notebooks! :)
# 
# Thanks to @dgomonov for uploading the complete dataset.

# ## Motivation for EDA
# * Opportunity to work on complete ready dataset (~49,000 listings)
# * Find interesting trends - incl. finding the market value of the rent for appartment + shared room.
# * Explore AirBnB NYC dataset 
# * Undertake a problem as a travellers prespective
# 
# ## Opportunity to work using different libaries 
# * GeoPandas
# * Bokeh
# * Folium

# Import Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from shapely import wkt


# In[ ]:


# Reading the files
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Reading the Dataset

# In[ ]:


df=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# # Getting to know the dataset

# In[ ]:


#Make the data look cleaner, to 1.dp when finding out statistical analysis on the price
pd.set_option("display.precision", 1)


# In[ ]:


df.head()


# In[ ]:


#check the amount of rows and columns within the dataset
df.shape


# In[ ]:


#Checking for Null values in the dataset
print('Null values in Airbnb dataset: \n')
print(df.isnull().sum())
print('\n')
print('Percentage of null values in review columns: ')
print(round(df['last_review'].isnull().sum()/len(df)*100, 2),"%")


# In[ ]:


#Review the listings by boroname
plt.figure(figsize=(10,10))
sns.scatterplot(x='longitude', y='latitude', hue='neighbourhood_group',s=20, data=df, palette="coolwarm")


# In[ ]:


#let's proceed examine the categorical unique values

#examining the unique values of group
print('There are X number of unique neighbourhoods:')
print(len(df.neighbourhood_group.unique()))
print('\n')
print(df.neighbourhood_group.unique())


# In[ ]:


print('There are X number of unique neighbourhoods:')
print(len(df.neighbourhood.unique()))
# print('\n')
# print(df.neighbourhood.unique())


# In[ ]:


print('There are X number of unique room types:')
print(len(df.room_type.unique()))
print('\n')
print(df.room_type.unique())


# # A look at Listings by New York Neighbourhoods 

# Lets first plot a map of the number of listings on AirBnB via a map to get an overview of the picture.

# In[ ]:


#Here we are retrieve the NYC boroughs from Geopandas
nyc = gpd.read_file(gpd.datasets.get_path('nybb'))
nyc.head(5)


# In[ ]:


#Get a count by borough
borough_count = df.groupby('neighbourhood_group').agg('count').reset_index()

#Rename the column to boroname, so that we can join the data to it on a common field
nyc.rename(columns={'BoroName':'neighbourhood_group'}, inplace=True)
nyc_geo = nyc.merge(borough_count, on='neighbourhood_group')


# In[ ]:


#Plot the count by borough into a map
fig,ax = plt.subplots(1,1, figsize=(20,10))
nyc_geo.plot(column='id', cmap='coolwarm', alpha=.5, ax=ax, legend=True)
nyc_geo.apply(lambda x: ax.annotate(s=x.neighbourhood_group, color='black', xy=x.geometry.centroid.coords[0],ha='center'), axis=1)
plt.title("Number of Airbnb Listings by NYC Borough")
plt.axis('off')

#Thanks to @geowiz34 https://www.kaggle.com/geowiz34 for this plot.


# Great! We can see from the plot that Manhattan has the most number of listings.

# # Interactive Map with Folium

# Using Folium requires two key inputs, the location (lat and long) and the zoom.
# 
# The lat and long can be found here by typing in
# 
# 
# https://www.latlong.net/

# Interactive Map of the Listings

# In[ ]:


import folium
from folium.plugins import HeatMap
m=folium.Map([40.7128,-74.0060],zoom_start=11)
HeatMap(df[['latitude','longitude']].dropna(),radius=8,gradient={0.2:'blue',0.4:'purple',0.6:'orange',1.0:'red'}).add_to(m)
display(m)


# Clustering of the listings

# In[ ]:


from folium.plugins import FastMarkerCluster

Lat=40.80
Long=-73.80

locations = list(zip(df.latitude, df.longitude))

map1 = folium.Map(location=[Lat,Long], zoom_start=11)
FastMarkerCluster(data=locations).add_to(map1)
map1


# # Neighbourhoods via types of room
# 
# Now, let's see if we can break it down by the types of rooms per neighbourhood groups.

# In[ ]:


plt.figure(figsize=(15,6))
sns.countplot(data=df, x='neighbourhood_group', hue='room_type', palette='coolwarm')
plt.title('Counts of AirBnB in neighbourhood group with Room Type Category', fontsize=15)
plt.xlabel('Neighbourhood group')
plt.ylabel("Count")
plt.legend(frameon=False, fontsize=12)


# Interestingly, private rooms are listed the most in all New York Neighbourhood groups except Manhattan on AirBnB.

# # Statistical Analaysis on Neighbourhood groups
# Let's do a quick statistical analysis of the price in relationship with the neighbourhood group

# In[ ]:


df.pivot_table(index=['neighbourhood_group'],
               values='price',
               aggfunc=['count', 'mean','median', 'std'])


# In[ ]:


#Alternative to than using pivot tables
df.groupby('neighbourhood_group').count()['price'].nlargest(n=20, keep='all')


# We can see that Manhattan 
# * has the highest mean and median price
# * highest standard deviation (biggest spread of data)

# # Price Distribution - Finding the market value?

# A distribution of the types of neighbourhood below $400 per night.

# In[ ]:


plt.figure(figsize=(15,6))
sns.violinplot(data=df[df.price <400], x='neighbourhood_group', y='price', palette='coolwarm')
plt.title('Density and distribution of prices - ALL TYPES OF ROOMS', fontsize=15)
plt.xlabel('Neighbourhood group')
plt.ylabel("Price")


# It seems that the most price range below $100 per night, it would be interesting to understand if there are any add-on towards the prices. More importantly a filter, to see how the price differ if it's whole appartment or private room.

# ### Let's see price split via 'Entire home/apt

# In[ ]:


dfa = df[df.room_type== 'Entire home/apt'] # panda chain rule
print(dfa.shape)


# In[ ]:


plt.figure(figsize=(15,6))
sns.violinplot(data=dfa[df.price <400], x='neighbourhood_group', y='price', palette='coolwarm')
plt.title('Density and distribution of prices - ENITRE APP', fontsize=15)
plt.xlabel('Neighbourhood group')
plt.ylabel("Price")


# We can see that for the majority to rent entire appartment it can cost well above $100 per night. 

# ### Let's see price split via Private Room

# In[ ]:


dfp = df[df.room_type== 'Private room'] # panda chain rule

plt.figure(figsize=(15,6))
sns.violinplot(data=dfp[df.price <400], x='neighbourhood_group', y='price', palette='coolwarm')
plt.title('Density and distribution of prices - PRIVATE ROOM', fontsize=15)
plt.xlabel('Neighbourhood group')
plt.ylabel("Price")


# From breaking the distribution to different room types, we can see the price range and the distrbutions.
# 
# This is useful to see what the market price is for a host or a someone renting. As someone renting you want to get a good value, price and someone hosting, you don't want to be charging much higher than the competition.

# # Top neighbourhood listings by average price

# In[ ]:


#Top 10 neighbourhoods
df.groupby('neighbourhood').mean()['price'].nlargest(n=10, keep='all')


# # Word Cloud

# In[ ]:


#word cloud visualisation to show the popular neighbourhoods

from wordcloud import WordCloud

plt.subplots(figsize=(20,15))
wordcloud = WordCloud(
                          width=1920,
                          height=1080
                         ).generate(" ".join(df.neighbourhood))
plt.imshow(wordcloud)
plt.title('Word Cloud for Neighbourhoods')
plt.axis('off')
plt.show()


# # Prediciting Listing Prices

# Interesting... small correlation for reviews per month and avalaibility and calculated host listing and avalaibility
# 
# The obvious correlations are number of reviews and reviews per month. 

# In[ ]:


airbnb=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


airbnb.drop(['name','id','host_name','last_review'], axis=1, inplace=True)


# In[ ]:


airbnb.drop(['host_id','latitude','longitude','neighbourhood','number_of_reviews','reviews_per_month'], axis=1, inplace=True)
#examing the changes
airbnb.head(5)


# In[ ]:


#Encode the input Variables
def Encode(airbnb):
    for column in airbnb.columns[airbnb.columns.isin(['neighbourhood_group', 'room_type'])]:
        airbnb[column] = airbnb[column].factorize()[0]
    return airbnb

airbnb_en = Encode(airbnb.copy())


# In[ ]:


airbnb_en.head(15)


# In[ ]:


#Get Correlation between different variables
corr = airbnb_en.corr(method='kendall')
plt.figure(figsize=(14,9))
sns.heatmap(corr, annot=True)
airbnb_en.columns


# In[ ]:


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import r2_score


# In[ ]:


#Defining the independent variables and dependent variables
x = airbnb_en.iloc[:,[0,1,3,4,5]]
y = airbnb_en['price']
#Getting Test and Training Set
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=353)
x_train.head()
y_train.head()


# In[ ]:


x_train.shape


# In[ ]:


#Prepare a Linear Regression Model
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred=reg.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_pred)


# In[ ]:


#Prepairng a Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.1,random_state=105)
DTree=DecisionTreeRegressor(min_samples_leaf=.0001)
DTree.fit(x_train,y_train)
y_predict=DTree.predict(x_test)
from sklearn.metrics import r2_score
r2_score(y_test,y_predict)


# # Summary
# 
# We learn that there are a lot of listings in Manhatten, the price range can really vary too! We also found a feeling of what the market value of the rent for appartment + shared room.
# 
# 
# ## More to come
# 
# In the near future I aim to do more interactive features in the notebook using Folium library, and have provide more in detail analysis on the indiviual neighbourhoods and the relationship with the tourist attractions, underground stations - if can find ready data avalaible.
# 
# ### To do Analysis
# 
# These are the things - intend to undertake
# * Interactive chart using - chart from geopandas
# * Size of the borough (area) - geopandas
# 
# ### Additional Spatial Analysis
# 
# * Add in features by underground stations
# * Add in features by main tourist attractions! - Madison Square Garden, Times Square, Rockafella, Empire state building etc
# * Problem by want to be proximity of certain criteria - (distance) - inspired by geopandas distance
