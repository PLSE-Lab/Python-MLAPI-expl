#!/usr/bin/env python
# coding: utf-8

# #  overview of the data
# ### Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique,
# ### personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019.
# 
# ### This dataset has around 49,000 observations in it with 16 columns and it is a mix between categorical and numeric values.

# In[ ]:


#importing necessery libraries for future analysis of the dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


airbnb = pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


airbnb.head(10)


# In[ ]:


#checking amount of rows in given dataset to understand the size we are working with
len(airbnb)


# In[ ]:


#checking type of every column in the dataset
airbnb.dtypes


# In[ ]:


# Checking for Null values in the dataset

airbnb.isnull().sum()


# In[ ]:


# Summary of the dataset

airbnb.describe()


# In[ ]:


airbnb.info()


# # Data Preprocessing

# In[ ]:


# Replacing null values in the column reviews_per_month with 0 in the dataset

airbnb['reviews_per_month'].fillna(0,inplace = True)


# In[ ]:


#dropping columns that are not significant or could be unethical to use for our future data exploration and predictions
airbnb.drop(['id','host_name','last_review'], axis=1, inplace=True)
#examing the changes
airbnb.head(5)


# # Unique neighbourhoods

# In[ ]:


# These are the different unique neighbourhoods in the 4 different neighbourhood groups
# including Brooklyn, Manhattan, Queens, Staten Islands, Bronx.

airbnb.neighbourhood.unique()


# In[ ]:


#examining the unique values of room_type as this column will appear very handy for later analysis
airbnb.room_type.unique()


# In[ ]:


airbnb.room_type.value_counts()


# In[ ]:


airbnb.neighbourhood.value_counts()


# In[ ]:


#let's see what hosts (IDs) have the most listings on Airbnb platform and taking advantage of this service
top_host=airbnb.host_id.value_counts().head(10)
top_host


# In[ ]:


# From the plot, we can easily visualize that maximum number of houses or apartments listed on Airbnb 
f,ax = plt.subplots(figsize=(15,6)
ax = sns.countplot(airbnb.neighbourhood_group,palette="muted")
plt.show()


# # Price Distribution of Airbnb in Brooklyn

# In[ ]:


airbnb.columns


# In[ ]:


airbnb['neighbourhood_group'].value_counts()


# In[ ]:


df1 = airbnb[airbnb.neighbourhood_group == "Brooklyn"][["neighbourhood","price"]]
d = df1.groupby("neighbourhood").mean()
sns.distplot(d)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = airbnb[airbnb.neighbourhood_group=="Brooklyn"]['price']
sns.distplot(df1)
plt.show()


# # Price Distribution of Airbnb in Manhattan

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = airbnb[airbnb.neighbourhood_group=="Manhattan"]['price']
sns.distplot(df1)
plt.show()


# # Price Distribution of Airbnb in Queens

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = airbnb[airbnb.neighbourhood_group=="Queens"]['price']
sns.distplot(df1)
plt.show()


# # Price Distribution of Airbnb in Staten Island

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 =airbnb[airbnb.neighbourhood_group=="Staten Island"]['price']
sns.distplot(df1)
plt.show()


# # Price Distribution of Airbnb in Bronx

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = airbnb[airbnb.neighbourhood_group=="Bronx"]['price']
sns.distplot(df1)
plt.show()


# # Number of different room types
# The maximum number of rooms listed on Airbnb are private rooms and entire home and apartments and a very small number of shared rooms are listed on Airbnb.

# In[ ]:


f,ax = plt.subplots(figsize=(12,5))
ax = sns.countplot(airbnb.room_type,palette="muted")
plt.show()


# # Price Distribution of Private rooms
# Private rooms on average are prised from 60-120 dollars per night on an average depending upon the neghbourhood group it is loacted.

# In[ ]:


df1 = airbnb[airbnb.room_type == "Private room"][["neighbourhood_group","price"]]
d = df1.groupby("neighbourhood_group").mean()
sns.distplot(d)
plt.show()


# In[ ]:


df1 = airbnb[airbnb.room_type=='Private room']['price']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df1)
plt.show()


# # Price Distribution of Shared rooms
# Most of the shared rooms have the price range between 50-70 dollars per night depending upon the neighbourhood groups.

# In[ ]:


df1 = airbnb[airbnb.room_type=='Shared room']['price']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df1)
plt.show()


# # Price Distribution of Entire home/apt
# The average price of entire home or apartment varies from 120-250 dollars per night depending upon the neighbourhood they given house is situated.

# In[ ]:


df1 = airbnb[airbnb.room_type=='Entire home/apt']['price']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df1)
plt.show()


# # Distribution of Reviews per month
# Most of the houses listed on Airbnb has an average of around 1-10 reviews a month and this number may vary sometimes even upto 50.

# In[ ]:


f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(airbnb.reviews_per_month)
plt.show()


# # Distribution of Availability of rooms
# The availability of rooms in different neighbourhood groups and ranges from 0-360.

# In[ ]:


f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(airbnb.availability_365)
plt.show()


# # Minimum nights people stay in different room types
# In the private roooms people mostly stay for around 1-7 days depending upon the neighbourhood groups.

# In[ ]:


df1 = airbnb[airbnb.room_type=="Private room"]['minimum_nights']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.swarmplot(y= df1.index,x= df1.values)
plt.xlabel("minimum_nights")
plt.show()


# In[ ]:


df1 = airbnb[airbnb.room_type=="Shared room"]['minimum_nights']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.swarmplot(y= df1.index,x= df1.values)
plt.xlabel("minimum_nights")
plt.show()


# In[ ]:


df1 = airbnb[airbnb.room_type=="Entire home/apt"]['minimum_nights']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.swarmplot(y= df1.index,x= df1.values)
plt.xlabel("minimum_nights")
plt.show()


# # Bivariate Analysis
# Longitude vs Latitude (representing different neighbourhood groups)

# In[ ]:


f,ax = plt.subplots(figsize=(16,8))
ax = sns.scatterplot(y=airbnb.latitude,x=airbnb.longitude,hue=airbnb.neighbourhood_group,palette="coolwarm")
plt.show()


# # Longitude vs Latitude (representing availability of rooms)
# The the given plot we can visualize the number of rooms available in different neighbourhood groups.

# In[ ]:


f,ax = plt.subplots(figsize=(16,8))
ax = sns.scatterplot(y=airbnb.latitude,x=airbnb.longitude,hue=airbnb.availability_365,palette="coolwarm")
plt.show()


# # Top 10 most popular Airbnb hosts
# This is the list of top 10 most popular host in the given neighbourhood groups. Maximum number of people love to stay at their place. The reason behind their popularity may depend upon the price, neighbourhood, cleanliness and many more.

# In[ ]:


df1 = airbnb.host_id.value_counts()[:11]
f,ax = plt.subplots(figsize=(16,5))
ax = sns.barplot(x = df1.index,y=df1.values,palette="muted")
plt.show()


# In[ ]:




