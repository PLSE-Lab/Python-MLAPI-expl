#!/usr/bin/env python
# coding: utf-8

# ![](https://miro.medium.com/max/1500/1*8Zcspj5yuoU5jsMv1xL7cg.png)

# Airbnb is an online marketplace for arranging or offering lodging, primarily homestays, or tourism experiences. The company does not own any of the real estate listings, nor does it host events; it acts as a broker, receiving commissions from each booking.The company is based in San Francisco, California, United States.
# 
# The company was conceived after its founders put an air mattress in their living room, effectively turning their apartment into a bed and breakfast, in order to offset the high cost of rent in San Francisco; Airbnb is a shortened version of its original name, AirBedandBreakfast.com.

# # Importing the Libraries

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt


# # Reading the dataset

# In[ ]:


df = pd.read_csv("../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")
df.head()


# # Checking for Null values in the dataset

# In[ ]:


df.isnull().sum()


# ## Summary of the dataset

# In[ ]:


df.describe()


# In[ ]:


df.info()


# # Data Preprocessing

# - Replacing null values in the column **reviews_per_month** with 0 in the dataset

# In[ ]:


df['reviews_per_month'].fillna(0,inplace = True)


# Replacing null values in the column **name** with the character **$** and **hostname** with character **#** in the dataset. Both name and hostname are not the main aspects in our analysis, that's why we have replaced them by some special characters.

# In[ ]:


df['name'].fillna("$",inplace=True)
df['host_name'].fillna("#",inplace=True)


# Dropping the column **last review** as more than 10,000 data points contains null values.

# In[ ]:


df.drop(['last_review'],axis=1,inplace=True)


# In[ ]:


df.head()


# # Unique neighbourhoods

# These are the different unique neighbourhoods in the 4 different neighbourhood groups including Brooklyn, Manhattan, Queens, Staten Islands, Bronx.

# In[ ]:


df.neighbourhood.unique()


# # Univariate Analysis

# ## Different Neighbourhood groups

# The following plot represents the count of Airbnb's in the different neighbourhood groups. From the plot, we can easily visualize that maximum number of houses or apartments listed on Airbnb is in 

# In[ ]:


f,ax = plt.subplots(figsize=(15,6))
ax = sns.countplot(df.neighbourhood_group,palette="muted")
plt.show()


# ## Price Distribution of Airbnb in Brooklyn

# The price distribution of Airbnb in Brooklyn averages around 70-500 dollars per night depending upon the neighbourhood.

# In[ ]:


df1 = df[df.neighbourhood_group == "Brooklyn"][["neighbourhood","price"]]
d = df1.groupby("neighbourhood").mean()
sns.distplot(d)
plt.show()


# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = df[df.neighbourhood_group=="Brooklyn"]['price']
sns.distplot(df1)
plt.show()


# ## Price Distribution of Airbnb in Manhattan

# The price distribution of Airbnb in Manhattan averages around 80-490 dollars per night depending upon the neighbourhood.

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = df[df.neighbourhood_group=="Manhattan"]['price']
sns.distplot(df1)
plt.show()


# ## Price Distribution of Airbnb in Queens

# The price distribution of Airbnb in Queens averages around 60-280 dollars per night depending upon the neighbourhood.

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = df[df.neighbourhood_group=="Queens"]['price']
sns.distplot(df1)
plt.show()


# ## Price Distribution of Airbnb in Staten Island

# The price distribution of Airbnb in Staten Islands averages around 50-800 dollars per night depending upon the neighbourhood.

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = df[df.neighbourhood_group=="Staten Island"]['price']
sns.distplot(df1)
plt.show()


# ## Price Distribution of Airbnb in Bronx

# The price distribution of Airbnb in Bronx averages around 50-450 dollars per night depending upon the neighbourhood.

# In[ ]:


f,ax = plt.subplots(figsize=(15,4))
df1 = df[df.neighbourhood_group=="Bronx"]['price']
sns.distplot(df1)
plt.show()


# ## Number of different room types

# The maximum number of rooms listed on Airbnb are private rooms and entire home and apartments and a very small number of shared rooms are listed on Airbnb.

# In[ ]:


f,ax = plt.subplots(figsize=(12,5))
ax = sns.countplot(df.room_type,palette="muted")
plt.show()


# ## Price Distribution of Private rooms

# Private rooms on average are prised from 60-120 dollars per night on an average depending upon the neghbourhood group it is loacted.

# In[ ]:


df1 = df[df.room_type == "Private room"][["neighbourhood_group","price"]]
d = df1.groupby("neighbourhood_group").mean()
sns.distplot(d)
plt.show()


# In[ ]:


df1 = df[df.room_type=='Private room']['price']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df1)
plt.show()


# ## Price Distribution of Shared rooms

# Most of the shared rooms have the price range between 50-70 dollars per night depending upon the neighbourhood groups.

# In[ ]:


df1 = df[df.room_type=='Shared room']['price']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df1)
plt.show()


# ## Price Distribution of Entire home/apt

# The average price of entire home or apartment varies from 120-250 dollars per night depending upon the neighbourhood they given house is situated. 

# In[ ]:


df1 = df[df.room_type=='Entire home/apt']['price']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df1)
plt.show()


# ## Distribution of Reviews per month

# Most of the houses listed on Airbnb has an average of around 1-10 reviews a month and this number may vary sometimes even upto 50.

# In[ ]:


f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df.reviews_per_month)
plt.show()


# ## Distribution of Availability of rooms

# The availability of rooms in different neighbourhood groups and ranges from 0-360.

# In[ ]:


f,ax = plt.subplots(figsize=(15,5))
ax = sns.distplot(df.availability_365)
plt.show()


# ## Minimum nights people stay in different room types

# In the private roooms people mostly stay for around 1-7 days depending upon the neighbourhood groups.

# In[ ]:


df1 = df[df.room_type=="Private room"]['minimum_nights']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.swarmplot(y= df1.index,x= df1.values)
plt.xlabel("minimum_nights")
plt.show()


# Mostly travellers, backpackers and people on low budget like to stay in the shared rooms. They live on an average of 1-2 days as they keep on moving from one place to another.

# In[ ]:


df1 = df[df.room_type=="Shared room"]['minimum_nights']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.swarmplot(y= df1.index,x= df1.values)
plt.xlabel("minimum_nights")
plt.show()


# People love to stay in the entire home or apartments as there are least restrictions when they travel with the family or friends. They can prepare their own meals if they want in these apartments. On an average people live in these apartments from 1-90 days.

# In[ ]:


df1 = df[df.room_type=="Entire home/apt"]['minimum_nights']
f,ax = plt.subplots(figsize=(15,5))
ax = sns.swarmplot(y= df1.index,x= df1.values)
plt.xlabel("minimum_nights")
plt.show()


# # Bivariate Analysis

# ## Longitude vs Latitude (representing different neighbourhood groups) 

# In[ ]:


f,ax = plt.subplots(figsize=(16,8))
ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.neighbourhood_group,palette="coolwarm")
plt.show()


# ## Longitude vs Latitude (representing availability of rooms)

# The the given plot we can visualize the number of rooms available in different neighbourhood groups.

# In[ ]:


f,ax = plt.subplots(figsize=(16,8))
ax = sns.scatterplot(y=df.latitude,x=df.longitude,hue=df.availability_365,palette="coolwarm")
plt.show()


# ## Top 10 most popular Airbnb hosts 

# This is the list of top 10 most popular host in the given neighbourhood groups. Maximum number of people love to stay at their place. The reason behind their popularity may depend upon the price, neighbourhood, cleanliness and many more.

# In[ ]:


df1 = df.host_id.value_counts()[:11]
f,ax = plt.subplots(figsize=(16,5))
ax = sns.barplot(x = df1.index,y=df1.values,palette="muted")
plt.show()

