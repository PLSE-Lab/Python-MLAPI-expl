#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt



# reading and cleaning data
data=pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')
data.isna().sum()
data.dropna()
ii=data.index[data.price==0]
data.drop(ii, inplace=True)# there are 11 records for which price is 0
data


# In[ ]:


sns.scatterplot(x=data.price,y=data.minimum_nights, hue=data.room_type);


# It seems like the data for price is highly skewed. 
# log transform helped with the skew 
# the figure suggests that there's distinction between the price ranges for room types 
# 

# In[ ]:


sns.scatterplot(x=np.log(data.price),y=np.log(data.minimum_nights), hue=data.room_type);


# Let's look at price with respect to neighbour hoods

# In[ ]:


plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Brooklyn'])


# In[ ]:


plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Manhattan'])


# In[ ]:


plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Queens'])


# In[ ]:


plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Staten Island'])


# In[ ]:


plt.figure(figsize=(11,10))
sns.scatterplot(y='price',x='room_type', data=data[data['neighbourhood_group']=='Bronx'])


# 1.  brooklyn:it has similar distributions for private room and entire home most of its value lie within 0-2000 and there are nearly same amout of homes in both the category
# 2. Manhattan:it has more entire home than private room and most of its value lie within 0-4000 and its highest amongst five boroughs for the particular category
# 3. Queens and staten island: both the borough's show similar distribution only difference is the price ranges latter one seems cheaper
# 4. Bronx: it's the cheapest amongs't all in all the three categories

# Distribution of number of host's in diffrent neighbourhoods for all the room_types'
# 
# one distinction is there seem's to be more private rooms in brooklyn similarly manhattan has more Entire homes/apt

# In[ ]:



nb_group=data.groupby('neighbourhood_group',as_index=False).agg({'host_id':'count'})
nb_group.columns=['neighbourhood_group','Count']
sns.catplot(x='neighbourhood_group', col='room_type', data=data, kind='count')


# Let's look like top 3 listing's by the host
# 1. moslty listed is an entire home/app in Manhattan area
# 2. distribution of  price seems to be the same 100-600 except kara has some highly priced appartments
# 
# Note: it seems like sonder's has 2 price ranges we will explore this further

# In[ ]:


#to get top 3 hoster's 
data.groupby(('host_name','host_id'),as_index=False).agg({'calculated_host_listings_count':'count'}).sort_values('calculated_host_listings_count',ascending=False)[0:3]  


# In[ ]:


plt.figure(figsize=(11,10))

sns.relplot(y='price',x='host_name',hue='room_type', data=data[data['host_id'].isin([219517861,107434423,30283594])])


# Distribution of days of the year on which listing was present 
# 
# All of them had listings (bookings) available for almost whole year

# In[ ]:


plt.figure(figsize=(11,10))

sns.scatterplot(y='availability_365',x='host_name',data=data[data['host_id'].isin([219517861,107434423,30283594])])


# sonder has 2 price range 100 to 300 and then 400 and above and most of them are available throughout the year

# In[ ]:


plt.figure(figsize=(11,10))

sns.relplot(x='availability_365',y='price',col='host_name', data=data[data['host_id'].isin([219517861,107434423,30283594])])

