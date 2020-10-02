#!/usr/bin/env python
# coding: utf-8

# # Data visualization for Classification 
# In this kernel we are focusing on data preprocessing and data visualisation of New York City Airbnb Open Data
# Airbnb listings and metrics in NYC, NY, USA (2019)for Classification.
# 
# ### Data
# 
# Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019. This data file includes all needed information to find out more about hosts, geographical availability, necessary metrics to make predictions and draw conclusions.<br>
# 
# This data contains 16 columns, 47905 unique values(samples).  Imported all necessary files and libraries, We removed unnecessary data from the datset like last review, reviews per month and host name as they donot support the data required. We filled the null values with zero constant and did the visualization using seaborn, pyplot, matplotlib.<br>
# 
# #### Variables
# id: listing ID<br>
# name: name of the listing<br>
# host_id: host ID<br>
# host_name: name of the host<br>
# neighbourhood_group: location<br>
# neighbourhood: area<br>
# latitude: latitude coordinateslatitude: latitude coordinates<br>
# longitude: longitude coordinates<br>
# room_type: listing space type<br>
# price: price in dollars<br>
# minimum_nights: amount of nights minimum<br>
# number_of_reviews: number of reviews<br>
# last_review: latest review<br>
# reviews_per_month: number of reviews per month<br>
# calculated_host_listings_count: amount of listing per host<br>
# availability_365: number of days when listing is available for booking<br>
# link: __[NY_AIRBNB_DATASET](https://www.kaggle.com/dgomonov/new-york-city-airbnb-open-data)__
# ****

# In[ ]:


#import required
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


#read the file 'NYC_2019.csv' from the file
df = pd.read_csv('/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


#obtain information about the dataframe
df.info()


# In[ ]:


#view dataframe df
df


# In[ ]:


#find no of columns and no of rows
df.shape


# In[ ]:


#obtaining the description of the dataframe
df.describe()


# In[ ]:


#finding out if there are any null or empty values
df.isnull().sum()


# In[ ]:


#delete the row 'last_review'
del df['last_review']


# In[ ]:


#delete the row 'last_review'
del df['host_name']


# In[ ]:


df


# In[ ]:


#find if there are any null values in the dataset
df.isnull().sum()


# In[ ]:


#fill NaN data with 0 in the dataframe and display the data
df.fillna('0',inplace=True)
df


# In[ ]:


#remove the null values from the dataset
df=df[~(df['name']=='0')]
df


# In[ ]:


#categorize the neighbourhood group into categories
df.neighbourhood_group = df.neighbourhood_group.astype('category')


# In[ ]:


#print the categories in neighbourhood group
df.neighbourhood_group.cat.categories


# In[ ]:


#crosstab the columns neighbourhood group and room type
pd.crosstab(df.neighbourhood_group, df.room_type)


# In[ ]:


#catplot room type and price
sns.catplot(x="room_type", y="price", data=df);


# In[ ]:


#catplot neighbourhood_group and price
sns.catplot(x="neighbourhood_group", y="price", kind="boxen",
            data=df);


# In[ ]:


# create countplot roomtype and neighbourhood type
plt.figure(figsize=(10,10))
df1 = sns.countplot(df['room_type'],hue=df['neighbourhood_group'], palette='plasma')


# In[ ]:


#boxplot neighbourhood_group and room availability
plt.figure(figsize=(10,10))
df1 = sns.boxplot(data=df, x='neighbourhood_group',y='availability_365',palette='plasma')


# > # Observation
# 
# We can see that lack of giving no reviews effected the data. We removed unnecessary data from the dataset. Lot of apartments are available in manhattan compared to any other place and bronx has less apartments and more single rooms. Apartments cost way more than single rooms. Manhattan and brooklyn has costlier rooms and apartments. The availability of rooms is very less in manhattan and brooklyn and you can find room any day in bronx. There are lot of housing options in manhattan.
