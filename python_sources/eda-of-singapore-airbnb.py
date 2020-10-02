#!/usr/bin/env python
# coding: utf-8

# # [Airbnb](https://www.airbnb.com/)

# * Room Type
# Airbnb hosts can list entire homes/apartments, private or shared rooms.
# 
# Depending on the room type, availability, and activity, an airbnb listing could be more like a hotel, disruptive for neighbours, taking away housing, and illegal.
# 
# * Activity
# Airbnb guests may leave a review after their stay, and these can be used as an indicator of airbnb activity.
# 
# The minimum stay, price and number of reviews have been used to estimate the occupancy rate, the number of nights per year and the income per month for each listing.
# 
# How does the income from Airbnb compare to a long-term lease?
# 
# Do the number of nights booked per year make it impossible for a listing to be used for residential housing?
# 
# And what is renting to a tourist full-time rather than a resident doing to our neighbourhoods and cities?
# * Availability
# An Airbnb host can setup a calendar for their listing so that it is only available for a few days or weeks a year.
# 
# Other listings are available all year round (except for when it is already booked).
# 
# Entire homes or apartments highly available year-round for tourists, probably don't have the owner present, could be illegal, and more importantly, are displacing residents.
# * Listings per Host
# Some Airbnb hosts have multiple listings.
# 
# A host may list separate rooms in the same apartment, or multiple apartments or homes available in their entirity.
# 
# Hosts with multiple listings are more likely to be running a business, are unlikely to be living in the property, and in violation of most short term rental laws designed to protect residential housing.

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py
import plotly.express as px
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True) 

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/singapore-airbnb/listings.csv')


# # Reduce Memory Usage

# In[ ]:


## Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


df = reduce_mem_usage(df)


# * Finding the values in the dataframe by using head()

# In[ ]:


df.head()


# **Creating the descriptive statistics of the data**

# In[ ]:


df.describe()


# **Checking information about the structure of dataframe**

# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df_null = df.loc[:, df.isnull().any()].isnull().sum().sort_values(ascending=False)
print(df_null)


# # Checking the correlation between the predictors

# In[ ]:


correlation = df.corr(method='pearson')


# In[ ]:


correlation


# # Creating Correlation Heatmap

# In[ ]:


f,ax = plt.subplots(figsize=(12,10))
sns.heatmap(df.iloc[:,2:].corr(),annot=True, linewidths=.1, fmt='.1f', ax=ax,cmap="YlGnBu")

plt.show()


# In[ ]:


df.isna().sum()


# # Exploratory Data Analysis

# * Creating Pairplot

# In[ ]:


sns.pairplot(df)
plt.show()


# # Relation between Price and Room_type

# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x=df['room_type'], y=df['price'])
plt.xticks(rotation= 45)
plt.xlabel('room_type')
plt.ylabel('Price')
plt.title('Price vs Room Type')


# # Relationship between Price and  neighbourhood_group

# In[ ]:


plt.figure(figsize=(25,10))
sns.barplot(x=df['neighbourhood_group'], y=df['price'])
plt.xlabel('neighbourhood_group')
plt.ylabel('Price')
plt.title('Price vs neighbourhood_group')


# # Relationship between Price and  neighbourhood

# In[ ]:


plt.figure(figsize=(25,10))
sns.barplot(x=df['neighbourhood'], y=df['price'])
plt.xticks(rotation= 90)
plt.xlabel('neighbourhood')
plt.ylabel('Price')
plt.title('Price vs neighbourhood')


# # Relationship between Price and  availability_365

# In[ ]:


f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='availability_365',y='price',data=df,color='lime',alpha=0.8)
plt.xlabel('availability_365',fontsize = 15,color='blue')
plt.ylabel('price',fontsize = 15,color='blue')
plt.title('availability_365  VS  price',fontsize = 20,color='blue')
plt.grid()


# # Relationship between minimum_nights and Price

# In[ ]:


f,ax1 = plt.subplots(figsize =(20,10))
sns.pointplot(x='minimum_nights',y='price',data=df,color='lime',alpha=0.8)
plt.xlabel('minimum_nights',fontsize = 15,color='blue')
plt.ylabel('price',fontsize = 15,color='blue')
plt.title('minimum_nights  VS  price',fontsize = 20,color='blue')
plt.grid()


# **Work in Progress!!**
