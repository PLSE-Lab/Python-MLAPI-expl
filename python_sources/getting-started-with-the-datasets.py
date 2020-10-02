#!/usr/bin/env python
# coding: utf-8

# This notebook is for a basic exploration of the competition datasets, and is meant to provide a quick overview before you get started on anything. There is a total of **7 datasets**: 
# 
# - `air_reserve.csv`
# - `hpg_reserve.csv`
# - `air_visit_data.csv`
# - `date_info.csv`
# - `store_id_relation.csv`
# - `air_store_info.csv`
# - `hpg_store_info.csv`

# In[66]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno # check for missing values

# Input data files are available in the "../input/" directory.
# reserves
air_reserve = pd.read_csv("../input/air_reserve.csv")
hpg_reserve = pd.read_csv("../input/hpg_reserve.csv")

# others
visits = pd.read_csv("../input/air_visit_data.csv")
dates = pd.read_csv("../input/date_info.csv")
relation = pd.read_csv("../input/store_id_relation.csv")

# store info
air_store_info = pd.read_csv("../input/air_store_info.csv")
hpg_store_info = pd.read_csv("../input/hpg_store_info.csv")


# # Check for missing values

# In[89]:


# visualization of missing data, white fields indicate NAs
# not plotting for relation and date info since it's meant to provide basic information
msno.matrix(air_reserve)
msno.matrix(hpg_reserve)
msno.matrix(visits)
msno.matrix(air_store_info)
msno.matrix(hpg_store_info)


# Yay! There are no missing values in the datasets.

# # Reservations
# Datasets: `air_reserve.csv` and `hpg_reserve.csv`

# In[67]:


air_reserve.head()


# In[84]:


hpg_reserve.head()


# In[69]:


# combine reserve datasets since store_id for air and hpg have labels
# renaming columns
cols = ["store_id", "visit_datetime", "reserve_datetime", "reserve_visitors"]
air_reserve.columns = cols 
hpg_reserve.columns = cols

# creating a new dataframe with new column names
reserves = pd.DataFrame(columns=cols)
reserves = pd.concat([air_reserve, hpg_reserve])

reserves.info()
reserves.describe()


# In[70]:


print("Number of restaurants with reservations from AirREGI = ", str(len(air_reserve['store_id'].unique())))
print("Number of restaurants with reservations from hpg = ", str(len(hpg_reserve['store_id'].unique())))


# In[71]:


# plot number of visitors per reservation
sns.set(color_codes=True)
visitors = reserves["reserve_visitors"]
sns.distplot(visitors)


# # Visits to Air Restaurants
# Dataset: `air_visit_data.csv`

# In[72]:


# plot number of visits to each air restaurant
sns.set(color_codes=True)
visitors = visits["visitors"]
sns.distplot(visitors, color="y")


# In[73]:


visits.info()
visits.describe()


# In[74]:


print("Number of Air restaurants = ", str(len(visits["air_store_id"].unique())))


# # Relation and Dates
# Datasets: `store_id_relation.csv` and `date_info.csv`

# In[75]:


relation.info()


# In[76]:


dates.head()


# # Store Info
# Datasets: `air_store_info.csv` and `hpg_store_info.csv`

# In[77]:


print("Number of Air restaurants = ", str(len(air_store_info)))
print("Number of hpg restaurants = ", str(len(hpg_store_info)))


# The number of air restaurants matches that in `air_visit_data.csv`.
# 
# However, there's a different number of restaurants in the `air_reserve.csv` and `hpg_reserve.csv` datasets. This is because only information of **selected **restaurants were found in `air_store_info.csv` and `hpg_store_info.csv`. 

# ## Air Restaurants

# In[78]:


air_store_info.info()
air_store_info.head()


# In[79]:


print("Genres:")
air_store_info["air_genre_name"].unique()


# In[94]:


print("Number of unique areas = ", str(len(air_store_info["air_area_name"].unique())))


# In[96]:


# unique areas (expand for the list)
air_store_info["air_area_name"].unique()


# ## hpg restaurants

# In[81]:


hpg_store_info.info()
hpg_store_info.head()


# In[91]:


print("Genres:")
hpg_store_info["hpg_genre_name"].unique()


# In[98]:


print("Number of unique areas = ", str(len(hpg_store_info["hpg_area_name"].unique())))


# In[97]:


# unique areas (expand for the list)
hpg_store_info["hpg_area_name"].unique()


# `air_store_info.csv` and `hpg_store_info.csv`  have **different** genre and area names. 

# In[ ]:




