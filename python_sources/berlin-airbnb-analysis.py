#!/usr/bin/env python
# coding: utf-8

# # Predicting the price for an Airbnb Host in Berlin
# Welcome to my another Kernel. In this Kernel I will explore Airbnb Data in Berlin. Airbnb is taking over typical hospitality services and that is the reason, why I want to explore this dataset.
# 
# **Context**
# 
# Airbnb has successfully disrupted the traditional hospitality industry as more and more travelers decide to use Airbnb as their primary accommodation provider. Since its beginning in 2008, Airbnb has seen an enormous growth, with the number of rentals listed on its website growing exponentially each year. In Germany, no city is more popular than Berlin. That implies that Berlin is one of the hottest markets for Airbnb in Europe, with over 22,552 listings as of November 2018.

# ## Table of Contents
# 
# ### <a href='#1. Obtaining and Exploring the Data'> 1. Obtaining and Exploring the Data </a>
# 
# ### <a href='#2. Preprocessing the Data'> 2. Preprocessing the Data </a>
# * <a href="#2.1 Picking columns to work with">2.1 Picking columns to work with</a>
# * <a href="#2.2 Cleaning Data">2.2 Cleaning Data</a>
# * <a href="#2.3 Dealing with missing values">2.3 Dealing with missing values</a>
# * <a href="#2.4 Feature Engineering">2.4 Feature Engineering</a>
# 
# 
# ### <a href='#3. Visualization of the Data'> 3. Visualization of the Data </a>
# 
# 

# ### <a id="1. Obtaining and Exploring the Data">1. Obtaining and Exploring the Data</a>

# In[ ]:


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


# Loading Dataset
df_listing = pd.read_csv("../input/listings_summary.csv")


# **General Exploration**

# In[ ]:


df_listing.head(2)


# In[ ]:


print(f"AirBnB Dataset has {len(df_listing.columns)} columns and {len(df_listing.index)} rows.")


# ### <a id='2. Preprocessing the Data'>2. Preprocessing the Data</a>

# #### <a id="2.1 Picking columns to work with">2.1 Picking columns to work with</a>

# In[ ]:


df_listing.head()


# In[ ]:


columns_to_keep = ["bedrooms", "bed_type", "amenities", "price",
                   "square_feet", "security_deposit", "cleaning_fee", "guests_included", 
                   "extra_people", "minimum_nights", "instant_bookable","latitude", "longitude"]

len(columns_to_keep)


# In[ ]:


# Overview of dataset
df_raw = df_listing[columns_to_keep] 
df_raw.head()


# #### <a id="2.2 Cleaning Data">2.2 Cleaning Data</a>

# In[ ]:


# making dummy variables from bed_type column - original column will be deleted in the near future!
bed_type_dummy = pd.get_dummies(df_raw["bed_type"], drop_first=True)

df_raw = pd.concat([df_raw, bed_type_dummy], axis=1)
df_raw.head()


# In[ ]:


# Wifi in amenities - 1 if there is, 0 if not
def wifi(value):
    if "wifi" in value.lower():
        return 1
    else:
        return 0
    
df_raw["Wifi"] = df_raw["amenities"].apply(wifi)


# In[ ]:


# Tv in amenities - 1 if there is, 0 if not
def tv(value):
    if "tv" in value.lower():
        return 1
    else:
        return 0
    
df_raw["Tv"] = df_raw["amenities"].apply(tv)


# In[ ]:


df_raw.head()


# #### <a id="2.3 Dealing with missing values">2.3 Dealing with missing values</a>

# In[ ]:


# security_deposit, cleaning_fee - NaNs replaced with 0
df_raw["security_deposit"].fillna("$0.00", inplace=True)
df_raw["cleaning_fee"].fillna("$0.00", inplace=True)

df_raw.head()


# In[ ]:


# Removing $ sign from price columns - some numbers contain "," in them - gonna delete that comma there
def remove_dollar(str_price):
    if "," in str_price:
        str_price = str_price.replace(",", "")
    else:
        pass
    return str_price[1:]


for dollar_column in ["price", "security_deposit", "cleaning_fee", "extra_people"]:
    price_no_dollar = df_raw[dollar_column].apply(remove_dollar)
    df_raw[dollar_column] = price_no_dollar
    
# Converting columns to "float64" dtype
price_num = pd.to_numeric(df_raw["price"])
security_deposit_num = pd.to_numeric(df_raw["security_deposit"])
cleaning_fee_num = pd.to_numeric(df_raw["cleaning_fee"])
extra_people_num = pd.to_numeric(df_raw["extra_people"])

# Removing the old columns
df_raw["price"] = price_num
df_raw["security_deposit"] = security_deposit_num
df_raw["cleaning_fee"] = cleaning_fee_num
df_raw["extra_people"] = extra_people_num


# In[ ]:


df_raw.head()


# In[ ]:


# NaNs in Dataframe check
df_raw.isna().sum()


# In[ ]:


# square_feet col drop
df_raw.drop("square_feet", axis=1, inplace=True)


# In[ ]:


df_raw["bedrooms"].mean()  # Mean is nearly number 1, so we will replace NaNs in "bedrooms" column with value 1


# In[ ]:


df_raw["bedrooms"].fillna(1.0, inplace=True)
df_raw["bedrooms"].isna().any()  # check if there are some NaNs 


# In[ ]:


df_raw.head()


# In[ ]:


# "instant_bookable" - replace True as 1, False as 0
def bookable(value):
    if value == "t":
        return 1
    else:
        return 0
    
df_raw["instant_bookable"] = df_raw["instant_bookable"].apply(bookable)


# In[ ]:


# Dropping other columns we won't need in future
df_raw.drop(["bed_type", "amenities"], axis=1, inplace=True)
df_raw.head()


# #### <a id="2.4 Feature Engineering">2.4 Feature Engineering</a>

# In this section we will use Python module geopy.distance to calculate distance from Berlin's centre, afterwards we will delete "latitude" and "longitude" columns.

# In[ ]:


from geopy.distance import great_circle

def distance_from_centre(lat, lon):
    berlin_centre = (52.520008, 13.404954)
    apartment_spot = (lat, lon)
    return round(great_circle(berlin_centre, apartment_spot).km, 1)

df_raw["Distance"] = df_raw.apply(lambda x: distance_from_centre(x.latitude, x.longitude), axis=1)
df_raw.head()


# In[ ]:


# Dropping "latitude" and "longitude" columns
df_raw.drop(["latitude", "longitude"], axis=1, inplace=True)
df_raw.head()


# In[ ]:


# Overview of working data
print("Our final working dataset has {} rows and {} columns.".format(*df_raw.shape))


# In[ ]:


# Assigning Dataframe to new final variable
df_final = df_raw.copy()  # shallow copy of the Dataframe
df_final.head()


# ### <a id='3. Visualization of the Data'> 3. Visualization of the Data </a>
# 

# In[ ]:


# How many apartments has Wi-Fi connection
sns.countplot(df_final["Wifi"], hue=df_final["Tv"], palette="pastel")
plt.title("Wifi and TV in apartments", pad=20)


# In[ ]:


# Distances from the centre of the Berlin to apartments
sns.distplot(df_final["Distance"], bins=10, kde=False)


# In[ ]:


# Type of bed in the apartments
sns.countplot(df_listing["bed_type"])


# In[ ]:




