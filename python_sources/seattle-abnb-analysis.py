#!/usr/bin/env python
# coding: utf-8

# # Seattle AirBnB analysis - using CRISP-DM process

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from datetime import datetime
import calendar
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn import preprocessing
from time import time
from sklearn.metrics import accuracy_score, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV, Lasso
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# Any results you write to the current directory are saved as output.


# ## **CRISP-DM (Cross-Industry Standard Process for Data Mining)**
# ### Business Understanding
#  * Understand the business drivers of the analysis to ensure value is provided in the correct focus areas. This involves understanding the business context and domain. 
#  
# ### Data Understanding
#  * Collect data, describe data and explore data. Usually includes checking for data quality issues and getting clarity around what data is available and what is not available.
# 
# ### Data Preparation
#  * Select data, clean data, dervie/generate new data fields, and integrate data. The output will be data sets ready for use in analysis and modelling.
# 
# ### Modeling
#  * Select modelling technique (simple or complex), build model and assess model.
# 
# ### Evaluation
#  * Evaluate degree to which the model meets business aims and answers the business questions posed. Ideally, we want to draw conclusions related to the questions posed. Plan for potential future steps.
# 
# ### Deployment
#  * Determine a strategy for deployment. Goes into the realms of operationalising models for the business.

# ## **Business understanding**
# 
# This step involves learning more about the business domain we are working on. Subsequently, this enables us to ask questions that will help us improve our understanding of the domain.
# 
# Airbnb is an online platform that connects travellers looking for accommodation to hosts of spaces available for rent.
# 
# In this analysis, we will be using Seattle Airbnb data. In order to set the scene, let's assume that we are looking to buy an investment property in Seattle. We are intending to rent out the property through Airbnb and maximise our revenue returns from rental income. Guided by that, we are interested in learning more about the following: 
# 
# > **1. What are the highest revenue generating locations for Seattle hosts?**
#  * We are keen to find out the best neighbourhood to buy the property. 
#  
# > **2. When in the year are the highest revenues generated? **
#  * Knowing the time of the year that drives the most revenue will enable us to vary prices to maximise revenue at those times.
# 
# > **3. What are the property traits that attract the highest revenues? ** 
#  * It is beneficial to purchase a property with characteristics that are in high demand.

# ## **Data understanding**
# 
# This step involves learning more about the data available for the business domain we are working on. The aim here is to see what data is available, look for high level characteristics and patterns in the data through summarisations, aggregations and visualisations.
# 
# Base on the questions posed above, the key data points required are:
# 1. **Price**
#  > - **Fields in the data (listings)**: 'price', 'weekly_price', 'monthly_price'
# 
# 2. **Revenue**
#  > - **Join (listings) and (calendar)**: Assume that if a property is not available, it has been rented out at the daily price provided in the (listings) dataset
#  > - **Revenue for a given day** = [if(calendar.available == 'f') then 1, else 0] * [listings.price] 
#  > - **Average revenue for a property** = ([if(calendar.available == 'f') then 1, else 0] * [listings.price]) / sum([if(calendar.available == 'f') then 1, else 0])
#  
# 3. **Location**
#  > - **Fields in the data (listings)**: 'street', 'neighbourhood', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market', 'smart_location'
# 
# 4. **Listing date**
#  > - **Fields in the data (calendar)**: 'date'
# 
# 5. **Property characteristics: Building type, building size, bedroom count, bed count, bed type, bathroom count, room type**  
#  > - **Fields in the data (listings)**: 'property_type', 'room_type', 'accommodates', 'bathrooms', 'bedrooms', 'beds', 'bed_type', 'amenities', 'square_feet'
# 
# 6. **Joining all 3 datasets**
#  > - **Fields in the data (listings)**: 'id'
#  > - **Fields in the data (calendar)**: 'listing_id'
#  > - **Fields in the data (reviews)**: 'listing_id'
# 
# With this preliminary data knowledge, we will be able to hone into the data sets that we will need and focus our data preparation and cleaning efforts on them.

# In[ ]:


# Read in data into data frames
listings = pd.read_csv('../input/listings.csv') 
calendar = pd.read_csv('../input/calendar.csv')
reviews = pd.read_csv('../input/reviews.csv')


# In[ ]:


# Quick look at data - listings

# Rows and columns count
print(listings.shape)

print('-'*90)

# List of the columns available
print(list(listings.columns))

# Sample
listings.head(5)


# In[ ]:


# Check for missing values in the columns for each dataset- returning missing value as a percentage
(listings.isnull().sum()/len(listings)*100).sort_values(ascending=False)


# In[ ]:


# List columns with greater than 30% null values 
[cols for cols in listings.columns.values if (listings[cols].isnull().sum()/len(listings)*100)>=30]


# In[ ]:


# Now that we have had a look at the original listings data set, we can proceed to gaining more focused insights

# Based on the questions that we are looking to answer, create a listings data set containing only fields/features that are useful
listings_relevant = listings[['id'
                              , 'price'
                              , 'weekly_price'
                              , 'monthly_price'
                              , 'street'
                              , 'neighbourhood'
                              , 'neighbourhood_cleansed'
                              , 'neighbourhood_group_cleansed'
                              , 'city'
                              , 'state'
                              , 'zipcode'
                              , 'market'
                              , 'smart_location'
                              , 'property_type'
                              , 'room_type'
                              , 'accommodates'
                              , 'bathrooms'
                              , 'bedrooms'
                              , 'beds'
                              , 'bed_type'
                              , 'amenities'
                              , 'square_feet'
                             ]]

# Quick look at data - listings_relevant
print(listings_relevant.shape)

print('-'*90)

# List of the columns available
print(list(listings_relevant.columns))

# Sample of table
listings_relevant.head(5)


# In[ ]:


# Check for missing values in the columns for each dataset- returning missing value as a percentage
(listings_relevant.isnull().sum()/len(listings_relevant)*100).sort_values(ascending=False)


# In[ ]:


# List columns with greater than 30% null values 
[cols for cols in listings_relevant.columns.values if (listings_relevant[cols].isnull().sum()/len(listings_relevant)*100)>=30]

# These columns will be removed as there are too many null values to properly impute values in place of nulls
# Additionally, there are also other fields that can be used to provide similar information: such as 'weekly_price' and 'monthly_price' provides same information as 'price'
    


# In[ ]:


# Quick look at data - calendar
print(calendar.shape)

print('-'*90)

# List of the columns available
print(list(calendar.columns))

# Sample of table
calendar.head(30)


# In[ ]:


# Check if the property is not available - what are the variety of 'prices' available
calendar[calendar['available'] == 'f']['price'].unique()


# In[ ]:


# Quick look at unique calendar values to gauge data date range
print(calendar['date'].nunique())

calendar['date'].unique()


# In[ ]:


# Check for missing values in the columns for each dataset- returning missing value as a percentage
(calendar.isnull().sum()/len(calendar)*100).sort_values(ascending=False)


# In[ ]:


# Now that we have had a look at the original calendar data set, we can proceed to gaining more focused insights

# Based on the questions that we are looking to answer, create a calendar data set containing only fields/columns and records that are useful
calendar_relevant = calendar[calendar['available'] == 'f']
calendar_relevant = calendar_relevant[['listing_id'
                                      , 'date'
                                      , 'available'
                                     ]]

# Quick look at data - calendar_relevant
print(calendar_relevant.shape)

print('-'*90)

# List of the columns available
print(list(calendar_relevant.columns))

# Sample of table
calendar_relevant.head(5)


# In[ ]:


# Quick look at data - reviews
print(reviews.shape)

print('-'*90)

# List of the columns available
print(list(reviews.columns))

# Sample data view 
reviews.head()


# In[ ]:


# Check for missing values in the columns for each dataset- returning missing value as a percentage
(reviews.isnull().sum()/len(reviews)*100).sort_values(ascending=False)


# From the quick checks above, we can make the following observations about each dataset.
# 
# ### listings
# - The raw dataset has (3818 rows, 92 columns/fields)
# - Based on the business questions we are trying to answer, we then create a 'listings_relevant' dataset focusing on 22 columns/fields
#     - 3 fields ('square_feet', 'monthly_price', 'weekly_price') have greater than 30% null values
#         - 'square_feet' is useful to determine a key property characteristic, which is the size of the property. Given the substantial number of null values, we will disregard it as it would be inaccurate to impute.
#         - 'monthly_price', 'weekly_price' fields can be disregarded as they provide approximately the same information as 'price'(daily price). For the purposes of our analysis, we will disregard the effects of bulk discounts and focus on non-discounted daily prices.  
# 
# 
# ### calendar 
# - The raw dataset has (1393570 rows, 4 columns/fields)
# - There are 365 unique date values
# - Based on the business questions, all fields are relevant, however from a records perspective, we are only interested in records where 'available' = 'f' (the property is unavailable, presumably as it has been rented out at the daily rate provided in the listings data set). This is as to calculate revenue, we only need data relating to when the property has been rented out. Thus, we create a 'calendar_relevant' dataset with that filter.
#     - The 'price' field has greater than 30% null values. However, we will be using price information from the 'listings' dataset instead. Additionally, when 'available' = 'f', there are no prices. Therefore, we will drop this field.
#     - We calculate 'revenue' by assuming that if a property is not 'available' for a given 'date', it is being rented out at the daily 'price' in the 'listings' dataset.
#     
#     
# ### reviews
# - The raw dataset has (84849 rows, 6 columns/fields)
# - Based on the business questions, we are not going to need this dataset. However, it will be useful for future use.

# ## **Data preparation**
# 
# This step involves preparing the dataset for analysis and modelling - en route to answering the business questions.
# 
# ### High level preparation steps to be performed:
# #### Clean the (listings_relevant) dataset --> Output: (listings_relevant_clean) dataset
#  - Drop fields with > 30% null values
#  - Convert 'price' from string into numeric float ('price_clean')
#      - Drop original 'price' field
#  - Explore property/building type related data
#      - Fields covered: ['property_type', 'room_type', 'bed_type', 'accommodates','bathrooms','bedrooms','beds']
#  - Clean property/building type related data     
#      - For categorical fields, drop records with null values
#      - For numeric fields, replace null values with 0
#  - Explore location related data
#      - Fields covered: ['street', 'neighbourhood', 'neighbourhood_cleansed','neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market','smart_location']
#      - Determine the fields to use to represent location:
#         > - 'street': Has too many (1442) unique values, limiting our ability to get a generalised view of revenue-maximising property locations. 
#         >> **Decision: EXCLUDE from analysis.** 
#         > - 'neighbourhood': Has substantial null records (416) and represents the same information as 'neighbourhood_cleansed'. 
#         >> **Decision: EXCLUDE from analysis.**
#         > - 'neighbourhood_cleansed': Has 0 null records and 87 unique values, which will enable us to get a generalised view of revenue-maximising property locations. 
#         >> **Decision: INCLUDE in analysis.**
#         >>> - Cleaning required: None
#         > - 'neighbourhood_group_cleansed': Has 0 null records and 17 unique values, which will enable us to get a generalised view of revenue-maximising property locations. 
#         >> **Decision: INCLUDE in analysis.**
#         >>> - Cleaning required: None        
#         > - 'city': Too high level, thus has limited use. 
#         >> **Decision: EXCLUDE from analysis.**
#         > - 'state': Only covers WA, thus has limited use. 
#         >> **Decision: EXCLUDE from analysis.**
#         > - 'zipcode': Has 0 null records and 17 unique values, which will enable us to get a generalised view of revenue-maximising property locations. 
#         >> **Decision: INCLUDE in analysis.**
#         >>> - Cleaning required: Yes - There are records with value '99\n98122' --> should be replaced with '98122'        
#         > - 'market': Only covers Seattle, thus has limited use. 
#         >> **Decision: EXCLUDE from analysis.**
#         > - 'smart_location': Too high level, thus has limited use. 
#         >> **Decision: EXCLUDE from analysis.**
#  - Clean location related data        
#      - In 'zipcode' - Replace records with value '99\n98122' --> with '98122'
#      - Drop records with null values
#  - Drop fields not required for analysis and modelling
#      - ['street', 'neighbourhood', 'city', 'state', 'market','smart_location', 'amenities']
#  - Rename key join 'id' field to be 'listing_id' to make it consistent with the calendar_relevant table    
#      
# #### Clean the (calendar_relevant) dataset --> Output: (calendar_relevant_clean) dataset
#  - Pre cleaning has been done in previous section
#  - Drop fields with > 30% null values
#  - Add additional date fields (year, month, day, day of week) to the (calendar_relevant_clean) dataset
#  
# #### Join/Merge the (listings_relevant_clean) and (calendar_relevant_clean) datasets
#  - **Create (all_listings_to_days_rented):** Left join/merge (listings_revelant_clean) as base with (calendar_relevant_clean) on ('listing_id')
#   > - For a given property, return all dates it has been rented out
#   > - This dataset will show: For a given property, what is the total_revenue
#  - **Create (all_days_to_listings_rented):** Left join/merge (calendar_relevant_clean) as base with (listings_revelant_clean) on ('listing_id')
#   > - For a given date, return all properties that have been rented out
#   > - This dataset will show: For a given date, what is the total revenue for all properties rented out on that date
# 

# In[ ]:


# Clean the (listings_relevant) dataset

# List columns with greater than 30% null values 
null_pct_greater30_columns = [cols for cols in listings_relevant.columns.values if (listings_relevant[cols].isnull().sum()/len(listings_relevant)*100)>=30]
print('Fields with > 30% null values to be dropped: {}'.format(null_pct_greater30_columns))

# Remove the fields with > 30% null values
listings_relevant_clean = listings_relevant.drop(columns=null_pct_greater30_columns)
listings_relevant_clean.head()


# In[ ]:


# Convert 'price' from string into numeric float ('price_clean')
listings_relevant_clean['price_clean'] = listings_relevant_clean['price'].replace('[\$,]', '', regex=True).astype(float)

# Drop 'price' column
listings_relevant_clean = listings_relevant_clean.drop(columns=['price'])

# Quick view
listings_relevant_clean.head()


# In[ ]:


# listings_relevant_clean.describe()
listings_relevant_clean.dtypes

# Fields of numeric data type (int, float)
#  - price, price_clean, accommodates, bathrooms, bedrooms, beds

# Fields of categorical nature /non-numeric data type (object, string)
#  - property_type, room_type, bed_type, amenities


# In[ ]:


# Explore and clean the categorical fields

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Create a function that - explores a given field in data set, and then drops null values
def explore_clean_dropna(df, categ_field_name, key_field_name):
    print('Exploring and cleaning: {}'.format(categ_field_name))
    
    print('-'*90)
    
    # Look at unique values
    print(df[categ_field_name].unique())

    print('-'*90)

    # Look at number of properties for each category
    print(df.groupby([categ_field_name]).count()[key_field_name].sort_values(ascending=False))

    print('-'*90)

    # Identify null values in category
    v1 = df[categ_field_name].isnull().sum()
    print('Number of null value records for category ({}) pre clean: {}'.format(categ_field_name,v1))

    # Drop records with null values as it is insignificant
    df.dropna(subset=[categ_field_name], inplace=True)
    
    print('-'*90)

    # Check null values in category post clean
    v2 = df[categ_field_name].isnull().sum()
    print('Number of null value records for category ({}) post clean: {}'.format(categ_field_name,v2))    

# ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------    
# Create a function that - explores a given field in data set, and then replaces null values with proposed value
def explore_clean_fillna(df, categ_field_name, key_field_name, null_replacement):
    print('Exploring and cleaning: {}'.format(categ_field_name))
    
    print('-'*90)
    
    # Look at unique values
    print(df[categ_field_name].unique())

    print('-'*90)

    # Look at number of properties for each category
    print(df.groupby([categ_field_name]).count()[key_field_name].sort_values(ascending=False))

    print('-'*90)

    # Identify null values in category
    v1 = df[categ_field_name].isnull().sum()
    print('Number of null value records for category ({}) pre clean: {}'.format(categ_field_name,v1))

    # Drop records with null values as it is insignificant
    df[categ_field_name] = df[categ_field_name].fillna(null_replacement)

    print('-'*90)

    # Check null values in category post clean
    v2 = df[categ_field_name].isnull().sum()
    print('Number of null value records for category ({}) post clean: {}'.format(categ_field_name,v2))        


# In[ ]:


# Explore and clean the categorical fields relating to property/building type by applying the created function (explore_clean_dropna)

# Define list of columns related to property/building type
columns_property_type_categ = list(listings_relevant_clean[['property_type', 'room_type', 'bed_type']]) 

# Check the number of unique values in each column
for i in columns_property_type_categ: 
        explore_clean_dropna(listings_relevant_clean, i, 'id')
        print(' '*150)
        print('~'*150)
        print(' '*150)


# In[ ]:


# Explore and clean the numerical fields relating to property/building type by applying the created function (explore_clean_fillna)

# Define list of columns related to property/building type
columns_property_type_numeric = list(listings_relevant_clean[['accommodates','bathrooms','bedrooms','beds']]) 

# Check the number of unique values in each column
for i in columns_property_type_numeric: 
        explore_clean_fillna(listings_relevant_clean, i, 'id', 0)
        print(' '*150)
        print('~'*150)
        print(' '*150)


# In[ ]:


# Explore location data
# We start with all the fields related to location = ['street', 'neighbourhood', 'neighbourhood_cleansed','neighbourhood_group_cleansed', 'city', 'state', 'zipcode', 'market','smart_location']

# Define list of columns related to location
columns_location_categ = list(listings_relevant_clean[['street', 'neighbourhood', 
                                        'neighbourhood_cleansed','neighbourhood_group_cleansed', 
                                        'city', 'state', 'zipcode', 'market','smart_location']]) 

# Check the number of unique values in each column
for i in columns_location_categ: 
        num_unique=(listings_relevant_clean[i].nunique())
        num_isnull=(listings_relevant_clean[i].isnull().sum())
        print('Field name: {}, Number of null records: {}, Number of unique values: {}'.format(i, num_isnull, num_unique))
        print('+'*90)
        print('List of unique values:')
        print(listings_relevant_clean[i].unique())
        print(' '*150)
        print('-'*150)
        print(' '*150)
        
# Based on the results, the following decisions have been made:
# - 'street': Has too many (1442) unique values, limiting our ability to get a generalised view of revenue-maximising property locations. **Decision: EXCLUDE from analysis.** 
# - 'neighbourhood': Has substantial null records (416) and represents the same information as 'neighbourhood_cleansed'. **Decision: EXCLUDE from analysis.**
# - 'neighbourhood_cleansed': Has 0 null records and 87 unique values, which will enable us to get a generalised view of revenue-maximising property locations. **Decision: INCLUDE in analysis.**
    # -- Cleaning required: None 
# - 'neighbourhood_group_cleansed': Has 0 null records and 17 unique values, which will enable us to get a generalised view of revenue-maximising property locations. **Decision: INCLUDE in analysis.**
    # -- Cleaning required: None
# - 'city': Too high level, thus has limited use. **Decision: EXCLUDE from analysis.**
# - 'state': Only covers WA, thus has limited use. **Decision: EXCLUDE from analysis.**
# - 'zipcode': Has 0 null records and 17 unique values, which will enable us to get a generalised view of revenue-maximising property locations. **Decision: INCLUDE in analysis.**
    # -- Cleaning required: Yes - There are records with value '99\n98122' --> should be replaced with '98122'
# - 'market': Only covers Seattle, thus has limited use. **Decision: EXCLUDE from analysis.**
# - 'smart_location': Too high level, thus has limited use. **Decision: EXCLUDE from analysis.**


# In[ ]:


# Explore and clean the categorical fields relating to location by applying the created function (explore_clean_dropna)

# Define list of columns related to location type
columns_location_categ = list(listings_relevant_clean[['neighbourhood_cleansed','neighbourhood_group_cleansed','zipcode']]) 

# Check the number of unique values in each column
for i in columns_location_categ: 
        explore_clean_dropna(listings_relevant_clean, i, 'id')
        print(' '*150)
        print('~'*150)
        print(' '*150)


# In[ ]:


# Clean specific location data

# Clean the location values 
listings_relevant_clean.loc[listings_relevant_clean['zipcode'] == '99\n98122', 'zipcode'] = '98122'
listings_relevant_clean['zipcode'].unique()


# In[ ]:


# Clean overall property/building and location data

# Drop columns to be excluded from analysis
listings_relevant_clean = listings_relevant_clean.drop(columns=['amenities', 'street', 'neighbourhood','city', 'state', 'market','smart_location'])

# Rename key join 'id' field to be 'listing_id' to make it consistent with the calendar_relevant table
listings_relevant_clean.columns = ['listing_id', 'neighbourhood_cleansed', 'neighbourhood_group_cleansed',
                                   'zipcode', 'property_type', 'room_type', 'accommodates', 'bathrooms',
                                   'bedrooms', 'beds', 'bed_type', 'price_clean']

# Quick view
listings_relevant_clean.head()


# In[ ]:


# Clean the (calendar_relevant) dataset

# List columns with greater than 30% null values 
null_pct_greater30_columns = [cols for cols in calendar_relevant.columns.values if (calendar_relevant[cols].isnull().sum()/len(calendar_relevant)*100)>=30]
print('Fields with > 30% null values to be dropped: {}'.format(null_pct_greater30_columns))

# Remove the fields with > 30% null values
calendar_relevant_clean = calendar_relevant.drop(columns=null_pct_greater30_columns)
calendar_relevant_clean.head()


# In[ ]:


# Add additional date fields (year, month, day, day of week) to the (calendar_relevant_clean) dataset
calendar_relevant_clean['year'] = pd.DatetimeIndex(calendar_relevant_clean['date']).year
calendar_relevant_clean['month'] = pd.DatetimeIndex(calendar_relevant_clean['date']).month
calendar_relevant_clean['monthname'] = pd.DatetimeIndex(calendar_relevant_clean['date']).month_name()
calendar_relevant_clean['dayofmonth'] = pd.DatetimeIndex(calendar_relevant_clean['date']).day
calendar_relevant_clean['dayofweekname'] = pd.DatetimeIndex(calendar_relevant_clean['date']).day_name()


# Quick view of table
calendar_relevant_clean.tail()


# In[ ]:


# Join/Merge the (listings_relevant_clean) and (calendar_relevant_clean) datasets

print('{} (record count:{}) has the following fields: {}'.format('listings_relevant_clean',listings_relevant_clean.shape[0], list(listings_relevant_clean.columns)))
print('-'*90)
print('{} (record count:{}) has the following fields: {}'.format('calendar_relevant_clean',calendar_relevant_clean.shape[0],list(calendar_relevant_clean.columns)))


# Part 1 --> Preparing data for Q1, Q3
# - Summarise (calendar_relevant_clean - summarised to show - for a given property, how many days it has been rented out) --> output is new dataset called (days_rented_per_property)
# - Create (all_listings_to_days_rented): Left join/merge (listings_relevant_clean) as base, with (days_rented_per_property) on (listing_id)
# -- Ie. For a given property, return all dates it has been rented out
# - Replace all null 'days_rented' with 0
# - Create a new 'revenue_per_property' field ('price_clean' * 'days_rented')


# Part 2 --> Preparing data for Q2
# - Get a subset of (listings_relevant_clean) - showing for a given property, the daily rate --> output is new dataset called (daily_price_per_property)
# - Create (all_days_to_listings_rented): Left join/merge (calendar_relevant_clean) as base with (daily_price_per_property) on ('listing_id')
# -- Ie. For a given date, return all properties that have been rented out
# - Create a new 'total_revenue_per_day' field = sum('price_clean') groupby ('date','available','year','month','monthname','dayofmonth','dayofweekname')


# In[ ]:


# Part 1 --> Preparing data for Q1, Q3
# Summarise (calendar_relevant_clean - summarised to show - for a given property, how many days it has been rented out) --> output is new dataset called (days_rented_per_property)
# Determine for a given 'listing_id', the number of days it has been rented out (regardless of when)
days_rented_per_property = pd.DataFrame(calendar_relevant_clean.groupby('listing_id')['date'].count().sort_values(ascending=False))
days_rented_per_property.reset_index(level=0, inplace=True)
days_rented_per_property.columns = ['listing_id', 'days_rented']

# Create a historgram to show the number of occurences for each "days_rented"
days_rented_per_property['days_rented'].hist(
                                             bins = 36
                                            ,figsize=(24,5)
                                            );
plt.suptitle('Number of occurences for each "days_rented"', x=0.5, y=1.05, ha='center', fontsize='x-large');


# In[ ]:


# Create (all_listings_to_days_rented): Left join/merge (listings_relevant_clean) as base, with (days_rented_per_property) on (listing_id)
all_listings_to_days_rented = pd.merge(listings_relevant_clean, days_rented_per_property, how='left', on=['listing_id'])

# Quick check of new data set
print(all_listings_to_days_rented.describe())
# There are property listings with no corresponding calendar data. This means that these properties have not been rented out in the time frame we are analysing.

print(' '*150)
print('-'*150)
print(' '*150)

# Replace all null 'days_rented' with 0
explore_clean_fillna(all_listings_to_days_rented, 'days_rented', 'listing_id', 0)


# In[ ]:


# Create a new 'revenue_per_property' field ('price_clean' * 'days_rented')
all_listings_to_days_rented['revenue_per_property'] = all_listings_to_days_rented['price_clean'] * all_listings_to_days_rented['days_rented']

# Quick look
all_listings_to_days_rented.head()


# In[ ]:


# Quick look at revenue per property distribution using a boxplot

figdim = (28,5)
sns.set(rc={'figure.figsize':figdim})
sns.boxplot(y=all_listings_to_days_rented["revenue_per_property"] , orient='h');


# In[ ]:


# Part 2 --> Preparing data for Q2
# - Get a subset of (listings_relevant_clean) - showing for a given property, the daily rate --> output is new dataset called (daily_price_per_property)
# Determine for a given 'listing_id', the 'price_clean'
daily_price_per_property = listings_relevant_clean[['listing_id', 'price_clean']]

# - Create (all_days_to_listings_rented): Left join/merge (calendar_relevant_clean) as base with (daily_price_per_property) on ('listing_id')
# -- Ie. For a given date, return all properties that have been rented out
all_days_to_listings_rented_raw = pd.merge(calendar_relevant_clean, daily_price_per_property, how='left', on=['listing_id'])

# - Create a new 'total_revenue_per_day' field = sum('price_clean') groupby ('date','available','year','month','monthname','dayofweek','dayofweekname')
all_days_to_listings_rented = pd.DataFrame(
    all_days_to_listings_rented_raw.groupby(['date','available','year','month','monthname','dayofmonth','dayofweekname'])['price_clean']
        .agg(['sum', 'count']))
all_days_to_listings_rented.reset_index(inplace=True)
all_days_to_listings_rented.columns = ['date','available','year','month','monthname','dayofmonth','dayofweekname', 'total_revenue_per_date', 'total_properties_rented_per_date']

# Quick look
all_days_to_listings_rented.tail()


# In[ ]:


# Create a histogram to show the number of occurences of 'total_revenue_per_date' for each "days_rented"
all_days_to_listings_rented['total_revenue_per_date'].hist(
                                             bins = 50
                                            ,figsize=(24,5)
                                            );
plt.suptitle('Number of occurences for each "total_revenue_per_date"', x=0.5, y=1.05, ha='center', fontsize='x-large');


# In[ ]:


# Create a histogram to show the number of occurences of 'total_properties_rented_per_date' for each "days_rented"
all_days_to_listings_rented['total_properties_rented_per_date'].hist(
                                             bins = 50
                                            ,figsize=(24,5)
                                            );
plt.suptitle('Number of occurences for each "total_properties_rented_per_date"', x=0.5, y=1.05, ha='center', fontsize='x-large');


# ## **Modelling, Evaluation, Deployment**
# 
# From this point onwards, we will be changing the approach slightly.
# We will be performing the Modelling, Evaluation and Deployment steps end to end for each of the following 3 business questions.
# 
# > **1. What are the highest revenue generating locations for Seattle hosts?**
#  * We are keen to find out the best neighbourhood to buy the property. 
#  
# > **2. When in the year are the highest revenues generated? **
#  * Knowing the time of the year that drives the most revenue will enable us to vary prices to maximise revenue at those times.
# 
# > **3. What are the property types that attract the highest revenues? ** 
#  * It is beneficial to pruchase a property that is in high demand.
#  
# Let's get started!!!
# 

# ## **Question 1: What are the highest revenue generating locations for Seattle hosts?**
# 
# We are keen to find out the best location (at the neighbourhood level) to buy the property. The cells after this contain the workings for the steps below.
# 
# ### **Modelling**
#  - We are using the (all_listings_to_days_rented) dataset prepared earlier.
#  - The dataset is then summarised to provide the average revenue per property, number of properties, and total revenue of all properties for each neighbourhood. 
#  - The summarised data is then presented in a table, where each row covers a neighbourhood. The records are sorted by average revenue per property in each neighbourhood in descending order. In each of the summarised columns, the maximum values are highlighted in 'Gold' and the values are shaded in different gradients of 'Green' (the darker the gradient, the higher the value is within the column).  
#  
# ### **Evaluation**
#  - It is best to look at the data from multiple angles. Here, we look at the average revenue per property, total revenue and number of properties in each neighbourhood.
#  - From the perspective of average revenue per property, Montlake ranks the highest where a property here makes $ 37,740 on average. 
# 
#  - From the viewpoint of total revenue, Broadway ranks the highest where all properties collectively generated $ 6,475,088 in revenue. Broadway also has the highest number of properties at 396. 
# 
#  - Properties in this neighbourhood makes $ 16,351 dollars on average, which is less than 50 percent of what a property in Montlake makes. Given that there are a large number of properties, there is greater competition, resulting in more competitive pricing.
#  
#  - This is an interesting insight. We need to consider all these information hollistically. 
#  
#  - We recommend buyers to focus on properties in neighbourhoods with high average revenue per property (Montlake, South Lake Union, North Beach/Blue Ridge) to get the greatest return on investment. It should be noted that these neighbourhoods have relatively small number of properties (19,27,14 respectively) which indicate a slightly more rural locality, where there may be challenges finding demand for properties here. 
#  - Therefore, a further recommendation would be for buyers to target properties in neighbourhoods which ranks well in terms of average revenue per property, total revenue and number of properties. Neighbourhoods that fit this bill are Belltown, East Queen Anne and West Queen Anne. Buying properties here could have a better flow of demand, given its more central locality. 
# 
#  
# ### **Deployment**
#  - There is no model deployment here.
#  

# In[ ]:


# Quick look at which level of neighbourhood to continue analysis on

all_listings_to_days_rented[['neighbourhood_group_cleansed', 'neighbourhood_cleansed']].nunique()


# In[ ]:


# Add in additional revenue metrics
revenue_by_location = pd.DataFrame(all_listings_to_days_rented.groupby(['neighbourhood_cleansed'])['revenue_per_property'].agg(['sum', 'count', 'mean']))
revenue_by_location.reset_index(inplace=True)
revenue_by_location.columns = ['neighbourhood_cleansed','total_revenue','number_of_properties','average_revenue_per_property']
revenue_by_location = revenue_by_location.sort_values(by='average_revenue_per_property', ascending=False)

# Apply color based conditional formatting on the data frame
cm = sns.light_palette("seagreen", as_cmap=True)

(revenue_by_location.style
  .background_gradient(cmap=cm, subset=['total_revenue','number_of_properties','average_revenue_per_property'])
  .highlight_max(subset=['total_revenue','number_of_properties','average_revenue_per_property'], color='gold')

  .format({'total_revenue': "${:,.0f}"})
  .format({'average_revenue_per_property': "${:,.2f}"}) 
)

# Look at average revenue per property, then total revenue


# ## **Question 2: When in the year are the highest revenues generated?**
# 
# This involves knowing the time of the year that generates the most revenue. Having this insight will enable us to vary prices and ensure our properties are available for rent at those times to maximise revenue. The cells after this contain the workings for the steps below.
# 
# ### **Modelling**
#  - We are using the (all_days_to_listings_rented) dataset prepared earlier.
#  - The dataset is then summarised at 3 levels (month level, phase of month level, day of week level) to provide the average revenue per property, number of properties, and total revenue of all properties. 
#  - For all 3 levels, the summarised data is presented in a table, where each row covers a level. The records are sorted by average revenue per property in each neighbourhood in descending order. In each of the summarised columns, the maximum values are highlighted in 'Gold' and the values are shaded in different gradients of 'Green' (the darker the gradient, the higher the value is within the column).  
#  - Charts are also created to help visualise and identify trends at all 3 levels.
#  
# ### **Evaluation**
#  - As for Question 1, it is best to look at the data from multiple angles. Here, we look at the average revenue per property, total revenue and number of properties in each time level.
#  - Firstly, at a 'month' level:
#  > - The highest average revenue generated per property is achieved in March, followed by February. In general, a relatively high average revenue is achieved in the Quarter 1 of the calendar year (January to March).
#  > - January recorded the highest number of properties rented and generated the highest revenue in the year. 
#  > - Hence, it is recommended that property owners put their place up for rent in the first quarter of the year (January to March).
#  - Secondly, at the 'phase of month' level:
#  > - The average revenue generated increases in the last 2 phases of the month and peaks in the last phase of the month (from the 24th to 31st days). 
#  > - However, it is interesting to note that the highest amounts of total revenue is generated in phases 2 and 3 of the month (from the 16th to the 23rd days). 
#  > - Given this conflicting trends, we would recommend property owners to put their places up for rent at the end of the month if they are after higher yield over a short time frame. Alternatively, if property owners are content with renting their places out over longer periods at lower yields, they should make their properties available in the middle of the month. 
#  - Thirdly, at the 'day of week' level:
#  > - Thursday generates the highest average revenue per property, followed by Friday, Wednesday and Saturday.
#  > - On the other hand, Monday generates the highest total revenue (and also has the highest number of properties rented). This is followed by Saturday, Friday, Thursday and Wednesday.
#  > - We recommend property owners to put their place up for rent between Wednesday to Saturday as there is good demand for places and also evidence that renters are willing to pay a slight premium for places. 
# 
#  - Overall, we recommend buyers and property owners to make their properties available for rent in the 1st quarter of the calendar year (January to March), focusing on periods towards the end of the month, especially between Wednesdays to Saturdays.
#  
# ### **Deployment**
#  - There is no model deployment here.
#  

# In[ ]:


print(list(all_days_to_listings_rented.columns))
all_days_to_listings_rented.head()


# In[ ]:


# Quick look at which level of dates to continue analysis on

all_days_to_listings_rented[['date', 'year', 'month', 'monthname', 'dayofmonth','dayofweekname']].nunique()

# From the output below, it makes sense to look for trends at 3 levels: 
# -- month
# -- phase of month --> day of month (which will be further summarised into phase of month --> p1, p2, p3, p4 at 8 days intervals)
# -- day of week

# Separate offshoot data sets will be created with revenue metrics summarised at each of the 3 levels.


# In[ ]:


# Analysis at the 'month' level

# Create a new data set for 'month' level analysis
all_days_to_listings_rented_month_lvl = all_days_to_listings_rented

# Create a unified 'monthofyear' column
all_days_to_listings_rented_month_lvl['month']= all_days_to_listings_rented_month_lvl['month'].apply('{:0>2}'.format)
all_days_to_listings_rented_month_lvl['monthofyear'] = all_days_to_listings_rented_month_lvl['month'] + ' - ' + all_days_to_listings_rented_month_lvl['monthname']

# Calculate total revenue generated per month
all_days_to_listings_rented_month_lvl_revenue = pd.DataFrame(all_days_to_listings_rented_month_lvl.groupby(['monthofyear'])['total_revenue_per_date'].agg(['sum']))
all_days_to_listings_rented_month_lvl_revenue.reset_index(inplace=True)
all_days_to_listings_rented_month_lvl_revenue.columns = ['monthofyear','total_revenue']

# Calculate total number of properties rented per month
all_days_to_listings_rented_month_lvl_propcount = pd.DataFrame(all_days_to_listings_rented_month_lvl.groupby(['monthofyear'])['total_properties_rented_per_date'].agg(['sum']))
all_days_to_listings_rented_month_lvl_propcount.reset_index(inplace=True)
all_days_to_listings_rented_month_lvl_propcount.columns = ['monthofyear','total_property_rented']

# Merge both component tables, and calculate average revenue generated per property for each month
all_days_to_listings_rented_month_lvl_combined = pd.merge(all_days_to_listings_rented_month_lvl_revenue, all_days_to_listings_rented_month_lvl_propcount, how='left', on=['monthofyear'])
all_days_to_listings_rented_month_lvl_combined['average_revenue_per_property'] = all_days_to_listings_rented_month_lvl_combined['total_revenue']/all_days_to_listings_rented_month_lvl_combined['total_property_rented']
all_days_to_listings_rented_month_lvl_combined


# In[ ]:


# Plot the 'month' level summarised data for insights

x_input = all_days_to_listings_rented_month_lvl_combined['monthofyear']
y1_input = all_days_to_listings_rented_month_lvl_combined['total_property_rented']
y2_input = all_days_to_listings_rented_month_lvl_combined['total_revenue']
y3_input = all_days_to_listings_rented_month_lvl_combined['average_revenue_per_property']


# ax1
fig1, ax1 = plt.subplots()
width = 0.35

ax1 = y1_input.plot(kind='bar', width = width, color='Gold', legend=True)
ax1 = y2_input.plot(kind='line', marker='o', secondary_y=True, legend=True)
ax1.set_title('Number of properties rented v Total revenue generated for each month'
              , fontdict={'fontsize': 'x-large'})
ax1 = plt.gca()
ax1.set_xticks(np.arange(len(x_input)))
ax1.set_xticklabels(x_input)


# ax2
fig2, ax2 = plt.subplots()
ax2 = y3_input.plot(kind='line', marker = 'o', color='seagreen')
ax2.set_title('Average revenue generated per property for each month'
              , fontdict={'fontsize': 'x-large'})
ax2 = plt.gca()
ax2.set_xticks(np.arange(len(x_input)))
ax2.set_xticklabels(x_input)


# Display plots
plt.show();


# ------------------------------------------------------------------------------------------------------------------------------

# Apply color based conditional formatting on the data frame
cm = sns.light_palette("seagreen", as_cmap=True)

(all_days_to_listings_rented_month_lvl_combined.style
  .background_gradient(cmap=cm, subset=['total_revenue','total_property_rented', 'average_revenue_per_property'])
  .highlight_max(subset=['total_revenue','total_property_rented', 'average_revenue_per_property'], color='gold')

  .format({'total_revenue': "${:,.0f}"})
  .format({'average_revenue_per_property': "${:,.2f}"})  
)

# Look at average revenue per property, then total revenue


# In[ ]:


# Analysis at the 'phase of month' level

# We will categorise the days of month (1 to 31) into 4 buckets:
# -- p1_of_month: Days 1 to 7
# -- p2_of_month: Days 8 to 15
# -- p3_of_month: Days 16 to 23
# -- p4_of_month: Days 24 to 31

# Create a new data set for 'phase of month' level analysis
all_days_to_listings_rented_phaseofmonth_lvl = all_days_to_listings_rented

# Create a 'phaseofmonth' column
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofmonth'] >= 1), 'phaseofmonth'] = '1 - p1 of month'
all_days_to_listings_rented_phaseofmonth_lvl.loc[all_days_to_listings_rented_phaseofmonth_lvl['dayofmonth'] >= 8, 'phaseofmonth'] = '2 - p2 of month'
all_days_to_listings_rented_phaseofmonth_lvl.loc[all_days_to_listings_rented_phaseofmonth_lvl['dayofmonth'] >= 16, 'phaseofmonth'] = '3 - p3 of month'
all_days_to_listings_rented_phaseofmonth_lvl.loc[all_days_to_listings_rented_phaseofmonth_lvl['dayofmonth'] >= 24, 'phaseofmonth'] = '4 - p4 of month'

# all_days_to_listings_rented_phaseofmonth_lvl['phaseofmonth'] = all_days_to_listings_rented_phaseofmonth_lvl['dayofmonth']

# # Calculate total revenue generated per 'phaseofmonth'
all_days_to_listings_rented_phaseofmonth_lvl_revenue = pd.DataFrame(all_days_to_listings_rented_phaseofmonth_lvl.groupby(['phaseofmonth'])['total_revenue_per_date'].agg(['sum']))
all_days_to_listings_rented_phaseofmonth_lvl_revenue.reset_index(inplace=True)
all_days_to_listings_rented_phaseofmonth_lvl_revenue.columns = ['phaseofmonth','total_revenue']

# # Calculate total number of properties rented per 'phaseofmonth'
all_days_to_listings_rented_phaseofmonth_lvl_propcount = pd.DataFrame(all_days_to_listings_rented_phaseofmonth_lvl.groupby(['phaseofmonth'])['total_properties_rented_per_date'].agg(['sum']))
all_days_to_listings_rented_phaseofmonth_lvl_propcount.reset_index(inplace=True)
all_days_to_listings_rented_phaseofmonth_lvl_propcount.columns = ['phaseofmonth','total_property_rented']

# # Merge both component tables, and calculate average revenue generated per property for each month phase
all_days_to_listings_rented_phaseofmonth_lvl_combined = pd.merge(all_days_to_listings_rented_phaseofmonth_lvl_revenue, all_days_to_listings_rented_phaseofmonth_lvl_propcount, how='left', on=['phaseofmonth'])
all_days_to_listings_rented_phaseofmonth_lvl_combined['average_revenue_per_property'] = all_days_to_listings_rented_phaseofmonth_lvl_combined['total_revenue']/all_days_to_listings_rented_phaseofmonth_lvl_combined['total_property_rented']
all_days_to_listings_rented_phaseofmonth_lvl_combined


# In[ ]:


# Plot the 'phase of month' level summarised data for insights

x_input = all_days_to_listings_rented_phaseofmonth_lvl_combined['phaseofmonth']
y1_input = all_days_to_listings_rented_phaseofmonth_lvl_combined['total_property_rented']
y2_input = all_days_to_listings_rented_phaseofmonth_lvl_combined['total_revenue']
y3_input = all_days_to_listings_rented_phaseofmonth_lvl_combined['average_revenue_per_property']


# ax1
fig1, ax1 = plt.subplots()
width = 0.35

ax1 = y1_input.plot(kind='bar', width = width, color='Gold', legend=True)
ax1 = y2_input.plot(kind='line', marker='o', secondary_y=True, legend=True)
ax1.set_title('Number of properties rented v Total revenue generated for each phase of month'
              , fontdict={'fontsize': 'x-large'})
ax1 = plt.gca()
ax1.set_xticks(np.arange(len(x_input)))
ax1.set_xticklabels(x_input)


# ax2
fig2, ax2 = plt.subplots()
ax2 = y3_input.plot(kind='line', marker = 'o', color='seagreen')
ax2.set_title('Average revenue generated per property for each phase of month'
              , fontdict={'fontsize': 'x-large'})
ax2 = plt.gca()
ax2.set_xticks(np.arange(len(x_input)))
ax2.set_xticklabels(x_input)


# Display plots
plt.show();


# ------------------------------------------------------------------------------------------------------------------------------

# Apply color based conditional formatting on the data frame
cm = sns.light_palette("seagreen", as_cmap=True)

(all_days_to_listings_rented_phaseofmonth_lvl_combined.style
  .background_gradient(cmap=cm, subset=['total_revenue','total_property_rented', 'average_revenue_per_property'])
  .highlight_max(subset=['total_revenue','total_property_rented', 'average_revenue_per_property'], color='gold')

  .format({'total_revenue': "${:,.0f}"})
  .format({'average_revenue_per_property': "${:,.2f}"})  
)

# Look at average revenue per property, then total revenue


# In[ ]:


# Analysis at the 'day of week' level

# Create a new data set for 'dayofweek' level analysis
all_days_to_listings_rented_dayofweek_lvl = all_days_to_listings_rented

# Create a 'dayofweek' column
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofweekname'] == 'Monday'), 'dayofweek'] = '1 - Monday'
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofweekname'] == 'Tuesday'), 'dayofweek'] = '2 - Tuesday'
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofweekname'] == 'Wednesday'), 'dayofweek'] = '3 - Wednesday'
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofweekname'] == 'Thursday'), 'dayofweek'] = '4 - Thursday'
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofweekname'] == 'Friday'), 'dayofweek'] = '5 - Friday'
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofweekname'] == 'Saturday'), 'dayofweek'] = '6 - Saturday'
all_days_to_listings_rented_phaseofmonth_lvl.loc[(all_days_to_listings_rented_phaseofmonth_lvl['dayofweekname'] == 'Sunday'), 'dayofweek'] = '7 - Sunday'

# # Calculate total revenue generated per 'dayofweek'
all_days_to_listings_rented_dayofweek_lvl_revenue = pd.DataFrame(all_days_to_listings_rented_dayofweek_lvl.groupby(['dayofweek'])['total_revenue_per_date'].agg(['sum']))
all_days_to_listings_rented_dayofweek_lvl_revenue.reset_index(inplace=True)
all_days_to_listings_rented_dayofweek_lvl_revenue.columns = ['dayofweek','total_revenue']

# # Calculate total number of properties rented per 'dayofweek'
all_days_to_listings_rented_dayofweek_lvl_propcount = pd.DataFrame(all_days_to_listings_rented_dayofweek_lvl.groupby(['dayofweek'])['total_properties_rented_per_date'].agg(['sum']))
all_days_to_listings_rented_dayofweek_lvl_propcount.reset_index(inplace=True)
all_days_to_listings_rented_dayofweek_lvl_propcount.columns = ['dayofweek','total_property_rented']

# # Merge both component tables, and calculate average revenue generated per property for each day of week
all_days_to_listings_rented_dayofweek_lvl_combined = pd.merge(all_days_to_listings_rented_dayofweek_lvl_revenue, all_days_to_listings_rented_dayofweek_lvl_propcount, how='left', on=['dayofweek'])
all_days_to_listings_rented_dayofweek_lvl_combined['average_revenue_per_property'] = all_days_to_listings_rented_dayofweek_lvl_combined['total_revenue']/all_days_to_listings_rented_dayofweek_lvl_combined['total_property_rented']
all_days_to_listings_rented_dayofweek_lvl_combined


# In[ ]:


# Plot the 'day of week' level summarised data for insights

x_input = all_days_to_listings_rented_dayofweek_lvl_combined['dayofweek']
y1_input = all_days_to_listings_rented_dayofweek_lvl_combined['total_property_rented']
y2_input = all_days_to_listings_rented_dayofweek_lvl_combined['total_revenue']
y3_input = all_days_to_listings_rented_dayofweek_lvl_combined['average_revenue_per_property']


# ax1
fig1, ax1 = plt.subplots()
width = 0.35

ax1 = y1_input.plot(kind='bar', width = width, color='Gold', legend=True)
ax1 = y2_input.plot(kind='line', marker='o', secondary_y=True, legend=True)
ax1.set_title('Number of properties rented v Total revenue generated for each day of week'
              , fontdict={'fontsize': 'x-large'})
ax1 = plt.gca()
ax1.set_xticks(np.arange(len(x_input)))
ax1.set_xticklabels(x_input)


# ax2
fig2, ax2 = plt.subplots()
ax2 = y3_input.plot(kind='line', marker = 'o', color='seagreen')
ax2.set_title('Average revenue generated per property for each day of week'
              , fontdict={'fontsize': 'x-large'})
ax2 = plt.gca()
ax2.set_xticks(np.arange(len(x_input)))
ax2.set_xticklabels(x_input)


# Display plots
plt.show();


# ------------------------------------------------------------------------------------------------------------------------------

# Apply color based conditional formatting on the data frame
cm = sns.light_palette("seagreen", as_cmap=True)

(all_days_to_listings_rented_dayofweek_lvl_combined.style
  .background_gradient(cmap=cm, subset=['total_revenue','total_property_rented', 'average_revenue_per_property'])
  .highlight_max(subset=['total_revenue','total_property_rented', 'average_revenue_per_property'], color='gold')

  .format({'total_revenue': "${:,.0f}"})
  .format({'average_revenue_per_property': "${:,.2f}"})  
)

# Look at average revenue per property, then total revenue


# ## **Question 3: Can we predict the property characteristics that attract the highest revenues?**
# 
# This involves predicting the traits of property that will attract high demand and generate the highest revenue. Having this insight will enable us to purchase properties with the specific traits to maximise revenue. The cells after this contain the workings for the steps below.
# 
# ### **Modelling**
#  - We are using the (all_listings_to_days_rented) dataset prepared earlier. 
#  - We select all the fields relevant to property traits and store it in a new data frame (all_listings_to_days_rented_proptrait).   
#  - We encode the categorical features, and standardize the data values.
#  - This is followed by initializing and running the prediction models and finally scoring the models, to determine if the assortment of features have predictive powers.
#  - We are using a mix of tree and linear, and the Decision Tree Regressor model performed better than the traditional and enhanced linear models. This indicates that the relationship isn't linear.
#  
# ### **Evaluation**
#  - From the Decision Tree Regressor model, we look at the features with the greatest feature importance. These features are:
#  > - bedrooms	(0.673662)
#  > - room_type_Entire home/apt	(0.214706)
#  > - accommodates	(0.054830)
#  > - bathrooms	(0.030718)
#  > - property_type_House	(0.026084)
#  - We also checked the correlations between the 'revenue per property' with all the features. The features with the highest correlations are:
#  > - bedrooms                         (0.360102)
#  > - accommodates                     (0.331536)
#  > - beds                             (0.288501)
#  > - room_type_Entire home/apt        (0.280029)
#  > - bathrooms                        (0.256875)
#  > - room_type_Private room          (-0.250865)
#  - Using both pieces of information, we can see an overlap which strengthens our confidence that these features have some form of predictive power that impacts the revenue a property can generate.
#  - We recommend that a property owners who are looking to renovate their existing property / buyers of new investment properties who are looking to rent it out via AirBnB, focus on ensuring their properties:
#  > - Have more bedrooms, beds and bathrooms, thus enabling the property to accommodate more people as these increase revenue generating potential
#  > - That are houses
#  > - Which can be rented out in its entirety, instead of a property where only a single private room can be rented out
#  
# ### **Deployment**
#  - There is no model deployment here. There will be further improvements to the predictive models in the future.
#  

# In[ ]:


# Limit the data set to only relevant fields
all_listings_to_days_rented_proptrait = all_listings_to_days_rented[[
                                                                     'property_type',
                                                                     'room_type',
                                                                     'accommodates',
                                                                     'bathrooms',
                                                                     'bedrooms',
                                                                     'beds',
                                                                     'bed_type',
                                                                     'revenue_per_property'
                                                                    ]]
all_listings_to_days_rented_proptrait.head()


# In[ ]:


# Check the data type of each field
all_listings_to_days_rented_proptrait.dtypes

# Identify the non-numeric fields that will be encoded
encode_cols = ['property_type', 'room_type', 'bed_type']

# Perform 1 hot encoding
all_listings_to_days_rented_proptrait=pd.get_dummies(data=all_listings_to_days_rented_proptrait, columns=encode_cols,drop_first=False)
# all_listings_to_days_rented_proptrait.head()

# Split data into (x)features and (y)target
y_target =  all_listings_to_days_rented_proptrait['revenue_per_property']

x_features = all_listings_to_days_rented_proptrait.drop('revenue_per_property', axis=1)


# In[ ]:


x_features.head()


# In[ ]:


# Creating a scaling feature set
x_features_scaled = x_features

# Get column names first
col_names = list(x_features_scaled.columns)

# Create the Scaler object
scaler = preprocessing.StandardScaler()

# Fit your data on the scaler object - perform standard scaling on (x)
x_features_scaled = scaler.fit_transform(x_features_scaled)
x_features_scaled = pd.DataFrame(x_features_scaled, columns=col_names)


# In[ ]:


#split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_features_scaled, 
                                                    y_target, 
                                                    test_size=0.20, 
                                                    random_state=71)


# In[ ]:


# Create a function to establish a training and testing pipeline 
# (drawn from Term 1: https://github.com/MikeBong/udacity_datascience_nd/blob/master/project_finding_donors/finding_donors-MikeBong-02.ipynb).
def train_predict(learner, sample_size, X_train, y_train, X_test, y_test): 
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: review_scores_rating training set
       - X_test: features testing set
       - y_test: review_scores_rating testing set
    '''
    results = {}
    
    #Fit the learner to the training data and get training time
    start = time() 
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time() 
    results['train_time'] = end-start
    
    # Get predictions on the test set(X_test), then get predictions on first 300 training samples
    start = time() 
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time() 
    
    # Calculate the total prediction time
    results['pred_time'] = end-start
    
    #Compute accuracy on the first 300 training samples
    results['mse_train'] = mean_squared_error(y_train[:300],predictions_train)
    
    #Compute accuracy on test set
    results['mse_test'] = mean_squared_error(y_test,predictions_test)
       
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))
    print('-'*90)
    print("MSE_train: %.4f" % results['mse_train'])
    print("MSE_test: %.4f" % results['mse_test'])
    print("Training score:%.4f" % learner.score(X_train,y_train))
    print("Test score:%.4f" % learner.score(X_test,y_test))
    print(' '*90)
    print('+'*90)

    return results


# In[ ]:


#Initialize the models
clf1 = LinearRegression()
clf2 = DecisionTreeRegressor(max_depth=4,min_samples_leaf=10,min_samples_split=10,max_leaf_nodes=8,random_state=71)
clf3 = Lasso()
clf4 = RidgeCV() 

# Calculate the number of samples for 1% and 100% of the training data
samples_100 = len(y_train)
samples_1 = int(0.01*len(y_train))

# Collect results on the learners
results = {}
for clf in [clf1, clf2, clf3, clf4]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_100]):
        results[clf_name][i] =         train_predict(clf, samples, X_train, y_train, X_test, y_test)


# In[ ]:


# From the scores generated, both models are quite poor.
# The decision tree regressor performed better than the linear regression model.

# We now extract the feature importance from the relatively better decision tree regressor model
feature_importances = pd.DataFrame(clf2.feature_importances_,
                                   index = X_train.columns,
                                    columns=['coefficient']).sort_values('coefficient', ascending=False)

# The features ranked in order of importance per the decision tree regressor can be found below.

feature_importances.sort_values(by= 'coefficient', ascending=False)

# Admittedly the models can be improved, though this provides us with a good starting point to provide preliminary recommendations to buyers and owners.


# In[ ]:


# Plot the correlation between the features and targets in the data set
# We will use this to corroborate the sub par results from the models ran above

sns.set(style="white")

# Nominate dataset
d = all_listings_to_days_rented_proptrait

# Compute the correlation matrix
corr = d.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(30, 12))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});


# In[ ]:


corr['revenue_per_property'].sort_values(ascending=False) 


# 

# **References**
# * Airbnb business model: https://jungleworks.com/airbnb-business-model-revenue-insights/
