#!/usr/bin/env python
# coding: utf-8

# # Geocoding locations and localizing time
# 
# This notebook illustrates the impact of localized time on features which are based on the **time of the day**. It consists of two main sections
# 1. A recap of the effect and EDA visualization based on the Google Analytics data challenge
# 2. A walkthrough of the code and key points using
#     - geocoded geographical data from Google maps API
#     - localized UTC time usiffromromng pycountry_convert and pytz
# 
# In the context of the Google Analytics Kaggle challenge, localizing the UTC time data from visitStartTime shows a significant impact from an EDA point of view (impact on LB to be confirmed) 
# 
# --------------------------------------------
# 
# # In a nutshel
# Since sessions are recorded in [UTC time ][1] and users live in different time zones, any constructed feature such as *hour of the day* based on UTC time will not be representative of the local hour at which the event took place. The local hour of the day can be useful in a number of situations, for example when predicting online transactions in the Google Analytics challenge. 
# 
# We would expect the number of sessions to reflect patterns of every day life, with reduced frequency during the night, and a peak during day time. However when comparing the same plots for UTC and Local time we see that:
# 
# - **In UTC time**: the number of sessions vs hour of the day shows small variation between night and day time
# - **In local time**: the same plot shows significantly greater variation between night time and day time
# 
# Depending on the geographical distribution of users in the data set, the effect of their respective time zones may cancel out, producing a more "flat" distribution over hours of the day. Pushing this intuition to the extreme, a uniform distribution around the globe would lead to a nearly flat plot of frequency across hours of the day. 
# 
# See below for various plots of sessions and revenues for UTC and local time, both for the training and testing data sets. 
# 
# -----------------------------------------
# 
# 
# 
# 
# [1]: https://en.wikipedia.org/wiki/Coordinated_Universal_Time

# In[ ]:


from IPython.display import Image
url = 'https://i.imgur.com/KYoRYVX.png'
Image(url,width=1000, height=1000)


# In[ ]:


url = 'https://i.imgur.com/Q3gQ9sA.png'
Image(url,width=1000, height=1000)


# # Code and walkthrough
# 
# ## General approach
# 
# The main issue here is that the time zone library requires a specific input format of the type**tz**, see [here][1] for a list. The good news is that the Google Maps geocoding API provides all the necessary information (and much more). Moreover, google maps can work with inconsistent and even contradictory search terms, which makes our life much easier. 
# 
# Once we have the required tz information for each line, we can use the pytz library to create a locally time zone aware column, and then extract hours and other local time features 
# 
# **The approach can be summarised as follows:**
# 
# - Concatenate the train and test sets into a single data set (size > 1M rows)
# - Generate a "search term" column which will be passed to the Google Maps API 
# - Keep only the unique search term values (size = 2 - 3k)
# - This search term will also act as a foreign key to help us map the geocoded information and the original data sets
# - Once each line has a timezone, use pytz to generate a locally aware time column
# - Extract hour of the day
# 
# [1]: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
# 
# 
# ## Step 1) Geocoding the geographical information using Google Maps API
# 
# Before using the API you will need to sign up on Google API and generate an API key, this [tutorial][2] can be of use. 
# 
# **Attention:** The API is free as long as the cost of requests is below 300 $ per month. For this reason, make sure that you set quota limits on the number of requests per day, for example 5,000 or 10,000.
# 
# The actual API call proceeds in a few steps:
# 1. Call the *geocode* API to find the geocoded information corresponding to the search term
# 2. Extract the latitude and longitude from the returned info
# 3. Call the *timezone* API to obtain the corresponding time zone
# 
# 
# [2]: https://elfsight.com/blog/2018/06/how-to-get-google-maps-api-key-guide/

# In[ ]:


import os
import numpy as np
import pandas as pd
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from pycountry_convert import ( map_countries, country_name_to_country_alpha3,country_name_to_country_alpha2)
import pytz as pytz
import datetime


os.chdir("/Users/User/Desktop/Data/Kaggle/Google_Analytics/input")
pd.options.display.max_columns = None
warnings.filterwarnings('ignore')


# ### Example API call 

# In[ ]:


import googlemaps
from datetime import datetime

#Setting up API key
gmaps = googlemaps.Client(key='put_your_API_KEY_here')

###EXAMPLE
# Geocoding an address (API CALL)
geocode_result = gmaps.geocode('1600 Amphitheatre Parkway, Mountain View, CA')

#Extracting timezone from latitude and longitude (API CALL)
timez = gmaps.timezone(location = geocode_result[0]['geometry']['location'])


# ### Preparing data and search term

# In[ ]:


#Utility function
def remove_missing_vals(x):
    remove_list = ['(not set)', 'not available in demo dataset','unknown.unknown']
    if x in remove_list:
        return ''
    else:
        return x 

#Concatenate train and test set
all_df = pd.concat([train_df, test_df], axis = 0)

#Subset only geographical columns
all_geo = all_df[GEOGRAPHICAL_COLUMNS]

#Remove "Not set" and other inconsistent values
all_geo = all_geo.applymap(remove_missing_vals)

#Concatenate city, region, country into a single column
all_city_region_country = all_geo['geoNetwork.city'] + ' ' + all_geo['geoNetwork.region'] + ' ' + all_geo['geoNetwork.country']

#Keep only unique values
all_city_region_country = all_city_region_country.unique()
print('Number of unique values:', all_city_region_country.shape)

#delete for memory
#del all_geo


# ## API calls

# In[ ]:


from pandas.io.json import json_normalize
import json

#Initialize empty df
geocode_df = pd.DataFrame()

for search_term in all_city_region_country:
#Note this can be slow (30 - 45 mins for 3k lines)   
    try:
        # Geocoding an address
        geocode_result = gmaps.geocode(search_term)

        #Extracting timezone from latitude and longitude
        timezone_result = gmaps.timezone(location = geocode_result[0]['geometry']['location'])

        #Normalize the result so that we can work with it as a df
        temp_df = json_normalize(geocode_result)
        temp_df['search_term'] = search_term
        temp_df['timeZoneId'] = timezone_result['timeZoneId']
        geocode_df = geocode_df.append(temp_df, ignore_index = True)
    
    except:
        pass


# ### Utility functions 

# In[ ]:


def time_zone_converter(x):
    try:
        return pytz.country_timezones(x)[0]
    except AttributeError:
        return np.nan
    
def time_localizer(s):
    #format of series [time,zone]
    try:
        tz =pytz.timezone(s[1])
        return pytz.utc.localize(s[0], is_dst=None).astimezone(tz)
    except:
        return np.nan
    
def map_timezone(x):   
    try:
        return timezone_dict[x]
    except KeyError:
        return 'UTC'


# ### Saving / Loading 

# In[ ]:


get_ipython().run_cell_magic('time', '', "##Load geocoded information\ngeocode_df= pd.read_csv('geocodes_timezones.csv')\n#train_df = pd.read_pickle('train_flat.pkl')\n#test_df = pd.read_pickle('train_flat.pkl')")


# ## Step 2) Using the time zone information wity pytz to localize time 

# In[ ]:


#Generate foreign key '_search_term' by concatenating city, region, country
train_df['_search_term'] = train_df['geoNetwork.city'].map(remove_missing_vals) + ' ' + train_df['geoNetwork.region'].map(remove_missing_vals) + ' ' + train_df['geoNetwork.country'].map(remove_missing_vals)
test_df['_search_term'] = test_df['geoNetwork.city'].map(remove_missing_vals) + ' ' + test_df['geoNetwork.region'].map(remove_missing_vals) + ' ' + test_df['geoNetwork.country'].map(remove_missing_vals)

#Set global variable, needed for map_timezone function
global timezone_dict
timezone_dict = dict(zip(geocode_df['search_term'], geocode_df['timeZoneId']))


#Map timezones
train_df['_timeZoneId'] = train_df['_search_term'].map(map_timezone)
test_df['_timeZoneId'] = test_df['_search_term'].map(map_timezone)

  
#Create time zone aware column
train_df['_local_time'] = train_df[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1).astype(str)
test_df['_local_time'] = test_df[['visitStartTime', '_timeZoneId']].apply(time_localizer, axis = 1).astype(str)  

#Localize hour time
train_df['_local_hourofday'] = train_df['_local_time'].str[11:13]
test_df['_local_hourofday'] = test_df['_local_time'].str[11:13]


# # Plotting UTC and local hour of the day 

# In[ ]:


#Creating a df with visitstarttime as the index
sub_cols = ['fullVisitorId', 'sessionId', 'visitId','visitStartTime', 
             '_local_time', '_timeZoneId', '_local_hourofday', 'totals.transactionRevenue' ]


train_ts = train_df[sub_cols].copy()
test_ts =  test_df[sub_cols].copy()
train_ts.index = train_ts['visitStartTime']
test_ts.index = test_ts['visitStartTime']

train_ts['_utc_hourofday'] = train_ts.index.hour
test_ts['_utc_hourofday'] = test_ts.index.hour


# ### Sessions per hour of day (UTC vs Local time) 

# In[ ]:


df1 = train_ts.groupby('_utc_hourofday').count()['sessionId']
df2 = train_ts.groupby('_local_hourofday').count()['sessionId']
df3 = test_ts.groupby('_utc_hourofday').count()['sessionId']
df4 = test_ts.groupby('_local_hourofday').count()['sessionId']

plt.figure(figsize = (15,15))
plt.subplot(2,2,1)

sns.barplot(x = df1.index ,y = df1.values, color = 'darkblue', alpha = .6)
plt.title('Sessions per hour of day (Training UTC)')

plt.subplot(2,2,2)
sns.barplot(x = df2.index ,y = df2.values, color = 'darkblue', alpha = .6)
plt.title('Sessions per hour of day (Training Local)')

plt.subplot(2,2,3)
sns.barplot(x = df3.index ,y = df3.values, color = 'darkred', alpha = .6)
plt.title('Sessions per hour of day (Test UTC)')

plt.subplot(2,2,4)
sns.barplot(x = df4.index ,y = df4.values, color = 'darkred', alpha = .6)
plt.title('Sessions per hour of day (Test Local)')


plt.show()


# ### Revenues per hour of day (UTC vs local)

# In[ ]:


df1 = train_ts.groupby('_utc_hourofday').sum()['totals.transactionRevenue']
df2 = train_ts.groupby('_local_hourofday').sum()['totals.transactionRevenue']
df3 = test_ts.groupby('_utc_hourofday').sum()['totals.transactionRevenue']
df4 = test_ts.groupby('_local_hourofday').sum()['totals.transactionRevenue']

plt.figure(figsize = (15,15))

plt.subplot(2,2,1)
sns.barplot(x = df1.index ,y = df1.values, color = 'darkblue', alpha = .6)
plt.title('Revenues per hour of day (Train UTC) ')

plt.subplot(2,2,2)
sns.barplot(x = df2.index ,y = df2.values, color = 'darkblue', alpha = .6)
plt.title('Revenues per hour of day (Train Local)')


plt.subplot(2,2,3)
sns.barplot(x = df3.index ,y = df3.values, color = 'darkred', alpha = .6)
plt.title('Revenues per hour of day (Test UTC)')

plt.subplot(2,2,4)
sns.barplot(x = df4.index ,y = df4.values, color = 'darkred', alpha = .6)
plt.title('Revenues per hour of day (Test Local)')


plt.show()

