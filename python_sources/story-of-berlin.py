#!/usr/bin/env python
# coding: utf-8

# # 1. Load Data

# In[ ]:


import pandas as pd
import numpy as np

import os
print(os.listdir("../input/berlin-airbnb-data/"))


# In[ ]:


listings = pd.read_csv("../input/berlin-airbnb-data/listings.csv", index_col= "id")
listings_summary = pd.read_csv("../input/berlin-airbnb-data/listings_summary.csv", index_col= "id")
calendar_summary = pd.read_csv("../input/berlin-airbnb-data/calendar_summary.csv", parse_dates=['date'], index_col='listing_id')
reviews = pd.read_csv("../input/berlin-airbnb-data/reviews.csv", parse_dates=['date'], index_col='listing_id')
reviews_summary = pd.read_csv("../input/berlin-airbnb-data/reviews_summary.csv", parse_dates=['date'], index_col='id')


# In[ ]:


print(listings.shape)


# In[ ]:


listings.head()


# In[ ]:


listings_summary.info()


# Let's combine listing and summary together while selecting the useful columns

# In[ ]:


listings_summary.columns


# In[ ]:


target_columns = ["property_type", "accommodates", "first_review", "review_scores_value", "review_scores_cleanliness", 
                  "review_scores_location", "review_scores_accuracy", "review_scores_communication", "review_scores_checkin",
                  "review_scores_rating", "maximum_nights", "listing_url", "host_is_superhost", "host_about", "host_response_time",
                  "host_response_rate", "street", "weekly_price", "monthly_price", "market"]
listings = pd.merge(listings, listings_summary[target_columns], on='id', how='left')
listings.info()


# In[ ]:


listings['host_response_rate'] = pd.to_numeric(listings.host_response_rate.str.strip('%'))
listings.head()


# # 2. EDA

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
sns.set_palette(palette='deep')
import folium
from folium.plugins import FastMarkerCluster


# In[ ]:


freq = listings['neighbourhood_group'].value_counts().sort_values(ascending=True)
freq.plot.barh(figsize=(10, 8), width=1)
plt.title("Number of Listings by Neighbourhood Group")
plt.xlabel('Number of Listings')
plt.show()


# Let's see this population on the map.

# In[ ]:


lat = listings['latitude'].tolist()
lon = listings['longitude'].tolist()
locations = list(zip(lat, lon))

map1 = folium.Map(location=[52.5200, 13.4050], zoom_start=12)
FastMarkerCluster(locations).add_to(map1)
map1


# In[ ]:


listings.property_type.unique()


# In[ ]:


listings.room_type.unique()


# ## Room and Property Type

# In[ ]:


freq = listings['room_type'].value_counts().sort_values(ascending=True)
freq.plot.barh(figsize=(10, 5), width=1)
plt.show()


# We see that private rooms much common than apartments. This is due to WG system in Germany.

# In[ ]:


freq = listings['property_type'].value_counts().sort_values(ascending=True)
freq = freq[freq > 20]  # Eliminate types less than 20 counts.
freq.plot.barh(figsize=(15, 8), width=1)
plt.xscale('log')
plt.show()


# ## Number of people per Listings

# In[ ]:


freq = listings['accommodates'].value_counts().sort_index()
freq.plot.bar(figsize=(12, 8), width=1, rot=0)
plt.title("Number of People")
plt.ylabel('Number of Listings')
plt.xlabel('Accommodates')
plt.show()


# # Renting Prices
# 
# Places for 2 people are very common in Berlin. Let's imagine you are going to Berlin and want to rent an apartment. Where should you look at?

# In[ ]:


freq = listings[listings['accommodates']==2]
freq = freq.groupby('neighbourhood_group')['price'].mean().sort_values(ascending=True)
freq.plot.barh(figsize=(12, 8), width=1)
plt.title("Average Daily Price for 2 People")
plt.xlabel('Average Daily Price (Dollar)')
plt.ylabel("Neighbourhodd")
plt.show()


# ### Finding a Super Host

# In[ ]:


listings.host_is_superhost = listings.host_is_superhost.replace({"t": "True", "f": "False"})
freq=listings['host_is_superhost'].value_counts()
freq.plot.bar(figsize=(10, 8), width=1, rot=0)
plt.title("Number of Listings with Superhost")
plt.ylabel('Number of Listings')
plt.show()


# ### Eliminating Bad Hosts
# 
# We can check their response rate and time 

# In[ ]:


listings10 = listings[listings['number_of_reviews']>=10]
fig = plt.figure(figsize=(16,10))

ax = fig.add_subplot(121)
freq = listings10['host_response_rate'].dropna()
freq.plot.hist('host_response_rate', ax=ax)
plt.title("Response Rate")
plt.ylabel("number of listings")
plt.xlabel("Percent")

ax = fig.add_subplot(122)
freq = listings10['host_response_time'].value_counts()
freq.plot.bar(width=1, rot=45, ax=ax)
plt.title("Response Time")
plt.ylabel("Number of Listings")

plt.tight_layout()
plt.show()


# The calendar file holds 365 records for each listing, which means that for each listing the price and availablity by date is specified 365 days ahead. Let's see the future prices

# In[ ]:


calendar_summary.head()


# Note: availability is f or FALSE means that either the owner does not want to rent out his property on the specific date, or the listing has been booked for that date already.

# In[ ]:


calendar_summary.price = calendar_summary.price.str.replace(",","")
calendar_summary.price = pd.to_numeric(calendar_summary.price.str.strip('$'))
calendar_summary = calendar_summary[calendar_summary.date < '2019-12-30']

listings.index.name = "listing_id"
calendar = pd.merge(calendar_summary, listings[['accommodates']], on="listing_id", how="left")
calendar.sample(10)


# In[ ]:


sum_available = calendar[calendar.available == "t"].groupby(['date']).size().to_frame(name='available').reset_index()
sum_available = sum_available.set_index('date')

sum_available.plot(kind='line', y='available', figsize=(12, 8))
plt.title('Number of Listings Available by Date')
plt.xlabel('Date')
plt.ylabel('Number of Listings Available')


# This graph tells us that on February not a lot of people are travelling and therefore number of available listings are big. It may be an optimal time to travel for a cheap accomadation prices!
