#!/usr/bin/env python
# coding: utf-8

# ## Data Description

# Since 2008, guests and hosts have used Airbnb to expand on traveling possibilities and present more unique, personalized way of experiencing the world. This dataset describes the listing activity and metrics in NYC, NY for 2019. This dataset was obtained from New York City Airbnb Open Data on Kaggle, uploaded by Dgomonov.

# I would also like to take this opportunity to express my sincere gratitude to the Stackoverflow community, as they have been my go to resource for all my data science related queries. 

# ## Loading Libraries and Data

# In[ ]:


#Importing Libraries and reading Data

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# ## Understanding the Data Set

# In[ ]:


NYC_data = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


NYC_data.shape


# In[ ]:


NYC_data.head()


# In[ ]:


NYC_data.describe()


# In[ ]:


import matplotlib.pyplot as plt

NYC_data.hist(bins=50, figsize=(20,15))
plt.show()


# The Histogram gives us an idea of the distribution of some important features like price and availability_365. Some features like Latitude, Longitude and id do not hold much significance. Next lets look at how well correlated the different features are. 

# In[ ]:


correlation = NYC_data.corr()


# In[ ]:


import seaborn as sns
plt.figure(figsize=(18,12))
ax=sns.heatmap(correlation, annot=True, square=True, annot_kws={"size":16})


# Price does not show good correlation to any of the other features. Hence going forward to do a ML prediction might not give very accurate results.

# ## Exploratory Data Analysis

# Let us use the Latitude and Longitude information provided to us to plot a map of NYC and then try to add price information to that plot

# In[ ]:


plt.figure(figsize=(12,12))
sns.scatterplot(x="latitude", y="longitude", data=NYC_data, hue="neighbourhood_group")
plt.show()


# In[ ]:


NYC_data.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, c="price", cmap=plt.get_cmap("jet"), colorbar=True, figsize=(12,10))
plt.show()


# In the plot above, we cannot see much variation in the price with the area. This could be because of some outliers which are skewing our data. Let us check the price variable with a box plot/violin plot to confirm are intuitions. 

# In[ ]:


plt.figure(figsize=(8,6))
sns.violinplot(NYC_data["price"])
plt.xticks(fontsize=16)
plt.show()


# The violin plot confirms that the majority of our data is in the less than 500 range. Hence let us draw out a subset of the data in which the prices are below 500.

# In[ ]:


NYC_data_subset = NYC_data.loc[NYC_data["price"]<=500]


# In[ ]:


NYC_data_subset.plot(kind="scatter", x="longitude", y="latitude", alpha=0.2, c="price", cmap=plt.get_cmap("jet"), colorbar=True, figsize=(12,10))
plt.show()


# On comparing with the plot by neighbourhood_group, the prices of listings in the north west section of Brooklyn and the southern part of Manhattan seem to be highest. Let us now look at how the price compares to the neighbourhood_group using the violin plot 

# In[ ]:


plt.figure(figsize=(10,8))
ax = sns.violinplot(x="neighbourhood_group", y="price", data=NYC_data_subset)


# Let us now dive down deeper to look at the room_type variable. Firstly, it would be good to know, how much percentage of each room_type makes up the total listings. Next, it would be interesting to look at the distribution of room_type with respect to the neighbourhood_group. 

# In[ ]:


NYC_data["Listings"] = NYC_data["calculated_host_listings_count"]
NYC_data["Listings"] = NYC_data["Listings"]/NYC_data["calculated_host_listings_count"]
NYC_data["Listings"] =NYC_data["Listings"].astype("int")


# In[ ]:


NYC_data.groupby("room_type").sum()


# In[ ]:


type_info_NYC = NYC_data.groupby("room_type")
total_listings = type_info_NYC.sum()["Listings"]


# In[ ]:


accomodation = [room for room, NYC_data in type_info_NYC]

plt.figure(figsize=(12,8))
ax = sns.barplot(accomodation, total_listings)
ax.set(ylabel="Listings")
plt.rcParams["axes.labelsize"] = 20
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)
plt.show()


# In[ ]:


neighbourhood_group_listing = NYC_data.groupby("neighbourhood_group")


# In[ ]:


NYC_data.groupby("neighbourhood_group").sum()


# In[ ]:


total_listings_neighbourhood = neighbourhood_group_listing.sum()["Listings"]


# In[ ]:


neighbourhood_listings = [listing for listing, NYC_data in neighbourhood_group_listing]

plt.figure(figsize=(12,8))
sns.barplot(neighbourhood_listings, total_listings_neighbourhood)
plt.xticks(fontsize=16)
plt.yticks(fontsize=14)
plt.show()


# In[ ]:


table = pd.crosstab(NYC_data["neighbourhood_group"], NYC_data["room_type"])
table


# In[ ]:


table.div(table.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(12,8))
plt.xlabel('')
plt.ylabel('Percentage')
plt.xticks(fontsize=16, rotation=45)
plt.yticks(fontsize=14)
plt.show()


# Let us now try to extract information on which hosts have maximum listings in NYC. 

# In[ ]:


host_info = NYC_data.sort_values(by=["calculated_host_listings_count"], ascending=False)
host_info.head()


# In[ ]:


host_unique = host_info.drop_duplicates(["host_id"])
host_unique.head()


# In[ ]:


host_unique_subset = host_unique[:10]
host_unique_subset = host_unique_subset.reset_index(drop=True)


# In[ ]:


plt.figure(figsize=(12,8))
ax = sns.barplot(x="host_id", y="calculated_host_listings_count", data=host_unique_subset, order=host_unique_subset["host_id"])
ax.set(ylabel="Total Listings", xlabel="Host ID")
plt.rcParams["axes.labelsize"] = 30
plt.xticks(fontsize=14, rotation=45)
plt.yticks(fontsize=14)
plt.show()


# ## Conclusions and Findings

# Following are my conclusions and findings after analyzing the data:
# 1. room_type: Majority of the listings belonged to the Entire room/apt & Private room category. Shared Rooms were hardly listed. This observation is understandable as not many people would like to share their room with anyone. Furthermore, room_type percentage with respect to the neighbourhood_group was plotted, to give insights into the distribution
# 2. neighbourhood_group: Maximum listings came from Brooklyn & Manhattan. Queens followed third with very small contributions from Bronx and Staten Island 
# 3. host_id: Hosts with the maximum listings were identified. The maximum number of listings by a single host was found to be 327
# 4. price: Upon plotting the map, price was found to be higher in Northwest Brookyln and in southern Manhattan
# 5. The correlation matrix and the histogram unfortunately revealed lack of correlation between the price and the other features. Had the correlation been slightly stronger an attempt would have been made to carry out some Machine learning on the data and validating it by setting aside a small portion of the data as test data

# Thank you very much for going through my data analysis of the NYC Airbnb data set. In case this notebook, helped you in any manner please do consider upvoting my first upload :) 
