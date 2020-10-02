#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import string

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# For this project I am interested in understanding listing features driving higher AirBnB prices. How can AirBnB hosts bring out the best existing qualities of their listings? How can they know if their pricing is fair compared to market? 
# 
# This piece of analysis is broken down into two parts to answer the questions:
# * What is a fair price for each listing?
# * How can they bring out the best qualities of their listing through accurate description?
# 
# The summary deck to this notebook can be found here: https://www.slideshare.net/BellynOoi/nyc-airbnb-pricing-211300578

# ![](http://)**1. Data Overview and Preprocessing**

# In[ ]:


airbnb = pd.read_csv("/kaggle/input/new-york-city-airbnb-open-data/AB_NYC_2019.csv")


# In[ ]:


# print("\nData size: \n", airbnb.shape)
# print("\nData types: \n", airbnb.dtypes)
# print("\nFeatures: \n", airbnb.columns)
# print("\nMissing Values :", airbnb.isnull().sum())
# print("\nUnique Values :", airbnb.nunique())


# In[ ]:


# separate our table from original table for further manipulation
airbnb_df = airbnb

# convert date into day, month, and year columns
airbnb_df.last_review = pd.to_datetime(airbnb.last_review)
airbnb_df["last_review_day"] = airbnb_df.last_review.dt.day
airbnb_df["last_review_month"] = airbnb_df.last_review.dt.month
airbnb_df["last_review_year"] = airbnb_df.last_review.dt.year


# In[ ]:


# do all the missing values occur at the same index?
last_review_missing = pd.isnull(airbnb_df["last_review"])
reviews_per_month_missing = pd.isnull(airbnb_df["reviews_per_month"])
# print("Do all missing values occur at the same index?")
# print(np.array_equal(last_review_missing,reviews_per_month_missing))

# create a new flag to tag these listings as new listings 
airbnb_df["new_listing_flag"] = np.where(last_review_missing, 1, 0)

airbnb_df.head(10)


# Preview of the data after some light cleaning. I created a separate column to identify new listings as I want to control for the effect of new listings.

# In[ ]:


time_trend_count = airbnb_df.groupby([ "neighbourhood_group","last_review_year"])["id"].count().reset_index()
time_trend_count.set_index(["last_review_year"]).reset_index().sort_values(by=["neighbourhood_group","last_review_year"])

time_trend_count = pd.DataFrame(time_trend_count)
time_trend_count = pd.pivot_table(time_trend_count,values="id",columns="neighbourhood_group",index="last_review_year").fillna(value=0)

plt.figure(figsize=(16, 10))

barWidth = 0.15  # the width of the bar
# set height of bar
bars1 = time_trend_count["Bronx"].tolist()
bars2 = time_trend_count["Brooklyn"].tolist()
bars3 = time_trend_count["Manhattan"].tolist()
bars4 = time_trend_count["Staten Island"].tolist()
bars5 = time_trend_count["Queens"].tolist()

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

plt.bar(r1, bars1, color='brown', width=barWidth, edgecolor='white', label='Bronx')
plt.bar(r2, bars2, color='blue', width=barWidth, edgecolor='white', label='Brooklyn')
plt.bar(r3, bars3, color='red', width=barWidth, edgecolor='white', label='Manhattan')
plt.bar(r4, bars4, color='green', width=barWidth, edgecolor='white', label='Staten Island')
plt.bar(r5, bars5, color='yellow', width=barWidth, edgecolor='white', label='Queens')

plt.xlabel('Year', fontweight='bold')
plt.ylabel('Number of listings', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], time_trend_count.index)

plt.legend()
plt.show()


# For the rest of the analysis, I want to focus on active listings. Based on the distribution of the data I set the assumption that "last_review_year" older than 2018 are inactive listings. 

# In[ ]:


airbnb_df = airbnb_df[ (airbnb_df["last_review_year"] > 2017) | (airbnb_df["new_listing_flag"] == 1)]


# **2. Analysis**

# In[ ]:


price_by_neighbourhood = airbnb_df.groupby(["neighbourhood_group","new_listing_flag"])["price"].agg({'Count':'count','Min': 'min',
                             'Max': 'max', 'Median':'median', 'Std':'std'})

price_by_neighbourhood = price_by_neighbourhood.sort_values(by=["neighbourhood_group","new_listing_flag"], ascending = False).reset_index()

# Grouped boxplot
df = pd.DataFrame(airbnb_df[["price","neighbourhood_group", "new_listing_flag"]])
plt.figure(figsize=(16, 10))
sns.boxplot(x="neighbourhood_group", y="price", hue="new_listing_flag", data=df, palette="Set1", showfliers=False)


# First I want to get a sense of scale of the Airbnb listing space. Here we see that Manhattan unsurprisingly has the most expensive listings, followed by Brooklyn. New listings are overall more expensive than existing listings, and this difference is more pronounced in Manhattan and Staten Island.

# In[ ]:


# Within each neighborhood group, which are the most expensive?

neighborhood = airbnb_df.groupby(["neighbourhood_group","neighbourhood"])["price"].agg({'Count':'count','Min': 'min',
                             'Max': 'max', 'Median':'median', 'Std':'std'})

neighborhood[neighborhood["Count"]>10].sort_values(by="Median", ascending = False).sort_values(by="neighbourhood_group").groupby("neighbourhood_group").head(5)


# Each borough has its pricier neighbourhoods. In Manhattan, these are usually the business/ tourist districts. In Brooklyn, these are the more upscale neighbourhoods with close proximity to Manhattan, access to a main subway station and the waterfront.  

# In[ ]:


# Popularity of listing
number_of_reviews_bins = [1,10,20,40,60,80,100,float("inf")]
number_of_review_labels = ["<10","10-19","20-39","40-59","60-79","80-99",">100"]
airbnb_df['binned_reviews'] = pd.cut(airbnb_df['number_of_reviews'], bins = number_of_reviews_bins, labels = number_of_review_labels)
airbnb_df[["number_of_reviews", "binned_reviews"]].dropna().groupby("binned_reviews").count()


# In[ ]:


reviews = airbnb_df.groupby("binned_reviews")["price"].agg({'Count':'count','Min': 'min',
                             'Max': 'max', 'Median':'median', 'Std':'std'})

# Grouped boxplot
reviews_boxplot = pd.DataFrame(airbnb_df[["binned_reviews","price"]])
plt.figure(figsize=(16, 10))
sns.boxplot(x="binned_reviews", y="price", data=reviews_boxplot, palette="Set1", showfliers=False)


# Most airbnb listings have less than 10 cumulative reviews. The number of reviews per listing doesn't seem to affect the price.
# 

# In[ ]:


room_trend = airbnb_df[airbnb_df["new_listing_flag"] == 0].groupby([ "neighbourhood_group","room_type"])["price"].median().reset_index()
room_trend.set_index(["room_type"]).reset_index().sort_values(by=["neighbourhood_group","room_type"])

room_trend = pd.DataFrame(room_trend)
room_trend = pd.pivot_table(room_trend,values="price",columns="neighbourhood_group",index="room_type").fillna(value=0)

plt.figure(figsize=(16, 10))

barWidth = 0.15  # the width of the bar
# set height of bar
bars1 = room_trend["Bronx"].tolist()
bars2 = room_trend["Brooklyn"].tolist()
bars3 = room_trend["Manhattan"].tolist()
bars4 = room_trend["Staten Island"].tolist()
bars5 = room_trend["Queens"].tolist()

# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]
r5 = [x + barWidth for x in r4]

plt.bar(r1, bars1, color='brown', width=barWidth, edgecolor='white', label='Bronx')
plt.bar(r2, bars2, color='blue', width=barWidth, edgecolor='white', label='Brooklyn')
plt.bar(r3, bars3, color='red', width=barWidth, edgecolor='white', label='Manhattan')
plt.bar(r4, bars4, color='green', width=barWidth, edgecolor='white', label='Staten Island')
plt.bar(r5, bars5, color='yellow', width=barWidth, edgecolor='white', label='Queens')

plt.xlabel('Room Type', fontweight='bold')
plt.ylabel('Price', fontweight='bold')
plt.xticks([r + barWidth for r in range(len(bars1))], room_trend.index)

plt.legend()
plt.show()


# We see that entire homes fetch a higher price than semi-private spaces. But the difference between boroughs is again pronounced here. For the same price as a private room in Manhattan, you can get a whole apartment if you move to the Bronx or Staten Island.[](http://)

# ** 3. Listing Description Analysis **
# 
# I'm interested in understanding the qualitative features that set prices apart. What are some features that makes listings more desirable? 
# 
# To control for the difference between boroughs, we'll look into the features within each borough. We'll also dive deeper by controlling for other features such as type of room and the overall popularity of the listing.

# In[ ]:


## BAG OF WORDS APPROACH ##
# check the top word descriptions by prices
import nltk 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 

# function to clean up the descriptions
def filter_sentence(description):
    filtered_sentence = [w.lower() for w in description if not w in stop_words and not w in string.punctuation and w.isalpha()] 
    filtered_sentence = [w.replace("bd","bedroom") for w in filtered_sentence] 
    filtered_sentence = [w.replace("bedrooms","bedroom") for w in filtered_sentence]
    filtered_sentence = [w.replace("bath", "bathroom") for w in filtered_sentence]
    filtered_sentence = [w.replace("bathrooms", "bathroom") for w in filtered_sentence]
    return filtered_sentence

airbnb_df["name"] = ["" if isinstance(x, float) else x for x in airbnb_df["name"] ]

# create a new column for the cleaned up descriptions
airbnb_df["description"] = airbnb_df["name"].apply(word_tokenize)
stop_words = set(stopwords.words('english')) 
airbnb_df["description"] = airbnb_df["description"].apply(filter_sentence)

# create a new row for each word in the description 
airbnb_df_expand = airbnb_df.explode("description")
airbnb_df_expand["description"] = [x if isinstance(x, str) else "" for x in airbnb_df_expand["description"]]
print(airbnb_df_expand.groupby("description")["description"].count().sort_values(ascending=False))


# In[ ]:


# set cutoff to exclude statistically insignificant words
cutoff = 50

# overall most expensive listings, without controlling for anything
description_price = airbnb_df_expand.groupby("description")["price"].agg({'Count':'count','Min': 'min',
                             'Max': 'max', 'Median':'median', 'Std':'std'})

description_price = description_price.loc[description_price["Count"] > 50].sort_values(by="Median", ascending = False).head(10)

description_price


# After some cleaning of the description column, we end up with a list of the top words present in the most expensive listings, evaluated by its frequency (measured by count).

# In[ ]:


# various cuts controlling for selected features
segment_list = ["room_type", "neighbourhood_group", "binned_reviews"]
segment_df = []
for s in segment_list:
    description_segment = airbnb_df_expand.groupby([s,"description"])["price"].agg({'Count':'count','Min': 'min',
                             'Max': 'max', 'Median':'median'})
    description_segment = description_segment.loc[description_segment["Count"] > cutoff].sort_values(by=[s,"Median"], ascending = False).groupby(s).head(5)
    segment_df.append(description_segment)


# In[ ]:


# view by each segment
display(segment_df[segment_list.index("neighbourhood_group")],segment_df[segment_list.index("room_type")])


# We can also see this list by neighbourhood and room type.
