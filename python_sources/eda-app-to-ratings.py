#!/usr/bin/env python
# coding: utf-8

# # Background
# As mentioned by tristan 581, the data were collected from the Apple App Store through use of the iTunes API and the App Store sitemap. We are informed that the dataset consists of 17007 strategy games. 
# 
# **The Goal:** In this notebook, we will perform EDA towards the goal of understanding how the variables relate to the rating.
# 
# # Beginning

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import scipy.stats as stats
# Any results you write to the current directory are saved as output.


# In[ ]:


#read in and examine 
df_raw = pd.read_csv('/kaggle/input/17k-apple-app-store-strategy-games/appstore_games.csv')
df = df_raw
df.head()


# In[ ]:


df.info()


# * The variables (columns) are all descriptively titled. 
# * It appears that the variables [URL] and [ID] both form unique identification columns. 
# * We opt to drop the [URL] variable. 
# * [User Rating Count] and [Average User Rating] are missing a large number of values perhaps because the data simply doesn't exist. 
# * [Subtitle] and [In-app Purchases] are missing a large number of values perhaps because the features don't exist. 
# * Given that this data is a collection of strategy games, the Primary Genre column would probably not be of much use to us here. 
# * We will also drop the [Icon URL] and examine it another time.
# * It is unfortunate to see that the number of downloads for the apps is not recorded in this dataset. 

# In[ ]:


df = df.drop(["URL", "Icon URL"], axis=1)


# In[ ]:


df["Average User Rating"].value_counts()


# We see that despite its numeric appearance, [Average User Rating] can be treated as an ordered categorical variable. We can assume NaN values represent unrated games, hence since our goal is to analyze rated games, we will remove this from our data set. From experience, we also know that Paid and Free games are commonly separated in the App Store. It stands to reason that we separate them here as well.

# In[ ]:


df = df[~df["Average User Rating"].isna()]


# In[ ]:


df["Primary Genre"].value_counts()


# It appears that [Primary Genre] contains several non-gaming genres, However they are very few and hence we will not consider this variable for this analysis

# In[ ]:


df = df.drop("Primary Genre", axis=1)


# In[ ]:


df.head()


# We have a couple of numerical variables available to us. We briefly examine them here

# In[ ]:


plt.figure(figsize=(16,15))
tmp = df[["User Rating Count", "Price", "Size"]]
for j in range(3):
    plt.subplot(3,2,2*j+1)
    sns.distplot(tmp[tmp.columns[j]])
    plt.title(tmp.columns[j] + " Histogram")
    plt.subplot(3,2,2*j+2)
    sns.boxplot(y=tmp[tmp.columns[j]])
    plt.title(tmp.columns[j] +" Boxplot")


# We can see that all of our quant variables are very heavily right-skewed.

# # Feature Engineering
# 
# We can create several variables for this dataset.
# 1. [Is Paid]: 1 if [Price] > 0, 0 otherwise
# 2. [Has_Subtitle]: 1 if [Subtitle] is not missing 0 otherwise
# 3. [Num_Languages]: The number of languages a game has
# 4. [Description_Length]: The number of characters in the description
# 5. [log_Size]: log([Size])
# 6. [num_Genres]: number of genres
# 7. [Age_of_App]: Today - [Original Release Date]
# 8. [Time_Since_Update]: Today - [Current Version Release]
# 

# In[ ]:


df["is_Paid"] = df.Price.map({0:0}).fillna(1)
df["has_Subtitle"] = df.Subtitle.fillna(0).map({0:0}).fillna(1)
df["num_Languages"] = df.Languages.str.count(",")+1
df["description_Length"] = df.Description.str.len()
df["log_Size"] = np.log(df.Size)
df["num_Genres"] = df.Genres.str.count(",")+1
df["has_In-app_Purchases"] = df["In-app Purchases"].fillna(0).map({0:0}).fillna(1)
df["age_of_App"]=(pd.Timestamp.today() - pd.to_datetime(df["Original Release Date"])).dt.days
df["time_Since_Update"]=(pd.Timestamp.today() - pd.to_datetime(df["Current Version Release Date"])).dt.days


# In[ ]:


df.head()


# In[ ]:


plt.figure(figsize=(36,28))
plt.subplot(431)
sns.countplot(df["Average User Rating"])
plt.title("Average User Rating Bar")
plt.subplot(432)
sns.countplot(df["is_Paid"])
plt.title("is_Paid Bar")
plt.subplot(433)
sns.countplot(df["has_In-app_Purchases"])
plt.title("has_In-app_Purchases Bar")

plt.subplot(434)
sns.countplot(df["Age Rating"])
plt.title("Age Rating Bar")

plt.subplot(435)
sns.distplot(df["log_Size"])
plt.title("log_size Histogram")

plt.subplot(436)
sns.distplot(df["description_Length"])
plt.title("description_Length Histogram")

plt.subplot(437)
sns.distplot(df["num_Languages"].dropna())
plt.title("num_Languages Histogram")

plt.subplot(438)
sns.countplot(df["has_Subtitle"])
plt.title("has_Subtitle Bar")

plt.subplot(439)
sns.countplot(df["num_Genres"].dropna())
plt.title("num_Genres Histogram")

plt.subplot(4,3,10)
sns.distplot(df["age_of_App"].dropna())
plt.title("afe_of_App Histogram")

plt.subplot(4,3,11)
sns.distplot(df["time_Since_Update"].dropna())
plt.title("time_Since_Update Histogram");


# 1. It seems overall, users are pretty generous in their ratings
# 2. There are many more free games than paid games however, it is also the case that many free games contain in-app purchases, as indicated in the plots
# 3. It seems also that most games are rated for ages 4+
# 4. Our log size seems to be working nicely, and perhaps there is something we can do with the description length.
# 5. As can be expected, very few games have a lot of languages listed
# 6. More games than expected have a subtitle. Perhaps developers know something about subtitles and downloads or ratings.
# 7. Again, from a marketing standpoint, it makes sense to list many genres on your app so it shows up in more places.
# 8. It seems that most apps are less than six years old, which was also about the time IOS 7, and the age rating variable were released.
# 9. Many games have been updated within the last year

# # Visual Relationships with User Ratings

# In[ ]:


plt.figure(figsize = (24,7))
plt.subplot(121)
tbl = pd.crosstab(df.is_Paid,df["Average User Rating"])
tbl = (tbl.T/tbl.T.sum(axis=0)).T
sns.heatmap(tbl, cmap = 'plasma',square=True)
plt.subplot(122)
for_bar = tbl.reset_index().melt(id_vars=["is_Paid"])
sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["is_Paid"])


# The barplot seems to indicate a small relationship between [is_Paid] and [Average User Rating]. In particular, a higher proportion of unpaid games is rated at 4.5 or above compared to paid games. This is a hypothesis that can be tested using a z-test in the analysis section.

# In[ ]:


plt.figure(figsize = (24,7))
plt.subplot(121)
tbl = pd.crosstab(df["has_In-app_Purchases"],df["Average User Rating"])
tbl = (tbl.T/tbl.T.sum(axis=0)).T
sns.heatmap(data=tbl, cmap = 'plasma',square=True)
plt.subplot(122)
for_bar = tbl.reset_index().melt(id_vars=["has_In-app_Purchases"])
sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["has_In-app_Purchases"])


# Interestingly we see a high proportion of games that have in-app purchases rated consistently higher than those that do not. This gives us another hypothesis to test later

# In[ ]:


plt.figure(figsize = (24,7))
plt.subplot(121)
tbl = pd.crosstab(df["Age Rating"],df["Average User Rating"])
tbl = (tbl.T/tbl.T.sum(axis=0)).T
sns.heatmap(data=tbl, cmap = 'plasma',square=True)
plt.subplot(122)
for_bar = tbl.reset_index().melt(id_vars=["Age Rating"])
sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["Age Rating"])


# In[ ]:


plt.figure(figsize = (24,7))
plt.subplot(121)
tbl = pd.crosstab(df["has_Subtitle"],df["Average User Rating"])
tbl = (tbl.T/tbl.T.sum(axis=0)).T
sns.heatmap(data=tbl, cmap = 'plasma',square=True)
plt.subplot(122)
for_bar = tbl.reset_index().melt(id_vars=["has_Subtitle"])
sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["has_Subtitle"])


# In[ ]:


plt.figure(figsize = (24,7))
plt.subplot(121)
tbl = pd.crosstab(df["num_Genres"],df["Average User Rating"])
tbl = (tbl.T/tbl.T.sum(axis=0)).T
sns.heatmap(data=tbl, cmap = 'plasma',square=True)
plt.subplot(122)
for_bar = tbl.reset_index().melt(id_vars=["num_Genres"])
sns.barplot(x =for_bar["Average User Rating"], y = for_bar["value"], hue = for_bar["num_Genres"])


# In[ ]:


plt.figure(figsize=(36,14))
plt.subplot(231)
sns.boxplot(y=df.log_Size,x=df["Average User Rating"])
plt.subplot(232)
sns.boxplot(y=df["num_Languages"],x=df["Average User Rating"])
plt.subplot(233)
sns.boxplot(y=df["description_Length"],x=df["Average User Rating"])
plt.subplot(234)
sns.boxplot(y=df["age_of_App"],x=df["Average User Rating"])
plt.subplot(235)
sns.boxplot(y=df["time_Since_Update"],x=df["Average User Rating"])


# For both the age of the app and the time since the last update there seem to be a relation with lower values and a higher rating.

# From our exploratory analysis, we are able to gain a decent understanding of our dataset and how the variables relate to user ratings. 
