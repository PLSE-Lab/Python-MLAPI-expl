#!/usr/bin/env python
# coding: utf-8

# # Introduction:
# 
# Hello, and thanks for browsing my EDA of US YouTube Data.
# 
# Here are some questions we'll explore:
# 
# * **Does it matter when you publish?** Relationship between time published and views/days trended
# * **Do YouTubers who post more often trend more days?** Relationship between how often videos are published and days trended
# * **What qualities matter?** Correlations between some qualities of the data
# * **Are videos of certain categories more likely to trend, or do some types of videos trend longer?** Relationship between category of video and views/days trended

# # Cleaning and preparing the data

# In[ ]:


import kaggle
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import json
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


sns.set_style("darkgrid")
sns.set_context("paper",font_scale=1)
plt.rcParams['figure.figsize'] = (10, 10)


# In[ ]:


data = pd.read_csv("../input/USvideos.csv")


# In[ ]:


data.head()


# In[ ]:


# Infomation about the data
data.info()


# In[ ]:


# Unique objects?

data["video_id"].nunique()


# ### Process the dates - currently in strings

# In[ ]:


type(data["trending_date"][0])


# In[ ]:


data["trending_date"] = pd.to_datetime(data["trending_date"],format="%y.%d.%m")
data["trending_date"].head()


# In[ ]:


type(data["publish_time"][0])


# In[ ]:


data["publish_time"] = pd.to_datetime(data["publish_time"],format="%Y-%m-%dT%H:%M:%S.%fZ")
data["publish_time"].head()


# In[ ]:


# Separate date from time for publish

data["publish_date"] = pd.DatetimeIndex(data["publish_time"]).date
data["publish_date"] = pd.to_datetime(data["publish_date"],format="%Y-%m-%d")
data["publish_date"].head()


# In[ ]:


data["publish_time_of_day"] = pd.DatetimeIndex(data["publish_time"]).time
data["publish_time_of_day"].head()


# In[ ]:


# Hour published
data["publish_hour"] = data["publish_time_of_day"].apply(lambda x: x.hour)


# In[ ]:


data["publish_hour"].head()


# In[ ]:


# Add column of days between published and trended

# How about just videos that trended the day of/after they were published

# Pull days trended for each
# Remember, can be zero, if it trends on the day it was published
data["days_trended_after_publish"] = data["trending_date"] - data["publish_date"]


# In[ ]:


# Change days_trended_after_publish to an integer
data["days_trended_after_publish"] = data["days_trended_after_publish"].dt.days


# In[ ]:


# Add column with sum of total days trended per video
trended_count = data.groupby("video_id").count()["days_trended_after_publish"].reset_index()
trended_count.columns = ["video_id","trended_count"]


# In[ ]:


data = data.merge(trended_count,on="video_id")


# In[ ]:


# Pull category ID data

id_to_category = {}

with open("../input/US_category_id.json","r") as f:
    id_data = json.load(f)
    for category in id_data["items"]:
        id_to_category[category["id"]] = category["snippet"]["title"]

id_to_category


# ### Change category_id into a string first

# In[ ]:


type(data["category_id"][0])


# In[ ]:


data["category_id"] = data["category_id"].astype(str)


# In[ ]:


# Map that data onto the dataset
data.insert(4, "category",data["category_id"].map(id_to_category))


# In[ ]:


# Ok, let's check out the data with the changes we made
data[data["video_id"] == "2kyS6SvSYSE"]


# # EDA

# ## A quick look at correlations between video metrics (more on this later)

# In[ ]:


sns.heatmap(data.corr(),cmap="Blues",annot=True)
# Note, this might not be a good proxy of much, since some videos are in the dataset multiple times


# Is the fact that we're counting videos multiple times (if it trended multiple days) impacting the data?
# How about correlations for only the last day a video trended?

# In[ ]:


# Correlation matrix for only videos on the last day trended
sns.heatmap(data[data["trended_count"] == data["days_trended_after_publish"]].corr(),cmap="Blues",annot=True)


# Ok, that doesn't make a huge difference in correlations. Good to know.

# # Relationship between time published and views, days trended

# Questions to answer:
# 
# * When should I publish my video? If I publish at a certain time, am I more likely to trend?
# * Do users interact more with videos that are published at a certain time? Because we know that there's a high correlation of views/comments to trending
# 
# **Hyphothesis:** It doesn't matter. The correlations between publish hour and everything else is low.

# ## Publish hour

# In[ ]:


# Sum of views per video by hour published
sns.barplot(data = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("publish_hour").sum()["views"].reset_index(),x="publish_hour",y="views",palette="coolwarm")


# Need to go deeper than that - the number of views is probably impacted by outliers, like that 4am hour. How about days trended, rather than just views?

# In[ ]:


# Total days trended by hour published
sns.barplot(x="publish_hour",y="trended_count",data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby("publish_hour").sum()["trended_count"].reset_index(),palette="coolwarm")


# It looks like there might just be more videos published in the afternoon, so the total days trended is higher. How about we look at average days trended by publish hour?

# In[ ]:


sns.barplot(data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby("publish_hour").mean()["trended_count"].reset_index(),
            x="publish_hour",y="trended_count",palette="coolwarm")


# I'm not convinced that is meaningful. Let's try to see something that strips out the outliers

# In[ ]:


sns.boxplot(data=data[data["trended_count"] == data["days_trended_after_publish"]],
            x="publish_hour",y="trended_count",palette="coolwarm")


# That makes more sense. It probably doesn't matter when you publish.

# **Next step:** Run test to calculate statistical significance.

# ## How about day of the week? Does it matter what  day you publish?

# What about day of the week? Are videos that are published certain days of the week more likely to trend?

# In[ ]:


# Add column for day of week that each video was published
data["publish_day"] = data["publish_date"].apply(lambda x: x.strftime('%A'))


# In[ ]:


sns.barplot(data=data.groupby("publish_day").sum()["trended_count"].sort_values(ascending=False).reset_index(),x="publish_day",y="trended_count",palette="coolwarm")


# Videos published on the weekends have the lowest trend count, by a lot. But what about days trended on average by day?

# In[ ]:


sns.barplot(data=data.groupby("publish_day").mean()["trended_count"].sort_values(ascending=False).reset_index(),x="publish_day",y="trended_count",palette="coolwarm")


# This also doesn't look like there is enough variation to make a difference. That is, it probably doesn't matter what day you publish.

# ### Takeaway: It probably doesn't matter what time or what day you publish, because it probably doesn't impact the number of days you trend

# # Question: Do YouTubers who post more often trend more days?

# We have heard though that the YouTube algorithm rewards consistency. If you publish more consistently, you are more likely to trend.

# In[ ]:


# Unique videos
unique_videos = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("channel_title").nunique()["video_id"].reset_index()
unique_videos.sort_values(by="video_id",ascending=False).head()


# In[ ]:


# Total days between first and last published date
last_published = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("channel_title").max()["publish_date"].reset_index()
last_published.head()


# In[ ]:


first_published = data[data["trended_count"] == data["days_trended_after_publish"]].groupby("channel_title").min()["publish_date"].reset_index()
first_published.head()


# In[ ]:


consistency = first_published.merge(last_published,on="channel_title")
consistency.columns = ["channel_title","first_published","last_published"]
consistency["total_days"] = consistency["last_published"] - consistency["first_published"]
consistency["total_days"] = consistency["total_days"].dt.days
consistency = consistency.merge(unique_videos,on="channel_title")
consistency["average_days_between_videos"] = consistency["total_days"]/consistency["video_id"]


# In[ ]:


consistency = consistency.merge(data[["video_id","channel_title","trended_count"]].drop_duplicates().groupby(by="channel_title").sum()["trended_count"].reset_index(),on="channel_title")
consistency.head()


# In[ ]:


sns.lmplot(data=consistency,x="average_days_between_videos",y="trended_count")


# Hmmm... The regression line is upward sloping, which suggests that longer intervals between videos means the total days trended is higher. But the data itself looks like the fewer days between videos published (the lower the average, the fewer days between videos published), the more total days trended

# Let's strip out channels that had fewer than some number of videos.

# In[ ]:


x=5
sns.lmplot(data=consistency[consistency["video_id"] >=x],x="average_days_between_videos",y="trended_count")


# That makes more sense. If you have fewer days between videos, the more likely you are to trend. But to make sure, let's normalize the data by taking the average days trended by average days between videos

# In[ ]:


consistency["average_days_trended"] = consistency["trended_count"]/consistency["video_id"]
consistency.head()


# In[ ]:


sns.lmplot(data=consistency,x="average_days_between_videos",y="average_days_trended")


# In[ ]:


sns.lmplot(data=consistency[consistency["video_id"] >=2],x="average_days_between_videos",y="average_days_trended")


# I limited the output to videos with at least two videos published (otherwise, there is a giant clump of one-off videos skewing the data).

# But the reults are interesting. This suggests that the longer between videos, the more days a video trends on average.

# In all, this suggests two things:
# 
# * First, you can cannibalize your own videos. The channels that post every two or three days trend for fewer days on average, because that channel's new videos replace the old ones. Fewer days trended on average, BUT in total, you trend for more days (as we saw in the average days between videos vs. total days trended chart).
# * Second, if a channel doesn't post that often, some other factor (quality of the video?) outweighs the consistency of posting.

# # Measuring quality of videos

# What is the best measure of quality?
# 
# Based on our correlation matrix above, likes had the highest correlation to views.
# 
# Let's try likes-to-dislikes - Is there a relationship between the like-to-dislike ratio and views?

# In[ ]:


data["likes-to-dislikes"] = data["likes"]/data["dislikes"]


# In[ ]:


sns.heatmap(data[["views","likes","dislikes","comment_count","likes-to-dislikes","trended_count"]].corr(),cmap="Blues",annot=True)


# In[ ]:


# Note: correlations taking into account only the last day a video trended doesn't do much to correlations
sns.heatmap(data[data["days_trended_after_publish"] == data["trended_count"]].corr(),cmap="Blues",annot=True)


# That "likes-to-dislikes" ratio doesn't tell us much. Likes still seem to be the best predictor of views. But interestingly, trended count isn't really correlated to the quality metrics either. That suggests that the algorithm 

# # Videos by category

# Are videos in some categories more likely to trend? Are they likely to trend for longer than videos in other categories?

# Total views:

# In[ ]:


sns.barplot(data=data.groupby("category").count()["views"].reset_index().sort_values(by="views",ascending=False),x="views",y="category",palette="coolwarm")


# Total trended days by category

# In[ ]:


sns.barplot(data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby(by="category").sum()["trended_count"].reset_index().sort_values(by="trended_count",ascending=False),y="category",x="trended_count",palette="coolwarm")


# Average days trended by category

# In[ ]:


sns.barplot(data=data[data["trended_count"] == data["days_trended_after_publish"]].groupby(by="category").mean()["trended_count"].reset_index().sort_values(by="trended_count",ascending=True),x="trended_count",y="category",palette="coolwarm")


# There seems to be a lot of competition in "Sports" and "News & Politics" - those categories have a shorter trending shelf-life, probably because the algorithm rewards recency in those categories.
