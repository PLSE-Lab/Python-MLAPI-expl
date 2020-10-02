#!/usr/bin/env python
# coding: utf-8

# ## Trending Youtube Videos Analysis

# In this analysis, I will perform EDA on the dataset of trending Youtube videos.  This is my first Python notebook and I am accustomed to performing this type of work in R (check out my other kernels to see those analyses), but I hope to improve my Python skills, as well.  I am a young learning college student  so feel free to leave a like and/or feedback in the comments! 

# ### Data Preparation

# #### Packages / Import

# We will first load packages and our data and see what we are working with.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime


# In[ ]:


youtube = pd.read_csv("../input/youtube-new/USvideos.csv")

youtube['trending_date'] = pd.to_datetime(youtube['trending_date'], format='%y.%d.%m') #parsing
youtube['publish_time'] = pd.to_datetime(youtube['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
youtube['category_id'] = youtube['category_id'].astype(str)

youtube.head()


# In[ ]:


youtube.info()


# In[ ]:


youtube.describe()


# #### Getting Categories

# we will read in the category_id json file to match appropriate categories with their ids.

# In[ ]:


id_to_category = {}

with open('../input/youtube-new/US_category_id.json' , 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_category[category['id']] = category['snippet']['title']
        
youtube['category'] = youtube['category_id'].map(id_to_category)


# #### Creating New Variables
# 

# The like to dislike ratio is an important measure of the viewers' approval of the video.

# In[ ]:


youtube["ldratio"] = youtube["likes"] / youtube["dislikes"]


# These variables record the extent to which people react to the video.

# In[ ]:


youtube["perc_comment"] = youtube["comment_count"] / youtube["views"]
youtube["perc_reaction"] = (youtube["likes"] + youtube["dislikes"]) / youtube["views"]


# Lastly, we will get more elements from our publishing time.

# In[ ]:


youtube['publish_date'] = youtube['publish_time'].dt.date
youtube['publish_tym'] = youtube['publish_time'].dt.time


# In[ ]:


youtube.head()


# The data is clean and ready to be analyzed.

# ### EDA

# #### Basic Distributions

# We will first plot distributions of various individual variables to better understand the data.

# In[ ]:


def distribution_cont(youtube, var):
    plt.hist(youtube[youtube["dislikes"] != 0][var])
    plt.xlabel(f"{var}")
    plt.ylabel("Count")
    plt.title(f"Distribution of Trending Video {var}")
    plt.show()
for i in ["views", "likes", "dislikes", "comment_count", "ldratio", "perc_reaction", "perc_comment"]:
    distribution_cont(youtube, i)


# All distributions are skewed to the left due to some outliers.

# #### Correlation Between Variables

# All of our continuous variables involve the users' interactions with the videos, including views, likes, dislikes, and comments.  We will visualize all of this with a correlation plot.

# In[ ]:


contvars = youtube[["views", "likes", "dislikes", "comment_count", "ldratio", "perc_comment", "perc_reaction"]]
corr = contvars.corr()

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})
plt.show()


# There exist many moderate correlations here.  One of the most meaningful observations is that comment count has a positive relationship with views, likes, and dislikes.  People who react with a like or dislike are also more likely to comment.

# #### Most Frequent Trending Channels

# In[ ]:


by_channel = youtube.groupby(["channel_title"]).size().sort_values(ascending = False).head(10)
sns.barplot(by_channel.values, by_channel.index.values, palette = "rocket")
plt.title("Top 10 Most Frequent Trending Youtube Channels")
plt.xlabel("Video Count")
plt.show()


# Sports channels (ESPN, NBA) and tv shows, especially late night comedies, are trending most often.

# #### Most Frequent Trending Categories
# 

# In[ ]:


by_cat = youtube.groupby(["category"]).size().sort_values(ascending = False)
sns.barplot(by_cat.values, by_cat.index.values, palette = "rocket")
plt.title("Most Frequent Trending Youtube Categories")
plt.xlabel("Video Count")
plt.show()


# Entertainment and music lead by far.

# #### Most Viewed Trending Channels
# 

# In[ ]:


top_channels2 = youtube.groupby("channel_title").size().sort_values(ascending = False)
top_channels2 = list(top_channels2[top_channels2.values >= 20].index.values)
only_top2 = youtube
for i in list(youtube["channel_title"].unique()):
    if i not in top_channels2:
        only_top2 = only_top2[only_top2["channel_title"] != i]

by_views = only_top2.groupby(["channel_title"]).mean().sort_values(by = "views", ascending = False).head(10)
sns.barplot(by_views["views"], by_views.index.values, palette = "rocket")
plt.title("Top 10 Most Viewed Trending Youtube Channels")
plt.xlabel("Average Views")
plt.show()


# The channel must have at least 20 trending videos in the dataset to make the list.  Many vevo (music video) channels make the list.  This is a preview for the results of our next plot:

# #### Most Viewed Trending Categories

# In[ ]:


by_views_cat = youtube.groupby(["category"]).mean().sort_values(by = "views", ascending = False)
sns.barplot(by_views_cat["views"], by_views_cat.index.values, palette = "rocket")
plt.title("Top 10 Most Viewed Trending Youtube Channels")
plt.xlabel("Average Views")
plt.show()


# #### Most Liked Trending Channels
# 

# In[ ]:


top_channels = youtube.groupby("channel_title").size().sort_values(ascending = False)
top_channels = list(top_channels[top_channels.values >= 20].index.values)
only_top = youtube
for i in list(youtube["channel_title"].unique()):
    if i not in top_channels:
        only_top = only_top[only_top["channel_title"] != i]

like_channel = only_top[only_top["dislikes"] != 0].groupby(["channel_title"]).mean().sort_values(by = "ldratio", ascending = False).head(10)
sns.barplot(like_channel["ldratio"], like_channel.index.values, palette = "rocket")
plt.title("Top 10 Most Liked Trending Youtube Channels")
plt.xlabel("Average Like to Dislike Ratio")
plt.show()


# The channel must have at least 20 trending videos in the dataset to make the list.  BANGTANTV crushes the competition with an average like to dislike ratio of nearly 300 to 1.

# #### Most Liked Trending Categories

# In[ ]:


like_category = youtube[youtube["dislikes"] != 0].groupby("category").mean().sort_values(by = "ldratio", ascending = False)
sns.barplot(like_category["ldratio"], like_category.index.values, palette = "rocket")
plt.title("Top 10 Most Liked Trending Youtube Categories")
plt.xlabel("Average Like to Dislike Ratio")
plt.show()


# News and politics, which can be very divisive, unsurprisingly averages the lowest like to dislike ratio. And of course, everyone loves pets and animals! 

# #### The Numbers Over Time
# 

# Let's visualize views, like to dislike ratio, and more over time using the trending date.

# In[ ]:


def over_time(youtube, var):
    averages = youtube[youtube["dislikes"] != 0].groupby("trending_date").mean()
    plt.plot(averages.index.values, averages[var])
    plt.xticks(rotation = 90)
    plt.xlabel("Date")
    plt.ylabel(f"Average {var}")
    plt.title(f"Average {var} Over Time (11/14/17 - 6/14/18)")
    plt.show()


# In[ ]:


over_time(youtube, "views")


# Views per trending video appeared to skyrocket beginning around April of 2018.

# In[ ]:


over_time(youtube, "ldratio")


# Some event caused the average like to dislike ratio on trending videos to decrease dramatically around January of 2018.

# In[ ]:


over_time(youtube, "perc_reaction") #Recall perc_reaction is (likes + dislikes) / views


# There was a large increase in people who liked and disliked trending videos in December of 2017, and a large decrease in February of 2018.

# In[ ]:


over_time(youtube, "perc_comment") #Recall perc_comment is comments / views


# The percent of people who comment on trending videos has been quite volatile, though exhibits similar patterns as our "perc_reation" chart (February, 2018 for example).

# #### What Publishing Time Receives the Most Views?

# In[ ]:


youtube["hour"] = youtube['publish_time'].dt.hour
by_hour = youtube.groupby("hour").mean()

plt.plot(by_hour.index.values, by_hour["views"])
plt.scatter(by_hour.index.values, by_hour["views"])
plt.xlabel("Hour of Day")
plt.ylabel("Average Number of Views")
plt.title("Average Amount of Views on Trending Videos by the Hour")
plt.show()


# Videos published at 4 AM received the most views on average.  This may be due to the fact that US citizens are first waking up after this time and have all day to make the video popular.  9 AM is also a good time to puublish a video when hoping for many views on the trending list.  Trending videos published later in the evening usually aren't as viewed.

# ### Other Fun Questions

# #### Trump's YouTube Stats

# Now, we will see how many trending videos contained the US president's last name.  

# In[ ]:


trump = youtube[youtube["title"].str.contains("Trump")]
trump.head()


# The US President's last name appeared 298 times out of the total of 40909 rows in the dataset for a proportion of 0.7%.  Now, let's see how viewers responded to videos containing his name:

# In[ ]:


trump.describe()


# The average like to dislike ratio for videos with Trump's name is 7.2, even lower than the average ratio for all news and politics videos (see previous visualization).  However, we have not stated whether these videos address him positively or negatively.  We will take a look at the videos with the min and max like to dislike ratios.

# In[ ]:


trump.sort_values(by = "ldratio").iloc[0]


# This video recieved the lowest like to dislike ratio out of all the Trump videos, and the video's sentiment appears to be negative against him based on the title.  Here's a similar story with second lowest like to dislike ratio and many more views.

# In[ ]:


trump.sort_values(by = "ldratio").iloc[2]


# Now, for the video with the best like to dislike ratio:

# In[ ]:


trump.sort_values(by = "ldratio").iloc[-1]


# The sentiment towards Trump is unclear from the title, but it falls under the comedy category and was uploaded on Christmas.

# Note: No political opinions associated with analysis :)

# #### Get the Top Youtubers

# Lastly, we will create a function that returns the top or bottom youtuber for a given category in a given measure between two given dates.  It will be used for the following questions.

# In[ ]:


def get_top_video(youtube, min_trend_date, max_trend_date, stat, top = True, cat = list(youtube["category"].unique())):
    
    min_trend_list = min_trend_date.split("-")
    max_trend_list = max_trend_date.split("-")
    min_date = datetime(int(min_trend_list[0]), int(min_trend_list[1]), int(min_trend_list[2]))
    max_date = datetime(int(max_trend_list[0]), int(max_trend_list[1]), int(max_trend_list[2]))
    
    youtube = youtube[(youtube["trending_date"] >= min_date) & (youtube["trending_date"] <= max_date)]
    
    if stat == "ldratio":
        youtube = youtube[youtube["views"] >= 100000]
    
    for i in list(youtube["category"].unique()): 
        if i not in cat:
            youtube = youtube[youtube["category"] != i]
            
    if top == True:
        leaders = youtube.loc[youtube[stat].idxmax()][["title", "channel_title"]]
    else:
        leaders = youtube.loc[youtube[stat].idxmin()][["title", "channel_title"]]
    
    title_channel = list(leaders)
    
    return title_channel[0] + " by " + title_channel[1]


# #### Most Overall Viewed Video

# In[ ]:


get_top_video(youtube, "2017-11-14", "2018-6-14", "views")


# #### Most Overall Liked Video

# In[ ]:


get_top_video(youtube, "2017-11-14", "2018-6-14", "likes")


# #### Most Overall Disliked Video

# In[ ]:


get_top_video(youtube, "2017-11-14", "2018-6-14", "dislikes")


# #### Best Overall Like to Dislike Ratio Video 

# In[ ]:


get_top_video(youtube, "2017-11-14", "2018-6-14", "ldratio")


# #### Worst Overall Like to Dislike Ratio Video

# In[ ]:


get_top_video(youtube, "2017-11-14", "2018-6-14", "ldratio", top = False)


# #### Most Viewed Video on Christmas

# In[ ]:


get_top_video(youtube, "2017-12-25", "2017-12-25", "views")


# #### Most Viewed Gaming Video in 2018

# In[ ]:


get_top_video(youtube, "2018-1-1", "2018-6-14", "views", cat = ["Gaming"])


# #### Most Viewed Science or Education Video in 2018

# In[ ]:


get_top_video(youtube, "2018-1-1", "2018-6-14", "views", cat = ["Education", "Science & Technology"])


# #### Most Reactive Entertainment Video in 2018

# In[ ]:


get_top_video(youtube, "2018-1-1", "2018-6-14", "perc_reaction", cat = ["Entertainment"])


# #### Most Active News/Politics Comment Section in 2018

# In[ ]:


get_top_video(youtube, "2018-1-1", "2018-6-14", "perc_comment", cat = ["News & Politics"])


# #### Worst Like to Dislike Ratio News/Politics Video in 2018

# In[ ]:


get_top_video(youtube, "2018-1-1", "2018-6-14", "ldratio", top = False, cat = ["News & Politics"])


# ### Conclusions / Recommendations

# * Views, likes, dislikes, and comment counts all have pretty strong positive relationships with one another.
# * Reactive comments sections are likely to be accompanied by reactive likes and dislikes.
# * If YouTube wants to optimize views, they should modify their algorithm to put mostly music videos on trending.  In the data, entertainment videos are more frequently trending than music videos.
# * If YouTube wants to optimize like to dislike ratio, they should modify their algorithm to put mostly pet/animal videos on trending and less news/politics.
# * YouTube took an action with their trending videos in April of 2018 that was successful in skyrocketing their views for the immediate months to come.  
# * Like to dislike ratios were extremely volatile in late 2017 and early 2018, but calmed down afterwards.
# * If YouTube wantsa bigger proportion of people to react to their trending videos more often with likes, dislikes, and comments, they should avoid whatever they did in early February of 2018

# #### Thank you for reading my YouTube analysis! If you liked it or have feedback, feel free to comment and like :)
