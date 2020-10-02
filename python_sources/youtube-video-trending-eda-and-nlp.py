#!/usr/bin/env python
# coding: utf-8

# # Introduction
# Youtube videos are not only produced by vbloger or their other name, "Youtubers". Media corporations including Disney, CNN, BBC, and Hulu also offer some of their material via YouTube as part of the YouTube partnership program.
# 
# If your company, or yourself, a potential million-view youtuber, intend to employ this huge platform to publish your video, it is essential to enhance the content quality, and to increase its visibility. But why Youtube? Because it offers the possibility to monetize your videos, by adding ads during the video progression. With an in-depth analysis of thousands of videos, we could find several keys to increase views, likes, and the most important of all, your incomes.
# 
# The data used in this report can be found at: https://www.kaggle.com/datasnaek/youtube-new/
# 
# The website says that it was last updated on May, 2019; however the lastest publish date in the data in 2018/06/14

# # Description
# The dataset includes data gathered from 40949 videos on YouTube that are contained within the trending category each day.
# 
# There are two kinds of data files, one includes comments (JSON) and one includes video statistics (CSV). They are linked by the unique video_id field.
# 
# The columns in the video file are:
# 1. title
# 2. channel_title
# 3. video_id(Unique id of each video)
# 4. trending_date
# 5. title
# 6. channel_title
# 7. category_id (Can be looked up using the included JSON file)
# 8. publish_time
# 9. tags (Separated by | character, [none] is displayed if there are no tags)
# 10. views
# 11. likes
# 12. dislikes
# 13. comment_count
# 14. thumbnail_link
# 15. comments_disabled
# 16. ratings_disabled
# 17. video_error_or_removed
# 18. description

# # Data Preparation

# Importing Libraries and Loading Data

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from subprocess import check_output
from wordcloud import wordcloud, STOPWORDS
import warnings
from collections import Counter
import datetime
import glob


# In[ ]:


#hiding warnings for cleaner display
warnings.filterwarnings('ignore')

#Configuring some options
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
#For interactive plots
get_ipython().run_line_magic('matplotlib', 'notebook')


# In[ ]:


df = pd.read_csv("../input/youtube-new/USvideos.csv")
df_ca = pd.read_csv("../input/youtube-new/CAvideos.csv")
df_de = pd.read_csv("../input/youtube-new/DEvideos.csv")
df_fr = pd.read_csv("../input/youtube-new/FRvideos.csv")
df_gb = pd.read_csv("../input/youtube-new/GBvideos.csv")


# # Setting up configuration for visuals

# In[ ]:


PLOT_COLORS = ["#268bd2", "#0052CC", "#FF5722", "#b58900", "#003f5c"]
pd.options.display.float_format = '{:.2f}'.format
sns.set(style="ticks")
plt.rc('figure', figsize=(8, 5), dpi=100)
plt.rc('axes', labelpad=20, facecolor="#ffffff", linewidth=0.4, grid=True, labelsize=14)
plt.rc('patch', linewidth=0)
plt.rc('xtick.major', width=0.2)
plt.rc('ytick.major', width=0.2)
plt.rc('grid', color='#9E9E9E', linewidth=0.4)
plt.rc('font', family='Arial', weight='400', size=10)
plt.rc('text', color='#282828')
plt.rc('savefig', pad_inches=0.3, dpi=300)


# # Data Exploration

# In[ ]:


df.head()


# # Statistical Information about the numerical columns of dataset!

# In[ ]:


df.describe()


# Notes from above table:
# 1. Average number of likes - 74266, whereas dislikes are 3711
# 2. Average number of views - 2,360,784 and median is 681861
# 3. Average comment count - 8446 and max - 13,61,580

# In[ ]:


df.shape


# Now let's see some information about our dataset

# In the dataset, the Trending Date and Published Time are not in the Unix date-time format. Let's fix this first.

# In[ ]:


df.info()


# There are 40379 entries in the dataset and all the columns in the dataset are complete(i.e. they have 40,949 non null entries) except description column which has some null values; it only has 40,379 non null values

# DATA CLEANING
# The Description column has some null values. These are some of the rows whose description values are null. We can see that null values are denoted by NaN
# 
# **For Data cleaning, and to get rid of null values that were set to empty string in place of each nullvalue in the description Column**

# In[ ]:


df[df["description"].apply(lambda x: pd.isna(x))].head(3)


# # Let's see the Collection year of Data

# In[ ]:


cdf = df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts().to_frame().reset_index().rename(columns={"index": "year", "trending_date": "No_of_videos"})

fig, ax = plt.subplots()
_ = sns.barplot(x="year", y="No_of_videos", data = cdf, 
                palette = sns.color_palette(['#ff764a','#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Year", ylabel="No. of videos")


# In[ ]:


df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts(normalize=True)


# We can see that dataset was collected in 2017-18
# 
# 77% in 2017 and 23% in 2018

# **To see the correlation between the likes, dislikes, comments,and views lets plot a correlation matrix**

# In[ ]:


columns_show = ['views', 'likes', 'dislikes', 'comment_count']
f, ax = plt.subplots(figsize=(8, 8))
dfe = df[columns_show].corr()
sns.heatmap(dfe,mask=np.zeros_like(dfe, dtype=np.bool), cmap='RdYlGn',linewidth=0.30,annot=True)


# In the correlation plot matrix, for USA dataset, the columns with:
# 1. High correlation - Views and likes, comment_count and Dislikes
# 2. Medium Correlation - Views and comment_count, comment_count and dislikes
# 3. Low Correlation - Likes and Dislike

# Let's verify that by plotting a scatter plot between views and likes to visualize the relationship between these variables

# In[ ]:


fig, ax = plt.subplots()
_ = plt.scatter(x=df['views'], y=df['likes'], color = PLOT_COLORS[2], edgecolors="#000000",
               linewidth=0.5)
_ = ax.set(xlabel="views", ylabel="likes")


# we can see that views and likes are truly positively correlated: as one increases, the other increases too-mostly.

# # Histogram of Views to verify-
# 1. How many videos are between 10 million and 20 million views
# 2. How many videos have between 20 million and 30 million views, and so on

# In[ ]:


fig, ax = plt.subplots()
_ = sns.distplot(df["views"], kde=False, color=PLOT_COLORS[4],
                hist_kws={'alpha': 1},bins=np.linspace(0, 2.3e8,47),ax=ax)
_ = ax.set(xlabel="Views",ylabel="No. of videos", xticks=np.arange(0, 2.4e8, 1e7))
_ = ax.set_xlim(right=2.5e8)
_ = plt.xticks(rotation=90)


# We note that the vast majority of trending videos have 5million views or less.
# 
# Now lets us plot the histogram for videos with 25million views or less to get a closer look at the distribution of the data

# In[ ]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["views"]<25e6]["views"], kde=False, color=PLOT_COLORS[1], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos")


# **Now we see the majority of trending videos have 1million views or less. let's see the exact percentage of videos less than 1million views**

# In[ ]:


df[df['views']<1e6]['views'].count()/df['views'].count() * 100


# Since, it's around 60%. Similarly, we see the percentage of videos with less than 1.5 million views is approx 71%, and that the percentage of videos with less than 5million views is around 91%

# # Likes Histogram
# 
# After views, we will plot the histogram for likes column

# In[ ]:


plt.rc('figure.subplot',wspace=0.9)
fig, ax = plt.subplots()
_ = sns.distplot(df["likes"], kde=False, color=PLOT_COLORS[3],
                hist_kws={'alpha':1}, bins=np.linspace(0, 6e6, 61), ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of Videos")
_ = plt.xticks(rotation=90)


# we note that the vast majority of trending videos have between 0 and 100,000 likes

# **Since, a video could be in trending for several days. There might be multiple rows of a particular video. In order to calculate the total_views, comments, likes, dislikes, of a video. we need to groupby with video_id. The below script will give the total no. of views/comment/likes, and dislikes of a video**

# In[ ]:


df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')
df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')

#Separate date and time into two columns from 'publish_time' column
df.insert(4, 'pub_date', df['publish_time'].dt.date)
df['publish_time'] = df['publish_time'].dt.time
df['pub_date'] = pd.to_datetime(df['pub_date'])


# In[ ]:


us_views = df.groupby(['video_id'])['views'].agg('sum')
us_likes = df.groupby(['video_id'])['likes'].agg('sum')
us_dislikes = df.groupby(['video_id'])['dislikes'].agg('sum')
us_comment_count = df.groupby(['video_id'])['comment_count'].agg('sum')


# In[ ]:


df_usa_sdtr = df.drop_duplicates(subset='video_id', keep=False, inplace=False)
df_usa_mdtr = df.drop_duplicates(subset='video_id', keep='first', inplace=False)

frames = [df_usa_sdtr, df_usa_mdtr]
df_usa_without_duplicates = pd.concat(frames)

df_usa_comment_disabled = df_usa_without_duplicates[df_usa_without_duplicates['comments_disabled']==True].describe()
df_usa_rating_disabled = df_usa_without_duplicates[df_usa_without_duplicates['ratings_disabled']==True].describe()
df_usa_video_error = df_usa_without_duplicates[df_usa_without_duplicates['video_error_or_removed']==True].describe()


# Removing duplicates to get the correct numbers otherwise there will be redundancy
# Getting the number of videos on which comments disabled/rating disabled/video error

# How many videos trending per day?

# In[ ]:


df_usa_sdtr.head()


# Approx 544 videos were trending per day in USA

# # Videos trending for more than a day

# In[ ]:


df_usa_mdtr.head()


# Approx 4079 videos were trending per day in USA

# # which videos were trending on maximum days and what is the title, likes, dislikes, comments, and views.

# In[ ]:


df_usa_mdtr = df.groupby(by=['video_id'],as_index=False).count().sort_values(by='title',ascending=False).head()

plt.figure(figsize=(8,8))
sns.set_style("whitegrid")
ax = sns.barplot(x=df_usa_mdtr['video_id'],y=df_usa_mdtr['trending_date'], data=df_usa_mdtr)
plt.xlabel("Video_Id")
plt.ylabel("Count")
plt.title("Top 5 videos that trended maximum days in USA")


# **Videos were trended for maximum days**
# The Maximum no. of days a video trended is 30 i.e. for 'j4KvrAUjn6c' video id. Now, the below script gives its likes, dislikes, views and comments.

# In[ ]:


df_us_max_views = us_views['j4KvrAUjn6c']
df_us_max_likes = us_likes['j4KvrAUjn6c']
df_us_max_dislikes = us_dislikes['j4KvrAUjn6c']
df_us_max_comment = us_comment_count['j4KvrAUjn6c']


# # **Top Trending Channel in USA**

# In[ ]:


cdf = df.groupby("channel_title").size().reset_index(name="video_count") .sort_values("video_count", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8,8))
_ = sns.barplot(x="video_count", y="channel_title", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=20, reverse=True),ax=ax)
_ = ax.set(xlabel="No. of videos", ylabel="Channel")


# ESPN is in the Top list of channels in USA

# # **TOP 5 USA Categories**

# In[ ]:


usa_category_id = df_usa_without_duplicates.groupby(by=['category_id'],as_index=False).count().sort_values(by='title',ascending=False).head(5)

plt.figure(figsize=(7,7))
sns.kdeplot(usa_category_id['category_id']);
plt.xlabel("category IDs")
plt.ylabel("Count")
plt.title("Top 5 categories IDs for USA")


# This graph visualizes 24 category Id to be maximum in the range of 22-27

# # Which Video Category has the largest number of trending videos?
# 
# First, we will add a column that contains category names based on the values in category_id column. We will use a category JSON file provided with the dataset which contains information about each category.

# In[ ]:


import simplejson as json
with open("../input/youtube-new/US_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['category_id'].map(cat_dict)


# Now we can see which category had the largest number of trending videos

# In[ ]:


cdf = df["category_name"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
fig, ax = plt.subplots()
_ = sns.barplot(x="category_name", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = ax.set(xlabel="Category", ylabel="No. of videos")


# We see that the Entertainment category contains the largest number of trending videos among other categories: around 10,000 videos, followed by Music category with around 6,200 videos, followed by Howto & Style category with around 4100 videos and so on.

# # How many video titles contain Capitalized word?
# 
# Now we want to see how many trending video titles contain atleast a capitalized word(e.g.HOW). To do that, we will add a new variable (column) to the dataset whose value is True if the video title has at least a capitalized word in it, and False otherwise

# In[ ]:


def Capitalized_word(s):
    for m in s.split():
        if m.isupper():
            return True
    return False

df["contains_capital_words"] = df["title"].apply(Capitalized_word)

value_counts = df["contains_capital_words"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No','Yes'],
          colors=['#003f5c', '#ffa600'], textprops={'color':'#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Title Contains Capitalized word?')


# In[ ]:


df["contains_capital_words"].value_counts(normalize=True)


# we can see that 44% of trending video titles contain atleast a capitalized word. we will later use this added new column contains_capital_words in analyzing correlation between variables

# # Video Title Lengths
# We will add another column to dataset to represent the length of each video title,and visualize it in histogram to understand about the lengths of trending video title

# In[ ]:


df["title_length"] = df["title"].apply(lambda x: len(x))

fig, ax = plt.subplots()
_ = sns.distplot(df["title_length"], kde = False, rug = False, color=PLOT_COLORS[4], hist_kws={'alpha':1},
                ax=ax)
_ = ax.set(xlabel="Title Lenth", ylabel = "Number of Videos", xticks=range(0, 110, 10))


# We can see that title-length distribution resembles a normal distribution, where most videos have title lengths between 30 and 60 characters approximately.
# 
# Now let's draw a scatter plot between title length and number of views to see the relationship between these two variables

# In[ ]:


fig, ax = plt.subplots()
_ = ax.scatter(x=df['views'], y=df['title_length'], color=PLOT_COLORS[2], edgecolors="#000000", 
               linewidth=0.5)
_ = ax.set(xlabel = "views", ylabel = "Title_Length")


# In the above scatter plot,we can see there is no relationship between these two variable i.e. title_length and number of views. 
# 
# There is one interesting thing - videos that have 100,00,000 views have more than title length between 33 and 55 characters approximately.

# # Most Common Words in video Title
# we will verify significant words in trending video titles. we will display the 25 most common words in all trending video titles

# In[ ]:


title_words = list(df["title"].apply(lambda x: x.split()))
title_words = [x for y in title_words for x in y]
Counter(title_words).most_common(25)


# In 40949 videos - "-" and "|" have appeared 11452 and 10663 times respectively.
# We notice also that words "Video","Trailer","How" and "2018" are common in trending video titles; each occured in 1613-1901 video titles.
# 
# **We will draw a word cloud for most frequent words used**

# In[ ]:


#wc = wordcloud.WordCloud(width=1200, height=600, collocations=False, Stopwords=None, 
#background_color="white",colormap="tab20b").generate_from_frequencies(dist(Counter(title_words).most_common(500)))

wc = wordcloud.WordCloud(width=1200, height=500, collocations=False, background_color="white",
                        colormap="tab20b").generate(" ".join(title_words))
plt.figure(figsize=(8,6))
plt.imshow(wc, interpolation='bilinear')
_ = plt.axis("off")


# # Trending Videos and Publishing Time
# 
# Publish time is Cordinated Universal Time(UTC) time zone
# 
# We will add the date and hour of publishing each video, remove original publishing time column as we will not need it

# # To be Continuedd....

# In[ ]:




