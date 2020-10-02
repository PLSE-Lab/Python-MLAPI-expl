#!/usr/bin/env python
# coding: utf-8

# Import all the important libraries

# In[ ]:


import pandas as pd
import numpy as np

import matplotlib as mpl
from matplotlib import pyplot as plt
import seaborn as sns

import warnings
from collections import Counter
import datetime
import wordcloud
import json


# Read the csv file and convert it to pandas dataframe

# In[ ]:


df = pd.read_csv('../input/youtube-new/CAvideos.csv')
df.head()


# In[ ]:


df.info()


# To check the total number of tuples(rows,columns) in the dataframe

# In[ ]:


df.shape


# In[ ]:


df[df["description"].apply(lambda x: pd.isna(x))].head(3)


# In[ ]:


df["description"] = df["description"].fillna(value="")


# Let's see in which years the dataset was collected
# 
# 

# In[ ]:


cdf = df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts()             .to_frame().reset_index()             .rename(columns={"index": "year", "trending_date": "No_of_videos"})

fig, ax = plt.subplots()
_ = sns.barplot(x="year", y="No_of_videos", data=cdf, 
                palette=sns.color_palette(['#ff764a', '#ffa600'], n_colors=7), ax=ax)
_ = ax.set(xlabel="Year", ylabel="No. of videos")


# In[ ]:


df["trending_date"].apply(lambda x: '20' + x[:2]).value_counts(normalize=True)


# HISTOGRAM
# 
# views histogram

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


# In[ ]:


fig, ax = plt.subplots()
_ = sns.distplot(df["views"], kde=False, color=PLOT_COLORS[4], 
                 hist_kws={'alpha': 1}, bins=np.linspace(0, 2.3e8, 47), ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos", xticks=np.arange(0, 2.4e8, 1e7))
_ = ax.set_xlim(right=2.5e8)
_ = plt.xticks(rotation=90)


# We notice that vast majority of trending videos have 5 million or less views. So lets plot a histogram with 25 million or less to get a closer look at the distribution.

# In[ ]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["views"] < 25e6]["views"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Views", ylabel="No. of videos")


# Now we see that the majority of trending videos have 1 million views or less. Let's see the exact percentage of videos less than 1 million views

# In[ ]:


df[df['views']<1e6]['views'].count() / df['views'].count()*100


# So it is around 75%. Similarly videos less than 1.5 million is 83% and less than 5 million is 96%.

# Likes Histogram

# In[ ]:


plt.rc('figure.subplot', wspace=0.9)
fig, ax = plt.subplots()
_ = sns.distplot(df["likes"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, 
                 bins=np.linspace(0, 6e6, 61), ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of videos")
_ = plt.xticks(rotation=90)


# In[ ]:


fig, ax = plt.subplots()
_ = sns.distplot(df[df["likes"] <= 1e5]["likes"], kde=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Likes", ylabel="No. of videos")


# Now we can see that the majority of trending videos have 20000 likes or less. 
# 

# In[ ]:


df[df['likes'] < 2e4]['likes'].count() / df['likes'].count() * 100


# After we described numerical columns previously, we now describe non-numerical columns

# In[ ]:


df.describe(include = 'O')


# In[ ]:


grouped = df.groupby("video_id")
groups = []
wanted_groups = []
for key, item in grouped:
    groups.append(grouped.get_group(key))

for g in groups:
    if len(g['title'].unique()) != 1:
        wanted_groups.append(g)

wanted_groups[0]


# Now we want to see how many trending video titles contain at least a capitalized word (e.g. HOW).

# In[ ]:


def contains_capitalized_word(s):
    for w in s.split():
        if w.isupper():
            return True
    return False


df["contains_capitalized"] = df["title"].apply(contains_capitalized_word)

value_counts = df["contains_capitalized"].value_counts().to_dict()
fig, ax = plt.subplots()
_ = ax.pie([value_counts[False], value_counts[True]], labels=['No', 'Yes'], 
           colors=['#003f5c', '#ffa600'], textprops={'color': '#040204'}, startangle=45)
_ = ax.axis('equal')
_ = ax.set_title('Title Contains Capitalized Word?')


# In[ ]:


df["contains_capitalized"].value_counts(normalize=True)


# Almost 49% of trending videos on youtube use capitalized words.

# Let's add another column to our dataset to represent the length of each video title, then plot the histogram of title length to get an idea about the lengths of trnding video titles

# In[ ]:


df["title_length"] = df["title"].apply(lambda x: len(x))

fig, ax = plt.subplots()
_ = sns.distplot(df["title_length"], kde=False, rug=False, 
                 color=PLOT_COLORS[4], hist_kws={'alpha': 1}, ax=ax)
_ = ax.set(xlabel="Title Length", ylabel="No. of videos", xticks=range(0, 110, 10))


# We can see that title-length distribution resembles a normal distribution, where most videos have title lengths between 30 and 60 character approximately.
# 
# Now let's draw a scatter plot between title length and number of views to see the relationship between these two variables

# In[ ]:


fig, ax = plt.subplots()
_ = ax.scatter(x=df['views'], y=df['title_length'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Title Length")


# Videos that have 100,000,000 views and more have title length between 33 and 55 characters approximately.

# In[ ]:


df.corr()


# We see for example that views and likes are highly positively correlated with a correlation value of 0.83; we see also a high positive correlation (0.84) between likes and comment count, and between dislikes and comment count (0.64).
# 
# There is some positive correlation between views and dislikes, between views and comment count, between likes and dislikes.

# In[ ]:


h_labels = [x.replace('_', ' ').title() for x in 
            list(df.select_dtypes(include=['number', 'bool']).columns.values)]

fig, ax = plt.subplots(figsize=(10,6))
_ = sns.heatmap(df.corr(), annot=True, xticklabels=h_labels, yticklabels=h_labels, cmap=sns.cubehelix_palette(as_cmap=True), ax=ax)


# The correlation map and correlation table above say that views and likes are highly positively correlated. Let's verify that by plotting a scatter plot between views and likes to visualize the relationship between these variables

# In[ ]:


fig, ax = plt.subplots()
_ = plt.scatter(x=df['views'], y=df['likes'], color=PLOT_COLORS[2], edgecolors="#000000", linewidths=0.5)
_ = ax.set(xlabel="Views", ylabel="Likes")


# Let's see if there are some words that are used significantly in trending video titles. We will display the 25 most common words in all trending video titles

# In[ ]:


title_words = list(df['title'].apply(lambda x: x.split()))
title_words = [x for y in title_words for x in y]
Counter(title_words).most_common(25)


# In[ ]:


wc = wordcloud.WordCloud(width=1200, height=500, 
                         collocations=False, background_color="white", 
                         colormap="tab20b").generate(" ".join(title_words))
plt.figure(figsize=(15,10))
plt.imshow(wc, interpolation='bilinear')
_ = plt.axis("off")


# 
# Which channels have the largest number of trending videos?

# In[ ]:


cdf = df.groupby("channel_title").size().reset_index(name="video_count")     .sort_values("video_count", ascending=False).head(20)

fig, ax = plt.subplots(figsize=(8,8))
_ = sns.barplot(x="video_count", y="channel_title", data=cdf,
                palette=sns.cubehelix_palette(n_colors=20, reverse=True), ax=ax)
_ = ax.set(xlabel="No. of videos", ylabel="Channel")


# In[ ]:


with open("../input/youtube-new/CA_category_id.json") as f:
    categories = json.load(f)["items"]
cat_dict = {}
for cat in categories:
    cat_dict[int(cat["id"])] = cat["snippet"]["title"]
df['category_name'] = df['category_id'].map(cat_dict)


# In[ ]:


cdf = df["category_name"].value_counts().to_frame().reset_index()
cdf.rename(columns={"index": "category_name", "category_name": "No_of_videos"}, inplace=True)
fig, ax = plt.subplots()
_ = sns.barplot(x="category_name", y="No_of_videos", data=cdf, 
                palette=sns.cubehelix_palette(n_colors=16, reverse=True), ax=ax)
_ = ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
_ = ax.set(xlabel="Category", ylabel="No. of videos")


# In[ ]:




