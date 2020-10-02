#!/usr/bin/env python
# coding: utf-8

# This notebook is about creating functions for the USvideos EDA. It goes through some basic data preprocessing like formatting datetime features, processing category_id and splitting dataset based on last and first trending date. It then heads into creating functions and basic documentation for explaining the parameters in the functions and how they can be used. 
# 
# These functions are pretty universal and can be applied to any of the csv files in the YouTube dataset. I'll be going into a more in depth analysis in my next notebook.

# In[ ]:


import numpy as np 
import pandas as pd
import json

import seaborn as sns
sns.set_style('whitegrid')
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings(action="ignore")


# In[ ]:


df = pd.read_csv('../input/youtube-new/USvideos.csv')


# In[ ]:


df.head()


# # Data Preprocessing

# ### Converting date and time columns to datetime

# In[ ]:


df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')


# In[ ]:


df['trending_date'].head()


# In[ ]:


df['publish_time'] = pd.to_datetime(df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')


# In[ ]:


df['publish_time'].head()


# In[ ]:


df.insert(5, 'publish_date', df['publish_time'].dt.date)


# In[ ]:


df['publish_time'] = df['publish_time'].dt.time


# In[ ]:


df['publish_date'] = pd.to_datetime(df['publish_date'])


# ### Processing category_id

# In[ ]:


id_to_cat = {}

with open('../input/youtube-new/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_cat[category['id']] = category['snippet']['title']


# In[ ]:


id_to_cat


# In[ ]:


df['category_id'] = df['category_id'].astype(str)


# In[ ]:


df.insert(5, 'category', df['category_id'].map(id_to_cat))


# ### Duplicated entries

# In[ ]:


df['video_id'].nunique()


# In[ ]:


len(df['video_id'])


# Out of 40949 videos, only 6351 are unique as these videos were trending on multiple days.

# In[ ]:


print(df.shape)
df_last = df.drop_duplicates(subset=['video_id'], keep='last', inplace=False)
df_first = df.drop_duplicates(subset=['video_id'], keep='first', inplace=False)
print(df_last.shape)
print(df_first.shape)


# In[ ]:


print(df['video_id'].duplicated().any())
print(df_last['video_id'].duplicated().any())
print(df_first['video_id'].duplicated().any())


# df has a lot of repetitions as many videos were trending for multiple days. We split the df into df_last to keep the updated stats on the last day it was trending and df_first for the first day it was trending.

# In[ ]:


df_last.head()


# # Functions

# ### Documentation
# 
# Functions:
# - top_10 ( df, col, num=10 )
# - bottom_10 ( df, col, num=10 )
# - channel_stats ( df, channel, num=5, arrange_by='views' )
# - find_videos_by_trending_date ( df, date, num=5, arrange_by='views', category=False )
# - find_videos_by_publish_date ( df, date, num=5, arrange_by='views', publish_to_trend_time=False )
# - find_videos_by_category ( df, cat, num=5, arrange_by='views' )
# 
# Parameters:
# - df: The dataframe that you wsih to extract the data from. It could be the entire dataframe or the dataframes created based on first and last trending dates.
# - col: views, likes, dislikes, comment_count
#     - Prescribe the parameter based on which the top_10 and bottom_10 function will parse the data
# - num: (default=10 or 5)
#     - The number of observations you wish to view
# - channel: (for channel_stats) 
#     - The channel_title for which you would like to see the stats
# - arrange_by: views, likes, dislikes, comment_count (default='views')
#     - The parameter based on which you would like to arrange the data
# - category: boolean, optional (default=False)
#     - Set True to print category with the highest number of trending videos on that date
# - publish_to_trend_time: boolean, optional (default=False)
#     - Set True to include a column in the df with the number of days it to took for the video to trend

# ### Visualize top 10 by feature
# 
# We'll be using df_last for all these functions as df_last has all the videos with the updates stats while it was trending. Though you can use any dataframe and plug it into these functions for data analysis

# In[ ]:


def top_10(df, col, num=10):
    sort_df = df.sort_values(col, ascending=False).iloc[:num]
    
    ax = sort_df[col].plot.bar()
   
    labels = []
    for item in sort_df['title']:
        labels.append(item[:10] + '...')
        
    ax.set_title(col.upper(), fontsize=16)
    ax.set_xticklabels(labels, rotation=45, fontsize=10)
    
    return sort_df[['video_id', 'title', 'channel_title', col]]


# In[ ]:


top_10(df_last, 'views', 10)


# In[ ]:


top_10(df_last, 'comment_count')


# ### Visualize bottom 10 by feature

# In[ ]:


def bottom_10(df, col, num=10):
    sort_df = df.sort_values(col, ascending=True).iloc[:num]
    
    ax1 = sort_df[col].plot.bar()
    
    labels = []
    for item in sort_df['title']:
        labels.append(item[:10] + '...')
        
    ax1.set_title('Bottom {} {} for videos'.format(num, col))
    ax1.set_xticklabels(labels, rotation=45)
    
    return sort_df[['title', 'channel_title', col]]


# In[ ]:


bottom_10(df_last, 'views')


# ### Function to display channel stats

# In[ ]:


def channel_stats(df, channel, num=5, arrange_by='views'):
    target_df = df.loc[df['channel_title'] == channel].sort_values(arrange_by, ascending=False)[:num]
    
    ax1 = target_df[['views']].plot.bar()
    
    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()
    
    labels = []
    for item in target_df['title']:
        labels.append(item[:15] + '...')
    
    ax1.set_title('Top {} views for channel {} arranged by {}'.format(num, channel, arrange_by))
    ax1.set_xticklabels(labels, rotation=45)
    
    ax2.set_title('Top {} Likes/Dislikes/Comments for channel {} arranged by {}'.format(num, channel, arrange_by))
    ax2.set_xticklabels(labels, rotation=45)
    
    return df.loc[df['channel_title'] == channel]


# In[ ]:


channel_stats(df_last, 'Logan Paul Vlogs', num=10, arrange_by='likes')


# ### Function to find videos by trending date

# In[ ]:


def find_videos_by_trending_date(df, date, num=10, arrange_by='views', category=False):
    
    target_df = df.loc[df['trending_date'] == date][:num].sort_values(arrange_by, ascending=False)
    
    if category==True:
        cat_target = df.loc[df['trending_date'] == date].sort_values(arrange_by, ascending=False)
        cat = cat_target.groupby(['category'])['video_id'].count().sort_values(ascending=False).head()
        print('The categories with the most videos on this trending date:', cat)
    
    ax1 = target_df[['views']].plot.bar()
    
    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()
    
    labels = []
    for item in target_df['title']:
        labels.append(item[:10] + '...')
        
    ax1.set_title('Top {} views for videos trending on date {} arranged by {}'.format(num, date, arrange_by))
    ax1.set_xticklabels(labels, rotation=45)
    
    ax2.set_title('Top {} likes/dislikes/comments for videos trending on date {} arranged by {}'.format(num, date, arrange_by))
    ax2.set_xticklabels(labels, rotation=45)
    
    return target_df


# In[ ]:


find_videos_by_trending_date(df_last, '2017-11-14', 5, category=True)


# ### Function to find videos by publish date

# In[ ]:


def find_videos_by_publish_date(df, date, num=5, arrange_by='views', publish_to_trend_time=False):
    
    target_df = df.loc[df['publish_date'] == date][:num].sort_values(arrange_by, ascending=False)
    
    if publish_to_trend_time==True:
        target_df.insert(6, 'publish_to_trend_time', target_df['trending_date'] - target_df['publish_date'])
    
    ax1 = target_df[['views']].plot.bar()
    
    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()
    
    labels = []
    for item in target_df['title']:
        labels.append(item[:10] + '...')
        
    ax1.set_title('Top {} views for videos published on date {} arranged by {}'.format(num, date, arrange_by))
    ax1.set_xticklabels(labels, rotation=45)
    
    ax2.set_title('Top {} likes/dislikes/comments for videos published on date {} arranged by {}'.format(num, date, arrange_by))
    ax2.set_xticklabels(labels, rotation=45
                       )
    return target_df


# In[ ]:


find_videos_by_publish_date(df_last, '2017-11-13', publish_to_trend_time=True)


# In[ ]:


find_videos_by_publish_date(df_last, '2017-11-10', 2, 'comment_count')


# ### Find videos by Category

# In[ ]:


def find_videos_by_category(df, cat, num=5, arrange_by='views'):
    
    target_df = df.loc[df['category'] == cat][:num].sort_values(arrange_by, ascending=False)
    
    ax1 = target_df[['views']].plot.bar()
    
    ax2 = target_df[['likes', 'dislikes', 'comment_count']].plot.bar()
    
    labels = []
    for item in target_df['title']:
        labels.append(item[:10] + '...')
        
    ax1.set_title('Top {} views for videos in category {} arranged by {}'.format(num, cat, arrange_by))
    ax1.set_xticklabels(labels, rotation=45)
    
    ax2.set_title('Top {} likes/dislikes/comments for videos in category {} arranged by {}'.format(num, cat, arrange_by))
    ax2.set_xticklabels(labels, rotation=45)
    
    return target_df


# In[ ]:


find_videos_by_category(df_last, 'Entertainment', 5)

