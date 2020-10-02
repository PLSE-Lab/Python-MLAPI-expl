#!/usr/bin/env python
# coding: utf-8

# # YouTube Trending Time Series Analysis with Python
# 
# This notebook covers preliminary data cleaning techniques with Python and Pandas, then explores the change in statistics of videos that were trending for multiple days.
# 
# For other data cleaning techniques and exploratory visualizations, check out my other notebook: [kaggle.com/quannguyen135/exploration-with-python](https://www.kaggle.com/quannguyen135/exploration-with-python).

# ## Importing libraries

# In[1]:


import pandas as pd
import numpy as np

import json

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm

from datetime import datetime

matplotlib.rcParams['figure.figsize'] = (10, 10)


# ## Reading in a dataset

# In[2]:


file_name = '../input/USvideos.csv' # change this if you want to read a different dataset
my_df = pd.read_csv(file_name, index_col='video_id')
my_df.head()


# ## Processing the dates
# 
# If we look at the `trending_date` or `publish_time` columns, we see that they are not yet in the correct format of datetime data.

# In[3]:


my_df['trending_date'] = pd.to_datetime(my_df['trending_date'], format='%y.%d.%m')
my_df['trending_date'].head()


# In[4]:


my_df['publish_time'] = pd.to_datetime(my_df['publish_time'], format='%Y-%m-%dT%H:%M:%S.%fZ')
my_df['publish_time'].head()


# In[5]:


# separates date and time into two columns from 'publish_time' column
my_df.insert(4, 'publish_date', my_df['publish_time'].dt.date)
my_df['publish_time'] = my_df['publish_time'].dt.time
my_df[['publish_date', 'publish_time']].head()


# ## Processing data types
# Some columns have their data types inappropriately registered by Pandas. For example, `views`, `likes`, and similar columns only need `int` data type, instead of `float` (to save memory), or `category_id`, a nominal attribute, should not carry `int` data type.
# 
# It is important that we ourselves assign their data types appropriately.

# In[6]:


type_int_list = ['views', 'likes', 'dislikes', 'comment_count']
for column in type_int_list:
    my_df[column] = my_df[column].astype(int)

type_str_list = ['category_id']
for column in type_str_list:
    my_df[column] = my_df[column].astype(str)


# ## Processing `category_id` column
# Here we are adding the `category` column after the `category_id` column, using the `US_category_id.json` file for lookup.

# In[7]:


# creates a dictionary that maps `category_id` to `category`
id_to_category = {}

with open('../input/US_category_id.json', 'r') as f:
    data = json.load(f)
    for category in data['items']:
        id_to_category[category['id']] = category['snippet']['title']

id_to_category


# In[8]:


my_df.insert(4, 'category', my_df['category_id'].map(id_to_category))
my_df[['category_id', 'category']].head()


# ## Handling videos with multiple appearances
# 
# This is where we identify the videos that were trending for multiple days, as those videos will have multiple entries in our dataset. We use the `.index.duplicated()` function to access those duplicated entries.

# In[9]:


mul_day_df = my_df[my_df.index.duplicated()]

print(mul_day_df.shape)
mul_day_df.head()


# Our `mul_day_df` DataFrame now contains all the duplicated entries from our original dataset. Next, we will make a set of the video IDs of these videos, so that each video ID would only appear once in the set. We will see that in the end, we have in total 3172 unique video IDs.

# In[10]:


dup_index_set = list(set(mul_day_df.index))
len(dup_index_set)


# Now, we will use the `value_counts()` function to get the frequency table of the video IDs, and consequently plot a histogram.

# In[11]:


freq_df = my_df.index.value_counts()
freq_df.head()


# In[12]:


freq_df.plot.hist()

plt.show()


# ## Changes over time visualization
# 
# Here we are trying to visualize changes in statistics of a video that trended over multiple days.

# In[13]:


import matplotlib.patches as mpatches

def visualize_change(my_df, my_id):
    temp_df = my_df.loc[my_id]
    
    ax = plt.subplot(111)
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['views'], fmt='b-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['views'], fmt='bo')
    
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['likes'], fmt='g-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['likes'], fmt='go')
    
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['dislikes'], fmt='r-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['dislikes'], fmt='ro')
    
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['comment_count'], fmt='y-')
    ax.plot_date(temp_df['trending_date'].astype(datetime), temp_df['comment_count'], fmt='yo')
    
    patches = [
        mpatches.Patch(color='b', label='Views'),
        mpatches.Patch(color='g', label='Likes'),
        mpatches.Patch(color='r', label='Dislikes'),
        mpatches.Patch(color='y', label='Comments')
    ]
    
    plt.legend(handles=patches)
    
    plt.title(temp_df.iloc[0]['title'])
    
    plt.show()


# In[14]:


# getting the id of the video that trended the longest
top_id = freq_df.index[0]
print(top_id)

visualize_change(my_df, top_id)


# In[15]:


# getting a random video id
sample_id = freq_df.sample(n=1, random_state=4).index
print(sample_id)

visualize_change(my_df, sample_id)


# ## Time Series Analysis
# 
# Here we are only collecting changes (increases) in viewing statistics, specifically for columns `views`, `likes`, `dislikes`, `comment_count`. We are also adding another column, `keep_trending`, to specify whether a specific video will keep trending with the next day with the current day's increases.
# 
# First, we create the extra columns.

# In[16]:


# default values for single-entry videos
my_df['delta_views'] = my_df['views']
my_df['delta_likes'] = my_df['likes']
my_df['delta_dislikes'] = my_df['dislikes']
my_df['delta_comment_count'] = my_df['comment_count']
my_df['keep_trending'] = False
my_df.iloc[:5, -7:]


# In[17]:


# has to have 2 rows or more
def get_delta_stat(video_id):
    temp_df = my_df.loc[video_id]
    
    temp_df.iloc[0, -1] = True

    for row_id in range(1, len(temp_df)):
        temp_df.iloc[row_id, -5] = temp_df.iloc[row_id]['views'] - temp_df.iloc[row_id - 1]['views'] # delta_views
        temp_df.iloc[row_id, -4] = temp_df.iloc[row_id]['likes'] - temp_df.iloc[row_id - 1]['likes'] # delta_likes
        temp_df.iloc[row_id, -3] = temp_df.iloc[row_id]['dislikes'] - temp_df.iloc[row_id - 1]['dislikes'] # delta_dislikes
        temp_df.iloc[row_id, -2] = temp_df.iloc[row_id]['comment_count'] - temp_df.iloc[row_id - 1]['comment_count'] # delta_comment_count
        temp_df.iloc[row_id, -1] = True # keep_trending

    temp_df.iloc[len(temp_df) - 1, -1] = False
    
    return temp_df


# As can be seen above, our `get_delta_stat()` function will first assign viewing statistics to corresponding delta columns for the first row (i.e. the first day that a specific video was trending). For the other days after that, it will compute the increases in viewing statistics from a specific day and the day before. As we mentioned, the `keep_trending` column would take the value `True` except for the last day that a specific video was trending.
# 
# We will now apply the function to our dataset. First, we will see its affect on our `sample_id` video first.

# In[18]:


my_df.loc[sample_id]


# In[19]:


sample_delta_df = get_delta_stat(sample_id)
sample_delta_df[['trending_date', 'views', 'likes', 'dislikes', 'comment_count', 'delta_views', 'delta_likes', 'delta_dislikes', 'delta_comment_count', 'keep_trending']]


# In[20]:


my_df.loc[sample_id] = sample_delta_df
my_df.loc[sample_id][['trending_date', 'views', 'likes', 'dislikes', 'comment_count', 'delta_views', 'delta_likes', 'delta_dislikes', 'delta_comment_count', 'keep_trending']]


# We see that for our `sample_id` video, all of our "delta" statistics are correctly updated, for example:
# - `delta_views` on the first day is the same as `views`
# - `delta_views` on the second day is the increase in `views` from the first day
# - `keep_trending` is `True` except for the last day
# 
# Now, we will apply the function to all the video IDs in our dataset:

# In[21]:


'''for video_id in freq_df[freq_df > 1].index:
    print(video_id)
    my_df.loc[video_id] = get_delta_stat(video_id)

my_df.head()[['trending_date', 'views', 'likes', 'dislikes', 'comment_count', 'delta_views', 'delta_likes', 'delta_dislikes', 'delta_comment_count', 'keep_trending']]'''


# In[ ]:




