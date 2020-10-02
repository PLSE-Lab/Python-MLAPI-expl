#!/usr/bin/env python
# coding: utf-8

# 1) reading from .csv file
# 2) reading json file and updating category using mapping.
# 3) selecting and finding most popular youtube channel
# 4) Is weekend or Weekdays videos getting viral?
# 5) Which YouTube channel have most viewers video uploads etc?
# 6) Does more video upload gives out the more video views?
# 7) Is there a way to predict the number of subscribers based on the number of video uploaded by the channel and number of video views on it?

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

plt.style.use('ggplot')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


youtubeIN_df = pd.read_csv('/kaggle/input/youtube-new/INvideos.csv')
youtubeIN_df.head(3)


# **Set the index to be the 'Trending Date'**

# In[ ]:


youtubeIN_df.set_index('trending_date', inplace = True)
youtubeIN_df[:3]


# **Selecting a column**

# In[ ]:


youtubeIN_df['views']


# In[ ]:


youtubeIN_df['views'].plot(figsize = (20,8))


# In[ ]:


youtubeIN_df[['likes','dislikes']].plot(figsize = (20,8))


# In[ ]:


youtubeIN_df.columns


# In[ ]:


youtubeIN_df['channel_title'].value_counts()


# **Number of viral videos per channel**

# In[ ]:


channel_title_count = youtubeIN_df['channel_title'].value_counts()[:30]


# In[ ]:


channel_title_count.plot(kind = 'bar', figsize = (15,8))


# **Which Youtube category is most popular?**

# In[ ]:


channelCategory = youtubeIN_df['category_id'].value_counts()
channelCategory


# **Reading JSON file**

# In[ ]:


import json
with open('/kaggle/input/youtube-new/IN_category_id.json', 'r') as f:
    data = json.load(f)
youtubeIN_category = pd.DataFrame(data)

youtubeIN_category.head(5)


# In[ ]:


youtubeIN_category['items'][0]


# In[ ]:


id_to_category = {}

for c in youtubeIN_category['items']:
    id_to_category[int(c['id'])] = c['snippet']['title']
    
id_to_category


# **Dictionary mapping with panda dataframe column**

# In[ ]:


youtubeIN_df['category_title'] = youtubeIN_df['category_id'].map(id_to_category)
youtubeIN_df['category_title'][:3]


# **Which category has more videos**

# In[ ]:


channel_category = youtubeIN_df['category_title'].value_counts()
channel_category


# In[ ]:


channel_category.plot(kind = 'bar', figsize = (20,8))


# In[ ]:


youtubeIN_df = youtubeIN_df.reset_index()
youtubeIN_df[:3]


# In[ ]:


youtubeIN_df = youtubeIN_df.reset_index()


# In[ ]:


youtubeIN_df['trending_date'] =  pd.to_datetime(youtubeIN_df['trending_date'], format='%y.%d.%m')


# In[ ]:


youtubeIN_df['trending_year'] = youtubeIN_df['trending_date'].dt.year
youtubeIN_df['trending_month'] = youtubeIN_df['trending_date'].dt.month
youtubeIN_df['trending_day'] = youtubeIN_df['trending_date'].dt.day
youtubeIN_df['is_weekend'] = youtubeIN_df['trending_day'].apply(lambda x:'weekend' if x > 4 else 'weekday') #1 means weekend


# In[ ]:


category_groupby = youtubeIN_df.groupby(['category_title','trending_year']).sum()
category_groupby                                   


# In[ ]:


category_groupby['views'].plot(kind = 'bar', figsize = (20,5))


# We can cleary see that 
# 1) Comedy
# 2) Entertainment
# 3) Film & Animation
# 4) Howto & Style
# 5) Music
# 6) People & blogs
# 7) Science & Technology
# 8) Sports 
# views increased in year 2018
# 
# Education channels are gaining popularity slowly.
# 

# In[ ]:


category_groupby = youtubeIN_df.groupby(['category_title','is_weekend']).sum()


# In[ ]:


category_groupby['views']


# In[ ]:


category_groupby['views'].plot(kind = 'bar', figsize = (20,5))


# Weekends has more views then weekdays for each category.

# In[ ]:


entertainment_df = youtubeIN_df.loc[youtubeIN_df['category_title'] == 'Entertainment']
entertainment_df


# In[ ]:


entertainment_df.groupby('trending_year')['channel_title'].sum()


# In[ ]:


entertainment_df_2017 = entertainment_df.loc[entertainment_df['trending_year'] == 2017]
entertainment_df_2017


# In[ ]:


entertainment_df_2017.shape[0]


# In 2017, there are 3976 videos published under Entertainment category

# In[ ]:


entertainment_df_2018 = entertainment_df.loc[entertainment_df['trending_year'] == 2018]
entertainment_df_2018


# In[ ]:


entertainment_df_2018.shape[0]


# 12736 videos published in 2018 under Entertainment category

# In[ ]:


ent_popular_2017 = entertainment_df_2017.groupby('channel_title')[['views','likes','dislikes']].sum()[:30]

ent_popular_2017 = ent_popular_2017.sort_values('views', ascending=False)
ent_popular_2017.plot(kind = 'bar', figsize = (20,5))


# In[ ]:


ent_popular_2018 = entertainment_df_2018.groupby('channel_title')[['views','likes','dislikes']].sum()[:30]

ent_popular_2018 = ent_popular_2018.sort_values('views', ascending=False)
ent_popular_2018.plot(kind = 'bar', figsize = (20,5))


# In[ ]:


ent_popular_2017 = entertainment_df_2017.groupby('channel_title')[['views','likes','dislikes']].sum()
ent_channel_count_2017 = (entertainment_df_2017.groupby('channel_title').count())['index']
popular_entertainment_channels_2017 = pd.merge(ent_channel_count_2017, ent_popular_2017, how='right', on=['channel_title'])
popular_entertainment_channels_2017.rename(columns={"index": "video count"}, inplace = True)
popular_entertainment_channels_2017.sort_values('video count', ascending = False)[:50]


# In[ ]:


popular_entertainment_channels_2017.sort_values('video count', ascending = False)[:50].plot(kind = 'bar', figsize = (15,5))


# Video upload count doesn't necessarily bring more views. 
# 'THIRU TV' has highest uploaded video in 2017, however Amit Bhadana has higest views under entertainment category.

# In[ ]:


ent_popular_2018 = entertainment_df_2018.groupby('channel_title')[['views','likes','dislikes']].sum()
ent_channel_count_2018 = (entertainment_df_2018.groupby('channel_title').count())['index']
popular_entertainment_channels_2018 = pd.merge(ent_channel_count_2018, ent_popular_2018, how='right', on=['channel_title'])
popular_entertainment_channels_2018.rename(columns={"index": "video count"}, inplace = True)
popular_entertainment_channels_2018.sort_values('video count', ascending = False)[:50]


# In[ ]:


popular_entertainment_channels_2018.sort_values('video count', ascending = False)[:50].plot(kind = 'bar', figsize = (15,5))

