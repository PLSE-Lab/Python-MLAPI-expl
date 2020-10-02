#!/usr/bin/env python
# coding: utf-8

# **Hello again Kagglers. This is my in-continuation Data Analysis from my previous [NOTEBOOK](https://www.kaggle.com/aviralpamecha/text-processing-similarity-and-sentiment-analysis). You can have a look at this Notebook for Sentiment Analysis and Text Similartiy. 
# I will list my major Points I inferred from visualization of data of all the Episodes at last of this Notebook.**[](http://)

# In[ ]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


episodes = pd.read_csv('/kaggle/input/chai-time-data-science/Episodes.csv')


# # DATA ANALYSIS

# LEST FIRST HAVE A LOOK THAT HOW MUCH NULL DATA WE HAVE.

# # Data Cleaning

# In[ ]:


plt.figure(figsize=(10,10)) 
sns.heatmap(episodes.isnull(), yticklabels=False, cbar=False, cmap='viridis')


# ABOVE VISUALIZATION CLEARLY SHOWS WE HAVE BUNCH OF NULL DATA

# In[ ]:


len(episodes)


# In[ ]:


episodes.drop(index=[46,47,48,49,50,51,52,53,54], inplace=True)


# In[ ]:


episodes.head()


# In[ ]:


episodes.isnull().sum()


# In[ ]:


def impute_username(cols):
    name = cols[0]
    if pd.isnull(name):
        
        return 'unknown'
    else:
        return name


# In[ ]:


def impute_num(cols):
    num = cols[0]
    if pd.isnull(num):
        return 0
    else:
        return num


# In[ ]:


episodes['heroes_kaggle_username'] = episodes[['heroes_kaggle_username']].apply(impute_username, axis=1)


# In[ ]:


episodes['heroes_twitter_handle'] = episodes[['heroes_twitter_handle']].apply(impute_username, axis=1)


# In[ ]:


episodes['heroes'] = episodes[['heroes']].apply(impute_username, axis=1)


# In[ ]:


episodes['heroes_gender'] = episodes[['heroes_gender']].apply(impute_username, axis=1)


# In[ ]:


episodes['heroes_location'] = episodes[['heroes_location']].apply(impute_username, axis=1)


# In[ ]:


episodes['heroes_nationality'] = episodes[['heroes_nationality']].apply(impute_username, axis=1)


# In[ ]:


episodes['anchor_url'] = episodes[['anchor_url']].apply(impute_username, axis=1)


# In[ ]:


episodes['anchor_thumbnail_type'] = episodes[['anchor_thumbnail_type']].apply(impute_username, axis=1)


# In[ ]:


episodes['anchor_plays'] = episodes[['anchor_plays']].apply(impute_num, axis=1)
episodes['spotify_starts'] = episodes[['spotify_starts']].apply(impute_num, axis=1)
episodes['spotify_streams'] = episodes[['spotify_streams']].apply(impute_num, axis=1)
episodes['spotify_listeners'] = episodes[['spotify_listeners']].apply(impute_num, axis=1)
episodes['apple_listeners'] = episodes[['apple_listeners']].apply(impute_num, axis=1)


# In[ ]:


episodes['apple_avg_listen_duration'] = episodes[['apple_avg_listen_duration']].apply(impute_num, axis=1)


# In[ ]:


episodes['apple_avg_listen_duration'] = episodes[['apple_avg_listen_duration']].apply(impute_num, axis=1)


# In[ ]:


episodes['apple_listened_hours'] = episodes[['apple_listened_hours']].apply(impute_num, axis=1)


# AS THERE ARE VERY FEW ROWS SO WE DIDN'T DROPPED ANY ROWS. TO SAVE DATA WE FILLED NULL ENTRIES WITH 'UNKNOWN'

# In[ ]:


episodes.isnull().sum()


# In[ ]:


episodes['heroes_location'].unique()


# In[ ]:


plt.figure(figsize=(25,20)) 
sns.countplot(x='heroes_location',  data=episodes)


# CLEAR FROM ABOVE VISUALIZATION THAT MOSTLY THE GUESTS ARE FROM UNITED STATES OF AMERICA(U.S.A)

# In[ ]:


plt.figure(figsize=(25,20)) 
sns.countplot(x='heroes_nationality',  data=episodes)


# ONE THING TO NOTE HERE, NATIONALITY IN ABOVE VISUAL OF INDIANS ROSE, WHICH MEANS MANY INDIANS AS DATA SCIENTISTS ARE WORKING ABROAD.

# In[ ]:


plt.figure(figsize=(15,10)) 
sns.countplot(x='flavour_of_tea',  data=episodes)


# MASALA CHAI AND GINGER CHAI WAS THE MOST FAMOUS CHAI IN THE SHOW.

# In[ ]:


episodes['apple_avg_listen_duration']


# In[ ]:





# In[ ]:


sns.countplot(x='category',  data=episodes)


# In[ ]:


plt.figure(figsize=(25,20)) 

sns.countplot(x='heroes_nationality',data=episodes,hue='heroes_gender')


# THE RATIO OF MALES TO FEMALES IS QUITE HIGH IN DATA SCIENCE

# In[ ]:


sns.scatterplot(x='episode_duration',y='youtube_impression_views',data=episodes)


# What can be inferred from here is, viewers generally prefer MEDIUM LENGTH episodes.

# In[ ]:


sns.scatterplot(x='episode_duration',y='youtube_avg_watch_duration',data=episodes,)


# From above visual also it is clearly visible that most viewers want the episodes of half the length of current duration

# In[ ]:


sns.scatterplot(x='youtube_avg_watch_duration',y='youtube_likes',data=episodes,)


# In[ ]:


plt.figure(figsize=(20,20))
sns.heatmap(episodes.corr(),annot=True,cmap='viridis')


# IN THE ABOVE VISUAL WE CAN SEE THE DEPENDENCY OR CORRELATION OF ONE FEATURE OVER ANOTHER.

# In[ ]:





# In[ ]:


episodes.corr()


# In[ ]:


episodes.corr()['episode_duration'].sort_values()


# In[ ]:


episodes.corr()['youtube_avg_watch_duration'].sort_values()


# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(episodes['episode_duration'], color='Green',  bins = 30 )


# The above DISTPLOT shows that majorly all episodes have duration near to 4000, which is good for viewers. 

# In[ ]:


plt.figure(figsize=(10,7))
sns.distplot(episodes['apple_avg_listen_duration'], color='Blue')


# # MAJOR POINTS OF INFERRENCE:
# 
# 
# 
# * WE HAVE BUNCH OF NULL ENTERIES SO DATA NEEDS TO BE CLEANED AND PROCESSED.
# 
# * MOSTLY THE GUESTS ARE FROM UNITED STATES OF AMERICA(U.S.A). 
# 
# * ONE THING TO NOTE HERE, MANY INDIANS AS DATA SCIENTISTS ARE WORKING ABROAD.
# 
# * MASALA CHAI AND GINGER CHAI WAS THE MOST FAMOUS CHAI IN THE SHOW.
# 
# * THE RATIO OF MALES TO FEMALES IS QUITE HIGH IN DATA SCIENCE.
# 
# * VIEWERS GENERALLY PREFER MEDIUM LENGTH EPISODES.
# 
# * MAJORLY ALL EPISODES HAVE DURATION NEAR TO 4000, WHICH IS GOOD AND APPRECIABLE BY VIEWERS. 

# In[ ]:




