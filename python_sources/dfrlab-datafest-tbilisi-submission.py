#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# *`libraries`*

# In[ ]:


import pandas as pd
import ast
from collections import Counter


# *`reading datasets`*

# In[ ]:


tweet_path = '/kaggle/input/datafest-tbilisi-2020/ghana_nigeria_takedown_tweets.csv'
tweet_data = pd.read_csv(tweet_path, encoding='utf-8', low_memory=False)

users_path = '/kaggle/input/datafest-tbilisi-2020/ghana_nigeria_takedown_users.csv'
users_data = pd.read_csv(users_path, encoding='utf-8')


# # `exploratory data analysis`

# In[ ]:


tweet_data.columns


# In[ ]:


users_data.columns


# In[ ]:


tweet_data.dtypes


# In[ ]:


users_data.dtypes


# In[ ]:


tweet_data.shape


# In[ ]:


users_data.shape


# Identify the following cases using data analysis techniques.
# 
# - `a` `-->` most active retweeter (user id) (1 points)
# - `b` `-->` most popular tweet --> return tweet id (1 points)
# - `c` `-->` who posted more original tweets (using the tweets dataset) (2 points)
# - `d` `-->` most mentioned author (userid) by the most active account posting either original posts or retweets (3 points)
# - `e` `-->` identify the hashtag that received the most engagement (number of mentions) (3 points)
# - `f` `-->` most common reported location by the top 10 most active accounts posting either original posts or retweets (5 points)

# In[ ]:





# *`most active retweeter (user id) (1 points)`*

# In[ ]:


tweet_data[tweet_data['is_retweet']==True]['userid'].value_counts().head()


# In[ ]:


# write result
a = 1149814579293241344


# In[ ]:





# *`most popular tweet --> return tweet id (1 points)`*

# In[ ]:


tweet_data['retweet_count'] = tweet_data['retweet_count'].fillna(0)
tweet_data['like_count'] = tweet_data['like_count'].fillna(0)
tweet_data['quote_count'] = tweet_data['quote_count'].fillna(0)
tweet_data['reply_count'] = tweet_data['reply_count'].fillna(0)


# In[ ]:


tweet_data['popularity'] = tweet_data['retweet_count'] + tweet_data['like_count'] + tweet_data['quote_count'] + tweet_data['reply_count']
tweet_data['popularity'] = tweet_data['popularity'].astype(int)


# In[ ]:


tweet_data[tweet_data['tweetid']==1181205733662179328]


# In[ ]:


popular_tweets = tweet_data.groupby('tweetid').agg({'popularity': sum}).sort_values(by='popularity', ascending=False)

popular_tweets.head()


# In[ ]:


b = 1181205733662179328


# In[ ]:





# *`who posted more original tweets (using the tweets dataset) (2 points)`*

# In[ ]:


tweet_data[tweet_data['is_retweet']==False]['userid'].value_counts().head()


# In[ ]:


# write result
c = 1144591876231749632


# In[ ]:





# *`most mentioned author (userid) by the most active account posting either original posts or retweets (3 points)`*

# In[ ]:


tweet_data['userid'].value_counts().head()


# In[ ]:


most_active_user = tweet_data[tweet_data['userid'] == '1149814579293241344']
most_active_user.shape


# In[ ]:


most_active_user.head()


# In[ ]:


tweet_data['user_mentions_list'] = tweet_data['user_mentions'].apply(lambda arg: ast.literal_eval(arg))


# In[ ]:


most_active_user['user_mentions_list'].head(10)


# In[ ]:


user_mentions = [user_mention for row in most_active_user['user_mentions_list'] for user_mention in row]


# In[ ]:


Counter(user_mentions).most_common()[:10]


# In[ ]:


d = 'KHGCiIy+FzzobtKolFsF1fS24+dqZx7RmEEXLcNwPjQ='


# In[ ]:





# *`identify the hashtag that received the most engagement (number of mentions) (3 points)`*

# In[ ]:


tweet_data['hashtags_list'] = tweet_data['hashtags'].apply(lambda arg: ast.literal_eval(arg))
tweet_data['hashtags_list'].head()


# In[ ]:


hashtags = [hashtag for row in tweet_data['hashtags_list'] for hashtag in row]


# In[ ]:


Counter(hashtags).most_common()[:10]


# In[ ]:


e = 'BlackLivesMatter'


# In[ ]:





# *`most common reported location by the top 10 most active accounts posting either original posts or retweets (5 points)`*

# In[ ]:


tweet_data['counter'] = 1


# In[ ]:


most_active_accounts = tweet_data.groupby('userid').agg({'counter': sum})    .sort_values(by='counter', ascending=False)    .reset_index()    ['userid'].loc[:9]

most_active_accounts


# In[ ]:


most_active_accounts_data = users_data[users_data['userid'].isin(most_active_accounts)]
most_active_accounts_data


# In[ ]:


most_active_accounts_data['user_reported_location'].value_counts()


# In[ ]:


f = 'Worldwide; Accra, Ghana; Florida, USA; Washington, USA   all have a count of 1'


# In[ ]:





# *`writing results`*

# In[ ]:


results = pd.Series(
    {
        'a': a,
        'b': b,
        'c': c,
        'd': d,
        'e': e,
        'f': f
    }, name='results'
)

results.index.name = 'key'
results.to_csv('/kaggle/working/results.csv')


# In[ ]:




