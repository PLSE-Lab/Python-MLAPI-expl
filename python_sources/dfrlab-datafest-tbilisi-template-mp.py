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

# Identify the following cases using data analysis techniques.
# 
# - `a` `-->` most active retweeter (user id) (1 points)
# - `b` `-->` most popular tweet --> return tweet id (1 points)
# - `c` `-->` who posted more original tweets (using the tweets dataset) (2 points)
# - `d` `-->` most mentioned author (userid) by the most active account posting either original posts or retweets (3 points)
# - `e` `-->` identify the hashtag that received the most engagement (number of mentions) (3 points)
# - `f` `-->` most common reported location by the top 10 most active accounts posting either original posts or retweets (5 points)

# In[ ]:


# import libraries
import pandas as pd


# In[ ]:


# load datasets
tweet_path = '/kaggle/input/datafest-tbilisi-2020/ghana_nigeria_takedown_tweets.csv'
tweet_data = pd.read_csv(tweet_path, encoding='utf-8', low_memory=False)

users_path = '/kaggle/input/datafest-tbilisi-2020/ghana_nigeria_takedown_users.csv'
users_data = pd.read_csv(users_path, encoding='utf-8')

get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# In[ ]:


tweet_path1 = '/kaggle/input/datafest-tbilisi-2020/ghana_trending_topics_200401-200519.csv'
tweet_data1 = pd.read_csv(tweet_path1, encoding='utf-8', low_memory=False)

# ghana_trending_topics_200401-200519.csv
# nigeria_trending_topics_200401-200519.csv


# # `exploratory data analysis`

# In[ ]:


type(users_data)


# In[ ]:


tweet_data.columns


# In[ ]:


tweet_data1.columns


# In[ ]:


users_data.columns


# In[ ]:


df = pd.DataFrame(
    {
        'column a': ['value 1', 'value 2'],
        'column b': [1, 2]
    }
)


# In[ ]:


df


# In[ ]:


S = pd.Series([1, 2, 3, 4, 5])
S


# In[ ]:


tweet_data.describe()


# In[ ]:


users_data.describe()


# In[ ]:


tweet_data.dtypes


# In[ ]:


users_data.dtypes


# In[ ]:


tweet_data.shape


# In[ ]:


users_data.shape


# In[ ]:


users_data['follower_count'].sum()


# In[ ]:


users_data.tail()


# In[ ]:


tweet_data.head()


# *`most active retweeter (user id) (1 points)`*

# In[ ]:


tweet_data['is_retweet'].unique()


# In[ ]:


tweet_data[tweet_data['is_retweet'] == True].shape


# In[ ]:


tweet_data[tweet_data['is_retweet'] == True]['userid'].value_counts().head()


# In[ ]:


# write result
a = tweet_data[tweet_data['is_retweet'] == True]['userid'].value_counts().head(1)
a


# *`most popular tweet --> return tweet id (1 points)`*

# In[ ]:


tweet_data[tweet_data['retweet_count'].isnull()].shape


# In[ ]:


tweet_data['retweet_count'] = tweet_data['retweet_count'].fillna(0)
tweet_data['like_count'] = tweet_data['like_count'].fillna(0)
tweet_data['quote_count'] = tweet_data['quote_count'].fillna(0)
tweet_data['reply_count'] = tweet_data['reply_count'].fillna(0)


# In[ ]:


tweet_data['popularity'] = tweet_data['retweet_count'] + tweet_data['like_count'] + tweet_data['quote_count'] + tweet_data['reply_count']
tweet_data['popularity'] = tweet_data['popularity'].astype(int)
pop_tweets = tweet_data.groupby('tweetid').agg(
    {
        'popularity': sum
    }
).sort_values(by='popularity', ascending=False)

# result
pop_tweets.head()


# In[ ]:


b = pop_tweets.head(1)


# *`who posted more original tweets (using the tweets dataset) (2 points)`*

# In[ ]:


# your code here


# In[ ]:


tweet_data[tweet_data['is_retweet'] == False]['userid'].value_counts().head()


# In[ ]:


# write result
c = tweet_data[tweet_data['is_retweet'] == False]['userid'].value_counts().head(1)


# In[ ]:





# *`most mentioned author (userid) by the most active account posting either original posts or retweets (3 points)`*

# In[ ]:


# simply most mentioned author
tweet_data['user_mentions'].value_counts().head()


# In[ ]:


d = 'KHGCiIy+FzzobtKolFsF1fS24+dqZx7RmEEXLcNwPjQ='


# In[ ]:


# here will be most mentioned author (userid) by the most active account...:


# In[ ]:


# most active account (by tweets):
tweet_data['userid'].value_counts().head(1)


# In[ ]:


tweet_data['is_retweet'].unique()


# In[ ]:


tweet_data[tweet_data['is_retweet'] == True].shape


# In[ ]:


# most active account (by retweets) (the same author):
tweet_data[tweet_data['is_retweet'] == True]['userid'].value_counts().head()


# In[ ]:


tweet_data[tweet_data['userid'] == '1149814579293241344']['user_mentions'].value_counts().head(2)


# In[ ]:


d = 'KHGCiIy+FzzobtKolFsF1fS24+dqZx7RmEEXLcNwPjQ='


# *`identify the hashtag that received the most engagement (number of mentions) (3 points)`*

# In[ ]:


# your code here


# In[ ]:


tweet_data['hashtags'].head()


# In[ ]:


import ast


# In[ ]:


tweet_data['hashtags_list'] = tweet_data['hashtags'].apply(lambda arg: ast.literal_eval(arg))


# In[ ]:


tweet_data['hashtags_list'].head()


# In[ ]:


hashtags = [h for item in tweet_data['hashtags_list'] for h in item]


# In[ ]:


from collections import Counter


# In[ ]:


Counter(hashtags).most_common()[:1]


# In[ ]:


e = Counter(hashtags).most_common()[:1]


# In[ ]:





# *`most common reported location by the top 10 most active accounts posting either original posts or retweets (5 points)`*

# In[ ]:


# your code here


# In[ ]:


tweet_data['counter'] = 1


# In[ ]:


most_active_accounts = tweet_data.groupby('userid').agg({'counter': sum})     .sort_values(by='counter', ascending=False)     .reset_index()     ['userid'].loc[:9]


# In[ ]:


most_active_accounts


# In[ ]:


most_active_accounts_data = users_data[users_data['userid'].isin(most_active_accounts)]
most_active_accounts_data.shape


# In[ ]:


most_active_accounts_data['user_reported_location'].value_counts()


# In[ ]:


f = most_active_accounts_data['user_reported_location'].value_counts()


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




