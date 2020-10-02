#!/usr/bin/env python
# coding: utf-8

# Kaggle automatically imports pandas and numpy. Here we are additionally importing **matplotlib**, and installing **vaderSentiment** and **tweepy**. These will be used throughout this notebook. Running the code below will allow us to use these but also give us the name of the files that we imported so that we can read them below.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().system('pip install vaderSentiment ')
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
get_ipython().system('pip install tweepy ')
import tweepy as tw 


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Running the code cell above gives us the two csv files that we have imported. One is the Tesla stock data from 2010 to 2020. The other is Elon Musk's tweets from 2012 to 2017. Below we are reading in the csv file using the **pandas function .read_csv()** and assigning it to a variable. By then calling that variable it prints out a dataframe of what the file contains.

# In[ ]:


tesla_stock_df = pd.read_csv('/kaggle/input/tesla-stock-data-from-2010-to-2020/TSLA.csv')
tesla_stock_df


# * Is this data useful? 
# * Is there a way that we can visualize this data? 
#     * We can plot a few columns of the data frame to give us a meaningful graph of the prices or volume by time. Below we are plotting the Close prices column by the date (time).

# In[ ]:


tesla_stock_df.plot(x='Date', y='Close')
plt.xticks(rotation=45)


# * Here we are **reading in the dataset of Elon Musk's (owner of Tesla) tweets** between November 2012 and September 2017

# In[ ]:


em_tweets = pd.read_csv("/kaggle/input/elon-musks-tweets/data_elonmusk.csv", encoding = "ISO-8859-1")
em_tweets


# This dataframe gives us lots of relevant information for what we can do. We can do sentiment analysis on Elon Musk's tweets user VADER. This will give us the breakdown of how positive, negative, and neutral the tweet is. 
# * Might this compare to the Tesla stock data plot above?

# In the code cell below, we are making a new dataframe which contains the "compound" item of sentiment analysis, which is a value between -1 and 1 that essentially tells us how positive or how negative a tweet is, 0 is neutral. The second column of the dataframe is time.

# In[ ]:


analyzer = SentimentIntensityAnalyzer()
tweets_list = [[analyzer.polarity_scores(tweet)['compound']] for tweet in em_tweets.Tweet]
em_sentiment = pd.DataFrame(tweets_list, columns=['sentiment_analysis'])

time = em_tweets['Time']
em_sentiment = em_sentiment.join(time)
em_sentiment


# In order for the graph to be readable and smooth, we can take a rolling average of the compound values. Here we have added a new column called "average" to the dataframe so that we can easily plot it by time.

# In[ ]:


em_sentiment['average'] = em_sentiment.iloc[:,0].rolling(window=200).mean()
em_sentiment


# In[ ]:


em_sentiment.plot(x='Time', y='average')

#rotating the x labels so that they are visible 
plt.xticks(rotation=45)


ax = plt.gca() #gca means get current axis 
# here we are inverting the graph/x-axis so that it will be increasing in time
ax.invert_xaxis()


# * Is this graph comparable to the Tesla stock data at all? 
# * Could this data be useful in any way?

# Additionally, we can get a feel for the overall positivity, negatively, and neutrality of his tweets. This can be accomplished using a bar graph.

# In[ ]:


pos_count = 0
neg_count = 0
neu_count = 0
for tweet in em_tweets.Tweet:
    if(analyzer.polarity_scores(tweet)['compound'] >= 0.05):
        pos_count += 1 
    elif(analyzer.polarity_scores(tweet)['compound'] <= - 0.05):
        neg_count += 1
    else: 
        neu_count += 1
s = ['positive', 'negative', 'neutral']
c = [pos_count, neg_count, neu_count]

plt.bar(s, c)

        


# The csv file above only gives us the tweets between 2012 and 2017. We obtain Musk's live tweets through access to the Twitter API. From this we can get the tweets from 2017 onward and again to sentiment analysis on those. 
# 
# * The code cell below sets up the Twitter API and the date since and date until are variable
# * Pulling the tweets from 2017 onward, we can create a new dataframe which nicely puts this data in tabular form

# In[ ]:


consumer_key = 'iLUAvVcsjcuO8Ixg1RTbxQanB'
consumer_secret = '6uT673QxmbGnmjSBc9WYhmPk4XwCl2xzhjjkknJ5nJFZ4Am9QX'
access_token = '1268544242613391362-9qJQupKaUtRMemjXhnpX7r1MBIVrKk'
access_token_secret = '857KDyCnA6x4d22gMqEPbXz9d1NOejJgDp03874HMacrL'
auth = tw.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tw.API(auth, wait_on_rate_limit=True)

date_since1 = '2017-09-30'
date_until1 = '2020-06-21'

tweets = tw.Cursor(api.user_timeline, id='elonmusk', since=date_since1, until=date_until1).items()

tweets_now_list = [[tweet.text, tweet.created_at] for tweet in tweets]
tweets_now_df = pd.DataFrame(tweets_now_list, columns=['text', 'time'])

tweets_now_df


# In[ ]:


now_list = [[analyzer.polarity_scores(tweet)['compound']] for tweet in tweets_now_df.text]
senti_now = pd.DataFrame(tweets_list, columns=['sentiment_analysis'])

time = tweets_now_df['time']
senti_now = senti_now.join(time)
senti_now


# In[ ]:


senti_now['average'] = senti_now.iloc[:,0].rolling(window=200).mean()
senti_now.plot(x='time', y='average')


# In[ ]:


date_since = '2017-09-01'
date_until = '2020-06-19'

p_count = 0
ng_count = 0
nu_count = 0

for tweet in tw.Cursor(api.user_timeline,id='elonmusk', since=date_since, until=date_until).items():
    if analyzer.polarity_scores(tweet.text)['compound'] >= 0.05:
        p_count += 1
    elif analyzer.polarity_scores(tweet.text)['compound'] <= -0.05:
        ng_count += 1
    else: 
        nu_count += 1
se = ['positive', 'negative', 'neutral']
co = [p_count, ng_count, nu_count]
plt.bar(se, co)

