#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')

tweets = pd.read_csv('../input/twcs/twcs.csv')
inbound_tweets = tweets[tweets.inbound == True]
inbound_tweets['timestamp'] = pd.to_datetime(inbound_tweets['created_at']).dt.date

#inbound_tweets = inbound_tweets.head(1000)


# Explore the dataset for inbound tweets:

# In[ ]:


display(inbound_tweets.head())
display(inbound_tweets.columns)


# Collect quantities by author:

# In[ ]:


count_by_author_id = inbound_tweets.groupby(['author_id'])     .count()[['tweet_id']]     .sort_values(['tweet_id'], ascending = False)     .rename({'tweet_id': 'qty'}, axis='columns')
plt.figure()
plt.plot(np.arange(len(count_by_author_id.index.values)), count_by_author_id.qty )
plt.xlabel('Author #')
plt.ylabel('Tweets Qty.')
plt.grid(True)
plt.show()


# Quantities by Date:

# In[ ]:


count_by_date = inbound_tweets.groupby(['timestamp'])     .count()[['tweet_id']]     .sort_values(['timestamp'], ascending = True)     .rename({'tweet_id': 'qty'}, axis='columns')

display(count_by_date.head())
plt.figure()
plt.plot(count_by_date.index.values, count_by_date.qty )
plt.xlabel('Date')
plt.ylabel('Tweets Qty.')
plt.grid(True)
plt.gcf().autofmt_xdate()
plt.show()


# Process the sentiment analysis using NLTK

# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sentiment_analyzer = SentimentIntensityAnalyzer()

inbound_tweets['score'] = inbound_tweets.text.apply(lambda x: sentiment_analyzer.polarity_scores(x)['compound'])
display(inbound_tweets.head())


# Check the Sentiment Analysis with an histogram of scores

# In[ ]:


hist_bins = 30
plt.figure(figsize=[10,5])
x = inbound_tweets[['score']]
scores_hist,edges = np.histogram(x, bins=hist_bins)
plt.bar(edges[:-1], scores_hist)
plt.grid(True)
plt.xlabel('Score')
plt.ylabel('Tweets Count')
plt.show()


# Define limit for Pos. vs Neg. comment and compute results

# 

# In[ ]:


limit_pos = 0.5
limit_neg = -0.5
limit_neg_per_author = 3
#plt.plot(inbound_tweets.index.values, inbound_tweets['score'], 'r')
#plt.axhline(0.5, 'g.')
inbound_tweets['is_pos'] = (inbound_tweets[['score']] > limit_pos)
inbound_tweets['is_neg'] = (inbound_tweets[['score']] < limit_neg)
inbound_tweets['is_other'] = ~inbound_tweets.is_pos & ~inbound_tweets.is_neg

def count_true(x):
    return np.sum(x == True)

scores_by_date = inbound_tweets.groupby(['timestamp'])     .agg({'tweet_id':'count', 'is_pos': count_true, 'is_neg': count_true, 'is_other': count_true})     .rename({'tweet_id':'total'}, axis='columns')
display(scores_by_date.head())
x = scores_by_date.index.values
y1 = scores_by_date['is_pos']
y2 = scores_by_date['is_neg']

scores_by_author = inbound_tweets.groupby(['author_id'])     .agg({'tweet_id':'count','is_neg': count_true})     .rename({'tweet_id':'total'}, axis='columns')
scores_by_author = scores_by_author.loc[scores_by_author['is_neg'] >= limit_neg_per_author]
display(scores_by_author.head())

#plt.scatter(x, y, alpha=0.5)
plt.figure(figsize=[10,5])
plt.plot(x, y1,'g-', x, y2, 'r--')
plt.xlabel('Date')
plt.ylabel('Qty.')
plt.legend(['Pos.Tweet', 'Neg.Tweet'])
plt.grid(True)
plt.show()


# 
