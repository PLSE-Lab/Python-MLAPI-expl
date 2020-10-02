#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pylab as plt # for plotting, pylab is very similiar to pyplot

# following examples as shown in:
# https://www.kaggle.com/dster/two-sigma-news-official-getting-started-kernel
from kaggle.competitions import twosigmanews
env = twosigmanews.make_env()

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Loading in the data

# In[ ]:


(df_market, df_news) = env.get_training_data()
# df_market = pd.read_csv('../input/marketdata_sample.csv', sep=',')
# df_news = pd.read_csv('../input/news_sample.csv', sep=',')


# In[ ]:


df_market


# In[ ]:


df_news


# # Data Processing
# 
# After loading in the date we process it into something more usefull. First for the market data we will combine the different measures of value into just one. This is expresive enough to do some elementary analysis and makes it much more easy to understand what it going on. 

# ## Market
# First we look at the stock data

# In[ ]:


df_market['price'] = np.mean((df_market['close'].values, df_market['open'].values), axis=0)
df_market = df_market.drop(['close', 'open',
                           'assetCode', 'universe',#asset code not important, dont know what universe means in this context
                           'returnsClosePrevRaw1', # don't know difference between Raw and Mktres, also dont know how relevant
                           'returnsOpenPrevRaw1',
                           'returnsClosePrevMktres1',
                           'returnsOpenPrevMktres1',
                           'returnsClosePrevRaw10',
                           'returnsOpenPrevRaw10',
                           'returnsClosePrevMktres10',
                           'returnsOpenPrevMktres10',
                           'returnsOpenNextMktres10'], axis=1)


# In[ ]:


df_market


# ## News
# Second we look at the news

# In[ ]:


df_news = df_news.drop(['noveltyCount12H','noveltyCount24H','noveltyCount3D','noveltyCount5D', 'noveltyCount7D',
                       'volumeCounts12H', 'volumeCounts24H', 'volumeCounts3D', 'volumeCounts5D', 'volumeCounts7D'], axis=1)


# In[ ]:


df_news['sentiment'] = df_news['sentimentNeutral'].values - df_news['sentimentNegative'].values + df_news['sentimentPositive'].values


# In[ ]:


df_news = df_news.drop(['sentimentNeutral', 'sentimentNegative', 'sentimentPositive',
                       'sentimentClass', # we dont need the sign of the sentiment as we use a scalar. Have to investigave is a scalar is precise enough vs just using the sign. 
                       'sentenceCount', # wordCount should contain at least similiar information as sentenceCount
                       'assetCodes', # we use assetName
                       'sourceTimestamp', 'firstCreated', 'sourceId', 'takeSequence', 'firstMentionSentence'], axis=1)
# df_news


# We plot bodySize vs wordCount to see if they are correlated, if yes we can drop one as they should include very similiar information.

# In[ ]:


df = pd.read_csv('../input/news_sample.csv', sep=',')

plt.plot(df['bodySize'])
plt.plot(df['wordCount'])
plt.legend(['bodySize', 'wordCount'])
plt.show()


# As we can see in the plot above indeed bodySize and wordCount are both a measure of the same thing. We can do four things: do nothing, remove bodyCount, remove wordCount, and combine them both into one feature. For sake of simplimplicity we drop bodySize.

# In[ ]:


df_news = df_news.drop(['bodySize'], axis=1)
# df_news


# ## Combine market and news

# In[ ]:


df_market


# In[ ]:


df_news


# In[ ]:


df = pd.merge(df_market, df_news, on='assetName')
df #seems to only merge a small part, need to figure out how to get more then 100 rows in each dataframe


# In[ ]:




