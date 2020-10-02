#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import spacy
nlp = spacy.load("en_core_web_sm")
import gensim
import matplotlib.pyplot as plt
import plotly
import datetime
import plotly.graph_objects as go
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob 
import re
from collections import Counter
# from allennlp.predictors.predictor import Predictor

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


tweets = pd.read_csv("../input/trump-tweets/trumptweets.csv")


# In[ ]:


tweets['year'] = tweets.date.apply(lambda x: int(x[0:4]))
tweets_filter = tweets[tweets['year'] >= 2016]


# In[ ]:


tweets_filter['formatted_date'] = pd.to_datetime(tweets_filter['date'])
tweets_filter['day_of_year'] = tweets_filter['formatted_date'].apply(lambda x: x.dayofyear)
tweets_filter['week_of_year'] = tweets_filter['formatted_date'].apply(lambda x: x.weekofyear)


# In[ ]:


start_date = datetime.datetime(2016,1,1).date()
dates = []
counts = []
reweets = []
count = 0
for el in tweets_filter.formatted_date.dt.date:
    if (el-start_date).days <= 7:
#         print("entered here")
        count += 1
    else:
        counts.append(count)
        dates.append(start_date.strftime("%Y %b-%d"))
        start_date = (start_date+datetime.timedelta(days = 7))
        count = 1


# In[ ]:


fig = go.Figure(data=go.Scatter(x=dates, y=counts,line=dict(color='firebrick', width=4)))
fig.update_layout(title='No of Tweets by POTUS',
                   xaxis_title='No of Tweets',
                   yaxis_title='Week',
                  xaxis = go.layout.XAxis(
        tickangle = 270))
fig.update_xaxes(nticks=10)
fig.show()


# There was a slight increase in no of tweets per week that peaked in the week of october 4th 2019

# In[ ]:


def clean_tweet(tweet): 
       ''' 
       Utility function to clean tweet text by removing links, special characters 
       using simple regex statements. 
       '''
       return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())
 
 


# In[ ]:


tweets_filter['content'] = tweets_filter.content.apply(clean_tweet)


# In[ ]:


def get_tweet_sentiment(tweet): 
        ''' 
        Utility function to classify sentiment of passed tweet 
        using textblob's sentiment method 
        '''
        # create TextBlob object of passed tweet text 
        analysis = TextBlob(tweet.lower()) 
        # set sentiment 
        if analysis.sentiment.polarity > 0: 
            return 'positive'
        elif analysis.sentiment.polarity == 0: 
            return 'neutral'
        else: 
            return 'negative'


# In[ ]:


get_ipython().run_cell_magic('time', '', "tweets_filter['sentiment'] = tweets_filter.content.apply(get_tweet_sentiment)")


# In[ ]:


get_ipython().run_cell_magic('time', '', 'word_cloud_data = {}\nNER_data = {}\nfor sentiment in tweets_filter.sentiment.unique():\n    data_filter = tweets_filter[(tweets_filter.formatted_date.dt.date >= \\\n                                 datetime.datetime(2017,1,1).date())&\\\n                               (tweets_filter.sentiment == sentiment)]\n    tweetText = data_filter.content.tolist()\n    words = []\n    NER = []\n    for t in tweetText:\n        doc = nlp(t)\n        ner = []\n        for ent in doc.ents:\n            NER.append(ent.label_)\n        for w in t.split():\n            if w.strip().lower() not in STOPWORDS:\n                words.append(w.strip().lower())\n    word_cloud_data[sentiment] = words\n    NER_data[sentiment] = NER')


# In[ ]:


fig = plt.figure(figsize=(12,6))
tweets_filter.sentiment.value_counts().plot(kind = "bar", title = "# Tweets")
plt.xlabel('Sentiment')
plt.ylabel('Tweets')
# plt.title.set_text("No of tweets")


# **The POTUS Tweets positively**

# **A word cloud of tweets by the POTUS**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'fig = plt.figure(figsize=(24,24))\naxes = fig.subplots(nrows=3, ncols=1)\ncounter = 0\nfor row in axes:\n    unique_string = (" ").join(list(word_cloud_data.values())[counter])\n    wordcloud = WordCloud(width = 1500, height = 750, background_color = "white").generate(unique_string)\n    row.title.set_text(list(word_cloud_data.keys())[counter])\n    row.imshow(wordcloud)\n    row.axis("off")\n    counter+=1')


# **A breakdown of Entities the POTUS talks about in the tweets**

# In[ ]:


get_ipython().run_cell_magic('time', '', 'for sentiment in tweets_filter.sentiment.unique():\n    counter = Counter(NER_data[sentiment])\n    labels = list(counter.keys())\n    values = list(counter.values())\n    fig = go.Figure(data=[go.Pie(labels=labels, values=values)])\n    fig.update_layout(title = sentiment)\n    fig.show()')

