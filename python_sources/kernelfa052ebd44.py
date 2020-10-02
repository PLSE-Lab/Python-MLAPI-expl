#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd

initial_data = pd.read_csv('../input/twitter_kabir_singh_bollywood_movie.csv', delimiter=',')
initial_data.head()


# ### Let's see how many Tweeter users comment the movie

# In[ ]:


initial_data['author'].value_counts()


# The audience is large.
# 
# **8685 different users** commented on the movie.
# 
# The most active ones tweeted more than **30 times.**

# ### Now  let's try to analyse sentiments of the tweets

# In[ ]:


text = initial_data['text_raw']
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names

def word_feats(words):
    return dict([(word, True) for word in words])
 
positive_vocab = [ 'awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great']
negative_vocab = [ 'bad', 'terrible','useless', 'hate']
neutral_vocab = [ 'movie','the','sound','was','is','actors','did','know','words','not']

positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

train_set = positive_features + negative_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set)


# In[ ]:


total_text = pd.DataFrame()
total_text['text'] = text
total_text['sentiment'] = 'neu'


for i in range(total_text.shape[0]):
    words = total_text['text'][i].split(' ')
    pos = 0
    neg = 0
    for word in words:
        classResult = classifier.classify(word_feats(word))
        if classResult == 'neg':
            neg = neg + 1
        if classResult == 'pos':
            pos = pos + 1
    if pos > neg:
        total_text['sentiment'][i] = 'pos'
    if neg > pos:
        total_text['sentiment'][i] = 'neg'
        
print(total_text.head())


# In[ ]:


total_text['sentiment'].value_counts()


# We can see that **positive** tweets prevail

# In[ ]:


total_text['favorite_count'] = initial_data['favorite_count']
total_text['reply_count'] = initial_data['reply_count']

total_text.sort_values(by=['favorite_count'], ascending = False).head(10)


# In[ ]:


total_text.loc[total_text['sentiment']=='neg'].sort_values(by=['favorite_count'], ascending = False).head(10)


# In[ ]:


total_text.sort_values(by=['reply_count'], ascending = False).head(10)


# In[ ]:


total_text.loc[total_text['sentiment']=='neg'].sort_values(by=['reply_count'], ascending = False).head(10)


# We can conclude that positive tweets induce significantly more favorites than negative ones. The most popular tweet has 7822 favorites and 175 replies while the most important negative tweet has only 60 favorites and only 3 replies.
# 
# In general, negative reviews have dramatically less favorites and shorter discussions (less replies).  
# 
# 

# In[ ]:




