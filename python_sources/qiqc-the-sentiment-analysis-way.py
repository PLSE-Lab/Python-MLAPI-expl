#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk #for NLP processing and sentiment analysis
from nltk.sentiment.vader import SentimentIntensityAnalyzer #for calculating sentiment scores
from textblob import TextBlob #another approach of sentiment analysis
import plotly.plotly as py #For interactive Data Visualization
import plotly.graph_objs as go #For interactive Data Visualization
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")



# In[ ]:


train_df.head()


# In[ ]:


test_df.head()


# **Split The training dataset into two, one with insencere and another with sincere questions:**

# In[ ]:



train1_df = train_df[train_df["target"]==1]
train0_df = train_df[train_df["target"]==0]


# In[ ]:


train1_df.head()
print("Insincere Group shape : ", train1_df.shape)
print("Sincere Group shape : ", train0_df.shape)


# In[ ]:


train0_df.head()


# In[ ]:


train1_df.head()


# **Try the TextBlob sentiment analysis package first, on the insicence group:**

# In[ ]:


train1_df[['polarity','subjectivity']] = train1_df['question_text'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))


# In[ ]:


print(train1_df['polarity'].mean())


# In[ ]:


print(train1_df['subjectivity'].mean())


# A simple Histogram of the polarity distribution, a feature engineered from the TextBlob package, on the Insincere Group, the basic graph (and the average) show that there is a tendency towards *polarity***** within the Insincere group.  The distribution turn towards (-1).

# In[ ]:


hist = train1_df['polarity'].hist()


# In[ ]:


hist = train1_df['subjectivity'].hist()


# In[ ]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
sid = SentimentIntensityAnalyzer()
train1_df['sentiment_scores'] = train1_df['question_text'].apply(sid.polarity_scores)


# In[ ]:


train1_df['sentiment'] = train1_df['sentiment_scores'].apply(lambda x: x['compound'])


# When using the sentiment scores approach **(from nltk.sentiment.vader)**, we find a concentration towards the negative values, and the mean is quite low:** (-0.099)**

# In[ ]:


hist = train1_df['sentiment'].hist()
print(train1_df['sentiment'].mean())


# *To be continued...*****
