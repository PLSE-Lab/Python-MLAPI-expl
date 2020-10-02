#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install vaderSentiment')


# In[ ]:


import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as SIA

analyser = SIA()


# In[ ]:


def readdata():
    return pd.read_csv("/kaggle/input/reuters_data1.csv")


# In[ ]:


# Sentiment Analyzer Scores
def sas(sentence, sentiment):
    score = analyser.polarity_scores(sentence)
    return score[sentiment]


# In[ ]:


def sentiment_lister(sentiment, df):
    trlist = list()
    for sentence in df['processed_header']:
        trlist.append(sas(sentence,sentiment))
    return trlist


# In[ ]:


def add_sentiments(df):
    df['neg'] = sentiment_lister('neg', df)
    df['neu'] = sentiment_lister('neu', df)
    df['pos'] = sentiment_lister('pos', df)
    df['compound'] = sentiment_lister('compound', df)
    
    return df


# In[ ]:


def output(df):
    df.to_csv(r"reuters_data2.csv",header = True)


# In[ ]:


df = readdata()


# In[ ]:


df = add_sentiments(df)


# In[ ]:


output(df)


# In[ ]:


df.head(10)

