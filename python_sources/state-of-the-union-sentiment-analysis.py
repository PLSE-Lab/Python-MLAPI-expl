#!/usr/bin/env python
# coding: utf-8

# # State of the Union Sentiment Analysis
# A basic approach to analyzing sentiment of State of the Union Addresses.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os

sotu_files = os.listdir("../input/addresses")

# Any results you write to the current directory are saved as output.


# ### First we Import NLTK
# We will use the Natural Language Toolkit's (NLTK) built in sentiment analyzer to do some basic sentiment analysis.

# In[ ]:


import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# nltk.download('vader_lexicon')


# ### Now we Define a Function to get Sentiment Scores
# This function takes a snippet of text, and returns the "polarity scores" for that text.

# In[ ]:


def sotu_sentiment(text):
    sentiment = SentimentIntensityAnalyzer()
    score = sentiment.polarity_scores(text)

    return score


# ### Now we Analyze the Opening Sentence of Every State of the Union Address
# We grab the first sentence of each State of the Union, and run that sentence through 'sotu_sentiment()' to get opening sentence scores.

# In[ ]:


scores = []

for file_name in sotu_files:
    president = file_name.split('[')[0].rstrip()
    date = file_name.split('[')[1].replace(']', '').replace('.txt', '')
    
    # print("Getting Sentiment Score: {} [{}]".format(president, date))
    file_dir = "../input/addresses/{}".format(file_name)
    with open(file_dir, 'r') as f:
        try:
            score = sotu_sentiment(f.read().split('.')[0])
            scores.append([president, date, score['compound'], score['neg'], score['neu'], score['pos']])
        except UnicodeDecodeError:
            pass


# ### List Opening Sentence Sentiment Scores from Most Postive to Most Negative

# In[ ]:


# Opening Sentence Scores Ranked from Most Positive to Most Negative #
scores_df = pd.DataFrame(data=scores, columns=['President', 'Date', 'Compound', 'Positive', 'Neutral', 'Negative']).sort_values(by=['Compound'], ascending=False).reset_index(drop=True)

scores_df


# In[ ]:




