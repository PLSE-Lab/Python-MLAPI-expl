#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np
import os
import pandas as pd
import sys
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression,SGDClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB,BernoulliNB
from sklearn.svm import SVC,LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import PorterStemmer
import nltk
from nltk import word_tokenize, ngrams
from nltk.classify import SklearnClassifier
from wordcloud import WordCloud,STOPWORDS
import xgboost as xgb
np.random.seed(25)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


ca_data = pd.read_csv("../input/CAvideos.csv")
de_data = pd.read_csv("../input/DEvideos.csv")
fr_data = pd.read_csv("../input/FRvideos.csv")
US_data = pd.read_csv("../input/USvideos.csv")


# # Let's explore Canada Region

# In[ ]:


ca_data.head()


# Let's check if there is any null value.

# In[ ]:


ca_data.isnull().sum(axis=0)


# There are few videos without any description so let's fill those null values.

# In[ ]:


ca_data['description'].fillna("Not available", inplace=True)


# In[ ]:


ca_data.dtypes


# In[ ]:


col = ['category_id', 'views', 'likes', 'dislikes','comment_count']

ca_data[col].describe()


# Let's see which video got most *popularity* in Canada.

# In[ ]:


ca_data.loc[ca_data['views'].idxmax()]


# It is **YouTube Rewind: The Shape of 2017** which got most views, likes as well as dislikes by the people of Canada.
# 
# Let's see which one got *least* views.

# In[ ]:


ca_data.loc[ca_data['views'].idxmin()]


# It was **Canadian Olympian receives death threats from South Koreans**. 
# 
# Let's see which video was most discussed amongst people.

# In[ ]:


ca_data.loc[ca_data['comment_count'].idxmax()]


# Hmmmm. Same video with different figures. That means there are some videos which have duplicate ids. We have to figure them out.

# # Let's do some Visualization

# Let's see which are the top 10 channels by most/least number of videos.

# In[ ]:


ca_data['channel_title'].value_counts().head(10).plot.barh()


# **The Young Turks** published highest number of videos.

# In[ ]:


ca_data['channel_title'].value_counts().tail(10).plot.barh()


# Top/Bottom 10 *category_id*

# In[ ]:


ca_data['category_id'].value_counts().head(10).plot.barh()


# In[ ]:


ca_data['category_id'].value_counts().tail(10).plot.barh()


# To be continued.....

# In[ ]:




