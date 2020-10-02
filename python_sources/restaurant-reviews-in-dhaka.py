#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('/kaggle/input/restaurant-reviews-in-dhaka-bangladesh/reviews.csv')


# In[ ]:


df.head()


# In[ ]:


df.isnull().sum()


# In[ ]:


df.head()


# In[ ]:


pd.set_option('display.max_colwidth',150)


# In[ ]:


df.head()


# In[ ]:


import string
import nltk
import re


# In[ ]:


def lower_caes(txt):
    return txt.lower()


# In[ ]:


def remove_punctuation(txt):
    txt_clean = "".join([c for c in txt if c not in string.punctuation])
    return txt_clean


# In[ ]:


df['lower_case'] = df['Review Text'].apply(lambda x: lower_caes(x))


# In[ ]:


df.head()


# In[ ]:


df = df[['lower_case']]


# In[ ]:


df.head()


# In[ ]:


df['review'] = df['lower_case'].apply(lambda x: remove_punctuation(x))


# In[ ]:


df.head()


# In[ ]:


df = df[['review']]


# In[ ]:


df.head()


# In[ ]:


from nltk.sentiment import SentimentIntensityAnalyzer


# In[ ]:


sid = SentimentIntensityAnalyzer()


# In[ ]:


sid.polarity_scores(df['review'].iloc[1])


# In[ ]:


df['scores'] = df['review'].apply(lambda x: sid.polarity_scores(x))


# In[ ]:


df.head()


# In[ ]:


df['compound'] = df['scores'].apply(lambda x: x['compound'])


# In[ ]:


df.head()


# In[ ]:


df['tag'] = df['compound'].apply(lambda x: 'pos' if x>0.15 else 'neg')


# In[ ]:


df.head()


# In[ ]:


df['tag'].value_counts()


# In[ ]:


df = df[['review', 'tag']]


# In[ ]:


df.head()


# In[ ]:


sns.countplot(x = 'tag', data = df)

