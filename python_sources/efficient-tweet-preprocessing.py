#!/usr/bin/env python
# coding: utf-8

# ## ** Tweet Preprocessing **
# 
# Since we are dealing with tweets in this competition, we need to do specific tweet text cleaning along with normal text pre-processing. A tweet may contains
# 
# * URL's
# * Mentions
# * Hashtags
# * Emojis
# * Smileys
# * Spefic words etc..
# 
# To clean the tweet , we can use a python library [tweet-preprocessor](https://pypi.org/project/tweet-preprocessor/) instead of writing the cleaning logic ourself.

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


get_ipython().system('pip install tweet-preprocessor')


# Install tweet-preprocessor using pip

# In[ ]:


import preprocessor as p


# In[ ]:


train_df = pd.read_csv('/kaggle/input/nlp-getting-started/train.csv')


# In[ ]:


train_df.count()


# Dropping duplicates and NaN from the dataframe

# In[ ]:


train_df = train_df.dropna()
train_df = train_df.drop_duplicates()


# In[ ]:


train_df.count()


# In[ ]:


train_df.head()


# Apply tweet preprocessing first. Define a preprocess function and use pandas apply to apply it on each value of 'text'

# In[ ]:


def preprocess_tweet(row):
    text = row['text']
    text = p.clean(text)
    return text


# In[ ]:


train_df['text'] = train_df.apply(preprocess_tweet, axis=1)


# Tweet has been cleaned to normal text.

# In[ ]:


train_df.head()


# Now we can apply normal text preprocessing like
# 
# * Lowercasing
# * Punctuation Removal
# * Replace extra white spaces
# * Stopwords removal
# 
# For stop word removal , i have used [gensim](https://pypi.org/project/gensim/) library

# In[ ]:


from gensim.parsing.preprocessing import remove_stopwords


# In[ ]:


def stopword_removal(row):
    text = row['text']
    text = remove_stopwords(text)
    return text


# In[ ]:


train_df['text'] = train_df.apply(stopword_removal, axis=1)


# In[ ]:


train_df.head()


# Remove extra white spaces, punctuation and apply lower casing

# In[ ]:


train_df['text'] = train_df['text'].str.lower().str.replace('[^\w\s]',' ').str.replace('\s\s+', ' ')


# In[ ]:


train_df.head()


# Now input tweet has been pre-processed and its ready to go for a ML training.
