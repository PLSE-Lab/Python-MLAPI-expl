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
import numpy as np
import scipy as sp
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from textblob import TextBlob, Word
import nltk
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = [line.rstrip() for line in open('../input/amazon_alexa.tsv')]
print (len(df))


# In[ ]:


df = pd.read_csv('../input/amazon_alexa.tsv', sep='\t')
df.head()


# In[ ]:


df.describe().T


# In[ ]:


df.dtypes


# In[ ]:


df.verified_reviews[10]


# In[ ]:


X = df.verified_reviews
y = df.rating


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


vect = TfidfVectorizer(stop_words='english')
dtm = vect.fit_transform(df.verified_reviews)
features = vect.get_feature_names()
dtm.shape


# In[ ]:


review = TextBlob(df.verified_reviews[105])


# In[ ]:


review


# In[ ]:


review.words


# In[ ]:


import nltk
# list the sentences
review.sentences


# In[ ]:



stemmer = SnowballStemmer('english')


# In[ ]:


# stem each word
print ([stemmer.stem(word) for word in review.words])


# **Lemmatization**

# In[ ]:


nltk.download('wordnet')


# In[ ]:


print ([word.lemmatize() for word in review.words])


# In[ ]:


# Function that accepts text and returns a list of lemmas
def split_into_lemmas(text):
    text = text.lower()
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]


# In[ ]:


split_into_lemmas


# In[ ]:


vect = CountVectorizer(analyzer=split_into_lemmas)


# In[ ]:


# Function that accepts a vectorizer and calculates the accuracy
def tokenize_test(vect):
    X_train_dtm = vect.fit_transform(X_train)
    print ('Features: ', X_train_dtm.shape[1])
    X_test_dtm = vect.transform(X_test)
    nb = MultinomialNB()
    nb.fit(X_train_dtm, y_train)
    y_pred_class = nb.predict(X_test_dtm)
    print ('Accuracy: ', metrics.accuracy_score(y_test, y_pred_class))


# In[ ]:


tokenize_test(vect)


# In[ ]:


print (vect.get_feature_names()[-50:])


# In[ ]:


print (vect.get_feature_names()[50:])


# In[ ]:


# polarity ranges from -1 (most negative) to 1 (most positive)
review.sentiment.polarity


# In[ ]:


# define a function that accepts text and returns the polarity
def detect_sentiment(text):
    return TextBlob(text).sentiment.polarity


# In[ ]:


# create a new DataFrame column for sentiment (WARNING: SLOW!)
df['sentiment'] = df.verified_reviews.apply(detect_sentiment)


# In[ ]:


df.boxplot(column='sentiment', by='rating')


# In[ ]:


df[df.sentiment == 1].verified_reviews.head()


# In[ ]:


df[df.sentiment == -1].verified_reviews.head()


# In[ ]:


# negative sentiment in a 5-star review
df[(df.rating == 5) & (df.sentiment < -0.3)].head(1)


# In[ ]:


# positive sentiment in a 1-star review
df[(df.rating == 1) & (df.sentiment > 0.5)].head(1)


# In[ ]:


feature_cols = ['verified_reviews', 'variation', 'feedback', 'sentiment']
X = df[feature_cols]
y = df.rating


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)


# In[ ]:


vect = CountVectorizer()
X_train_dtm = vect.fit_transform(X_train.verified_reviews)
X_test_dtm = vect.transform(X_test.verified_reviews)


# In[ ]:


print (X_train_dtm.shape)
print (X_test_dtm.shape)


# In[ ]:


X_train.drop('verified_reviews', axis=1).shape


# In[ ]:


logreg = LogisticRegression(C=1e9)
logreg.fit(X_train_dtm, y_train)
y_pred_class = logreg.predict(X_test_dtm)
print (metrics.accuracy_score(y_test, y_pred_class))

