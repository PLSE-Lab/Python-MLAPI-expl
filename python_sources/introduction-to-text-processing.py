#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Check input data
get_ipython().system("ls '../input/'")


# ## Import Package

# In[ ]:


import pandas as pd
import numpy as np


# ## Import Data

# In[ ]:


raw_data = pd.read_csv('../input/train.csv')
raw_data.head(10)


# In[ ]:


# Hitung label
raw_data['label'].value_counts()


# ## Preprocessing

# In[ ]:


# First make a function to delete repetitive alphabet
import itertools

def remove_repeating_characters(text):
    return ''.join(''.join(s)[:1] for _, s in itertools.groupby(text))

# Check our function
remove_repeating_characters('oooofel')


# In[ ]:


# Second make a function to remove non alphanumeric
import re

def remove_nonalphanumeric(text):
    text = re.sub('[^0-9a-zA-Z]+', ' ', text)
    return text

# Check our function
remove_nonalphanumeric('o,,,f!!e;;l')


# In[ ]:


# Last make a function to convert string to lower case

def to_lower_case(text):
    return text.lower()

# Check our function
to_lower_case('OFEL')


# In[ ]:


# Make function that combine them all

def preprocessing_text(text):
    text = remove_repeating_characters(text)
    text = remove_nonalphanumeric(text)
    text = to_lower_case(text)
    
    return text

# Check our function
preprocessing_text('Bagus\n\n\nNamun Akan Lebih Baik Apabila Lebih')


# ## Feature Engineering

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer
corpus = [
    'This is the first document.',
    'This document is the second document.',
    'And this is the third one.',
    'Is this the first document?',
]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(corpus)
print(vectorizer.get_feature_names())
print(X.toarray())


# ## Model

# We will use f1_score,for more information watch here: https://en.wikipedia.org/wiki/F1_score

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

def get_kfold():
    return KFold(n_splits=5, shuffle=True, random_state=1)


# In[ ]:


# The variable y is result of your prediction that you want to be score
# So don't worry if you get error
score = cross_val_score(MultinomialNB(), X, y, scoring='f1', cv=get_kfold())


# In[ ]:


score.mean()


# 
