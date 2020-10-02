#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import nltk

from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[2]:


train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")

train_comments = train_df['comment_text']
test_comments = test_df['comment_text']

all_comments = pd.concat([train_comments, test_comments])

train_df.head(3)


# In[4]:


labels = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# In[6]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[8]:


vectorizer = TfidfVectorizer(
    analyzer='word', 
    sublinear_tf=True,
    strip_accents='unicode',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    max_features=10000)


# In[9]:


print('Start Fit vectorizer')

tfidf = vectorizer.fit(train_comments)

print('Fit vectorizer')


# In[10]:


print('Start transform test comments')

test_comment_features = vectorizer.transform(test_comments)

print('Transformed test comments')


# In[11]:


print('Start transform train comments')

train_comment_features = vectorizer.transform(train_comments)

print('Transformed train comments')


# In[15]:


import pickle

pickle.dump(tfidf, open("tfidf.pickle", "wb"))
pickle.dump(train_comment_features, open("train_comment_features.pickle", "wb"))
pickle.dump(test_comment_features, open("test_comment_features.pickle", "wb"))


# In[ ]:





# In[ ]:




