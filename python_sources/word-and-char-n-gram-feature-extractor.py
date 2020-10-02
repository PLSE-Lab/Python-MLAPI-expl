#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from scipy.sparse import save_npz

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.datasets import dump_svmlight_file
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
import gc

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')

train_text = train['comment_text']
test_text = test['comment_text']
all_text = pd.concat([train_text, test_text])


# In[ ]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(all_text)
train_word_features = word_vectorizer.transform(train_text)
test_word_features = word_vectorizer.transform(test_text)


# In[ ]:


train_word_features.shape


# In[ ]:


test_word_features.shape


# In[ ]:


np.ones((159571,))


# In[ ]:





# In[ ]:


dump_svmlight_file(train_word_features, y=np.ones((159571,)), f='train_word_features')


# In[ ]:


dump_svmlight_file(test_word_features, y=np.zeros((153164,)), f='test_word_features')


# In[ ]:


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=50000)
char_vectorizer.fit(all_text)
train_char_features = char_vectorizer.transform(train_text)
test_char_features = char_vectorizer.transform(test_text)


# In[ ]:


dump_svmlight_file(train_char_features, y=np.ones((159571,)), f='train_char_features')
dump_svmlight_file(test_char_features, y=np.zeros((153164,)), f='test_char_features')


# In[ ]:


np.save('target', train[class_names].values)


# In[ ]:




