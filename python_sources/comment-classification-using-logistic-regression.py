#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# In[2]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

train = pd.read_csv('../input/train.csv').fillna(' ')
test = pd.read_csv('../input/test.csv').fillna(' ')


# ### Train, Test Text

# In[31]:


train_comment_text = train['comment_text']
test_comment_text = test['comment_text']
comment_text = pd.concat([train_comment_text, test_comment_text])


# ### Tf-IDf vectorizer - Converting Text to Numeric Features

# In[32]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    stop_words='english',
    ngram_range=(1, 1),
    max_features=10000)
word_vectorizer.fit(comment_text)
train_word_features = word_vectorizer.transform(train_comment_text)
test_word_features = word_vectorizer.transform(test_comment_text)


# ### Import Test id in Submission File 

# In[42]:


submission_file = pd.DataFrame.from_dict({'id': test['id']})


# In[43]:


train.head()


# ### Logistic Regression Classifier

# In[46]:


for class_name in class_names:
    train_target = train[class_name]
    classifier = LogisticRegression(C=0.1, solver='sag')
    classifier.fit(train_word_features, train_target)
    submission_file[class_name] = classifier.predict_proba(test_word_features)[:, 1]


# ### Submission File

# In[47]:


submission_file

