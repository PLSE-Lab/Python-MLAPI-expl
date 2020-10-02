#!/usr/bin/env python
# coding: utf-8

# **Loading necessary modules**

# In[ ]:



import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from scipy.sparse import hstack
from scipy.special import logit, expit
from sklearn.ensemble import ExtraTreesClassifier


# **Loading datasets**

# In[ ]:



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# **Cheking dataset**

# In[ ]:


print(train.shape,test.shape)


# In[ ]:


print(test.head())


# In[ ]:


print(train.head())


# In[ ]:


print(train.info())
print()
print(test.info())


# In[ ]:


print(train['comment_text'].value_counts())


# In[ ]:


length = train['comment_text'].str.len()
length.describe()


# **Extracting comment from train and test dataset and creating a new dataframe containing all comment**

# In[ ]:


train_comment_text = train['comment_text']
test_comment_text = test['comment_text']
COMMENT_TEXT = pd.concat([train_comment_text, test_comment_text])


# **Using TfidfVectorizer**

# In[ ]:


word_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 1),
    stop_words = 'english',
    max_features=20000)
word_vectorizer.fit(COMMENT_TEXT)
train_word_features = word_vectorizer.transform(train_comment_text)
test_word_features = word_vectorizer.transform(test_comment_text)


# In[ ]:


char_vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    ngram_range=(1, 5),
    max_features=25000)
char_vectorizer.fit(COMMENT_TEXT)
train_char_features = char_vectorizer.transform(train_comment_text)
test_char_features = char_vectorizer.transform(test_comment_text)

train_features = hstack([train_char_features, train_word_features])
test_features = hstack([test_char_features, test_word_features])


# In[ ]:


print(train_features.shape,test_features.shape)


# In[ ]:


class_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
predictions = {'id': test['id']}
for class_name in class_names:
    train_target = train[class_name]
    #classifier = LogisticRegression(solver='sag')
    classifier =  ExtraTreesClassifier(n_jobs=-1, random_state=3)
    classifier.fit(train_features, train_target)
    predictions[class_name] = classifier.predict_proba(test_features)[:, 1]
submission = pd.DataFrame.from_dict(predictions)
submission.to_csv('submission_log.csv', index=False)


# In[ ]:




