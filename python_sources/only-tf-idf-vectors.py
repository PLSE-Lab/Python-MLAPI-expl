#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from scipy.sparse import csr_matrix

get_ipython().run_line_magic('matplotlib', 'inline')
warnings.filterwarnings('ignore')


# In[ ]:


train_data = pd.read_csv('../input/train.csv')
print (train_data.shape)
train_data.head()


# In[ ]:


test_data = pd.read_csv('../input/test.csv')
print (test_data.shape)
test_data.head()


# In[ ]:


train_data = train_data.drop(['id', 'qid1', 'qid2'], 1)


# In[ ]:


test_data = test_data.drop(['test_id'], 1)


# In[ ]:


train_data.isnull().sum()


# In[ ]:


train_data = train_data.fillna('empty question')


# In[ ]:


test_data.isnull().sum()


# In[ ]:


test_data = test_data.fillna('empty question')


# In[ ]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(analyzer = 'word', stop_words = 'english', lowercase = True, norm = 'l1')


# In[ ]:


train_data_q1_tfidf = tfidf.fit_transform(train_data.question1.values)
train_data_q2_tfidf = tfidf.fit_transform(train_data.question2.values)


# In[ ]:


test_data_q1_tfidf = tfidf.fit_transform(test_data.question1.values)
test_data_q2_tfidf = tfidf.fit_transform(test_data.question2.values)


# In[ ]:


print (train_data_q1_tfidf.shape)
print (train_data_q2_tfidf.shape)


# In[ ]:


train_data_q1_tfidf = csr_matrix((train_data_q1_tfidf.data, train_data_q1_tfidf.indices, train_data_q1_tfidf.indptr), shape=(404290,90824))
train_data_q2_tfidf = csr_matrix((train_data_q2_tfidf.data, train_data_q2_tfidf.indices, train_data_q2_tfidf.indptr), shape=(404290,90824))


# In[ ]:


print (train_data_q1_tfidf.shape)
print (train_data_q2_tfidf.shape)


# In[ ]:


test_data_q2_tfidf = csr_matrix((test_data_q2_tfidf.data, test_data_q2_tfidf.indices, test_data_q2_tfidf.indptr), shape=(2345796,90824))


# In[ ]:


print (test_data_q1_tfidf.shape)
print (test_data_q2_tfidf.shape)


# In[ ]:


X = abs(train_data_q1_tfidf-train_data_q2_tfidf)
y = train_data[['is_duplicate']]


# In[ ]:


X_test = abs(test_data_q1_tfidf-test_data_q2_tfidf)


# In[ ]:


from xgboost import XGBClassifier

xg_model = XGBClassifier()


# In[ ]:


xg_model.fit(X, y)


# In[ ]:


xg_pred = xg_model.predict(X_test)


# In[ ]:


xg_pred = pd.Series(xg_pred, name='is_duplicate')
submission = pd.concat([pd.Series(range(2345796), name='test_id'),xg_pred], axis = 1)
submission.to_csv('xg_tfidf_submission_file.csv', index=False)

