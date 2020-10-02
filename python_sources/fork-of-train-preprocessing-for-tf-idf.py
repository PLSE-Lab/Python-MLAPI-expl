#!/usr/bin/env python
# coding: utf-8

# https://www.kaggle.com/chernyshov/logistic-regression-with-preprocessing

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from nltk.tokenize import TweetTokenizer
import datetime
import lightgbm as lgb
import string
import re
from scipy import stats
from scipy.sparse import hstack, csr_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import metrics
from wordcloud import WordCloud
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
pd.set_option('max_colwidth',400)
pd.set_option('max_columns', 50)
import json
import gc
import os


# In[ ]:


train = pd.read_csv('../input/jigsaw-train-preprocessed/train_preprocessed_tfidf.csv').fillna('')


# In[ ]:


train.columns


# In[ ]:


fts = ["raw_word_len", "raw_char_len", "nb_upper", "nb_fk", "nb_sk", "nb_dk", "nb_you", "nb_mother", "nb_ng", "start_with_columns",
       "has_timestamp", "has_date_long", "has_date_short", "has_mail", "has_emphasize_equal", "has_emphasize_quotes", "clean_word_len",
       "clean_char_len", "clean_chars", "clean_chars_ratio"]


# In[ ]:


annot_idx = train[train['identity_annotator_count'] > 0].sample(n=48660, random_state=13).index
not_annot_idx = train[train['identity_annotator_count'] == 0].sample(n=48660, random_state=13).index
x_val_idx = list(set(annot_idx).union(set(not_annot_idx)))


# In[ ]:


val_df = train.loc[x_val_idx]
print(train.shape)
print(val_df.shape)


# In[ ]:


X_train = train.loc[set(train.index) - set(x_val_idx)]


# In[ ]:


train_text = X_train['clean_comment'].apply(lambda x: re.sub('#', '', x))


# In[ ]:


word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        min_df=5,
        ngram_range=(1, 2),
        max_features=60000)

word_vectorizer.fit(train_text)


# In[ ]:


train_word_features = word_vectorizer.transform(train_text)


# In[ ]:


import pickle

with open('word_vectorizer.pickle', 'wb') as handle:
    pickle.dump(word_vectorizer, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


train_word_features.shape


# In[ ]:


X_train['rating'] = X_train['rating'].apply(lambda x: 0 if x == 'rejected' else 1)


# In[ ]:


train_features = hstack([X_train[fts], train_word_features]).tocsr()


# In[ ]:


del X_train
del train_word_features


# In[ ]:


y_train = train.loc[set(train.index) - set(x_val_idx)]['target']
y_train = (y_train >= 0.5).astype('int')


# In[ ]:


lr = LogisticRegression(solver='lbfgs', random_state=13)
lr.fit(train_features, y_train)


# In[ ]:


with open('lr_model.pickle', 'wb') as handle:
    pickle.dump(lr, handle, protocol=pickle.HIGHEST_PROTOCOL)


# In[ ]:


lgb_train = lgb.Dataset(train_features, y_train)

# specify your configurations as a dict
params = {
    'task': 'train',
    'boosting_type': 'gbdt',
    'objective':'binary',
    'metric': {'auc'},
    'nthread': -1,
    'feature_fraction': 0.4,
    'num_leaves': 50,
    'num_iterations': 200,
    'verbose': 1,
}

print('Start training...')
# train
gbm = lgb.train(params,
                lgb_train)


# In[ ]:


with open('gbm_model.pickle', 'wb') as handle:
    pickle.dump(gbm, handle, protocol=pickle.HIGHEST_PROTOCOL)

