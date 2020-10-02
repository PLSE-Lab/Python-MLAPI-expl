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


import json

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold


# In[ ]:


sample = pd.read_csv('/kaggle/input/made2019ml-cluster2/sample.txt')


# In[ ]:


sample.head()


# In[ ]:


sample['doc_id']


# In[ ]:


with open("/kaggle/input/made2019ml-cluster2/cluster_final_cut_train.json", "r") as read_file:
    cluster_final_cut_train = json.load(read_file)


# In[ ]:


cluster_final_cut_train


# In[ ]:


with open("/kaggle/input/made2019ml-cluster2/cosmo_content_storage_final_cut.jsonl", "r") as json_file:
    json_list = list(json_file)

results = [json.loads(jline) for jline in json_list]
data = pd.DataFrame(results)

data.head()


# In[ ]:


data['doc_id']


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndata['cat'] = 0\ndata['cat'] = data['doc_id'].astype(str).map(cluster_final_cut_train)")


# In[ ]:


data.head()


# In[ ]:


print(f"train len: {data[data['cat'].isnull() == False].shape[0]}, test len: {data[data['cat'].isnull()].shape[0]}")


# In[ ]:


data[data['cat'] == 3059]


# In[ ]:


data['description'] = data['description'].fillna('')
data['description_title'] = data['description'] + ' ' + data['title']
data['description_title'] = data['description_title'].str.lower()


# In[ ]:


data.isnull().sum()


# In[ ]:


data.head()


# In[ ]:


data['description_title'][0]


# In[ ]:


train = data[data['cat'].isnull() == False].reset_index(drop=True)
test = data[data['cat'].isnull()].reset_index(drop=True)


# In[ ]:


test.head()


# In[ ]:


def predict_one_class_tf(tfidfed, tfidfed_test, y, model, n_splits=5):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
#    skf = StratifiedShuffleSplit(n_splits=n_splits, random_state=42, train_size=0.8, test_size=0.2)
    to_check = train.copy()
    to_check[f'predict_proba'] = np.zeros(to_check.shape[0])
    pred = np.zeros(len(test))
    metrics = []
    
    for i, (train_index, valid_index) in enumerate(skf.split(tfidfed, y)):
        print(f'fold_{i}')
        
        X_train = tfidfed[train_index]
        y_train = list(y.loc[train_index])

        X_valid = tfidfed[valid_index]
        y_valid = list(y.loc[valid_index])
        
        best_model_skf = model
#        best_model_skf = LogisticRegression(**params)
        
        best_model_skf.fit(
       X_train, y_train,
    #   cat_features=new_cat_features,
    #   verbose=False,
    #   eval_set=(X_valid, y_valid),
    #   logging_level='Silent',
    #   plot=plot
        )
        
        prediction_skf = best_model_skf.predict(tfidfed_test)
        pred += prediction_skf / skf.n_splits  
        
        prediction = best_model_skf.predict(X_valid)
        to_check.loc[valid_index, f'predict_proba'] = prediction
        print(f'accuracy_score_{i}: {accuracy_score(y_valid, prediction)}')
#         metrics.append(log_loss(y_valid, prediction))
        
        print()
        print('_' * 100)
        print()
    
#    local_logloss = log_loss(y, to_check[f'predict_proba'])
#     print(f"log_loss_oof: {local_logloss}")
#     print(f"log_loss_mean: {np.mean(metrics)}")
#     print(f"metric_std: {np.std(metrics, dtype=np.float64)}")
    
    print(f"accuracy_score: {accuracy_score(y, to_check[f'predict_proba'])}")
    best_model_skf = model
    best_model_skf.fit(
       tfidfed, y,
    #   cat_features=new_cat_features,
    #   verbose=False,
    #   eval_set=(X_valid, y_valid),
    #   logging_level='Silent',
    #   plot=plot
        )
    pred = best_model_skf.predict(tfidfed_test)
    print()
    print('*' * 100)
    print()
    
    return pred, to_check


# In[ ]:


params = {
    "count_vectorizer_params": 
    {
       # "max_df": 1000,
        "ngram_range": [1, 1],
        'min_df': 3,
        'binary': False,
        'analyzer': 'word',
        'lowercase': True,
      #  'max_features': 3000
      #  'stop_words': 'english',
        
    }, 
    "tfidf_transformer_params": {
        'norm': 'l2',
        'use_idf': True,
        'smooth_idf': False,
        'sublinear_tf': True
    }, 
    "logistic_regression_params": {
        "random_state": 1337,
        "penalty": "l2",
        "C": 4.65,
        'max_iter': 100,
        'class_weight': 'balanced',
        'solver': 'liblinear',
        'multi_class': 'ovr',
        'n_jobs': -1
    }
}


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nvectorizer_no_stop_train_test = CountVectorizer(**params['count_vectorizer_params'])\n\nword_count_vector_no_stop_train_test = vectorizer_no_stop_train_test.fit_transform(data['description_title'])\n\ntfidf_no_stop_train_test = TfidfTransformer(**params['tfidf_transformer_params'])\n\ntfidfed_no_stop_train_test = tfidf_no_stop_train_test.fit_transform(word_count_vector_no_stop_train_test)")


# In[ ]:


tfidfed_no_stop_train_test.shape


# In[ ]:


model_lr = LogisticRegression(**params['logistic_regression_params'])


# In[ ]:


train_idx = data[data['cat'].isnull() == False].index
test_idx = data[data['cat'].isnull()].index
y = data[data['cat'].isnull() == False]['cat'].reset_index(drop=True)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\npred_tf_no_stop, to_check_tf_no_stop = predict_one_class_tf(tfidfed_no_stop_train_test[train_idx], tfidfed_no_stop_train_test[test_idx], y,\n                                                                                      model_lr, n_splits=3)')


# In[ ]:


test['cat'] = pred_tf_no_stop
sub = sample.copy()
sub = sub.drop(['cat'], axis=1)
sub = sub.merge(test, how='left', on='doc_id')
sub = sub[['doc_id', 'cat']]
sub['cat'] = sub['cat'].astype(int)
sub.to_csv('fit_predict.csv', index=False)


# In[ ]:


sub.head()

