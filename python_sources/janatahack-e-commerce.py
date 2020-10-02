#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import re

import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import mean_absolute_error, roc_auc_score, log_loss, accuracy_score

import warnings
warnings.filterwarnings("ignore")

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('/kaggle/input/train.csv')\nprint('train file have {} rows and {} columns'.format(train.shape[0], train.shape[1]))\ntest = pd.read_csv('/kaggle/input/test.csv')\nprint('test file have {} rows and {} columns'.format(test.shape[0], test.shape[1]))\nsubmission = pd.read_csv('/kaggle/input/sample_submission.csv')\nprint('submission file have {} rows and {} columns'.format(submission.shape[0], submission.shape[1]))\n\ntrain['startTime'] = pd.to_datetime(train['startTime'], format='%d/%m/%y %H:%M')\ntrain['endTime'] = pd.to_datetime(train['endTime'], format='%d/%m/%y %H:%M')\n\ntest['startTime'] = pd.to_datetime(test['startTime'], format='%d/%m/%y %H:%M')\ntest['endTime'] = pd.to_datetime(test['endTime'], format='%d/%m/%y %H:%M')\n\ngender_map = {'male': 0, 'female': 1}\nrev_gender_map = {v: k for k, v in gender_map.items()}\n\ntrain['gender'] = train['gender'].map(gender_map) \ndisplay(train['gender'].value_counts(dropna=False, normalize=True))")


# In[ ]:


train.head(2)


# In[ ]:


submission.head(2)


# In[ ]:


all_prod = ';'.join([*train['ProductList'], *test['ProductList']]).split(';')
all_prod = [p.strip('/').split('/') for p in all_prod]
all_prod = pd.DataFrame(all_prod, columns=['Category', 'Sub_Category', 'Sub_Sub_Category', 'Product']).drop_duplicates().reset_index(drop=True)
all_prod.head()


# In[ ]:


all_prod.apply(lambda x: x.str[0]).nunique()


# In[ ]:


all_prod.nunique()


# In[ ]:


list_prod = np.sort(np.unique(all_prod.values.reshape(-1)))
print(len(list_prod))
list_prod[:5]


# In[ ]:


vectorizer = CountVectorizer(preprocessor=lambda x: re.sub('[^A-Z0-9]', ' ', x))
vectorizer.fit([*train['ProductList'], *test['ProductList']])


# In[ ]:


print(len(vectorizer.get_feature_names()))
print(vectorizer.get_feature_names()[:10])

cat_col = [c for c in vectorizer.get_feature_names() if c[0] == 'A']
sub_cat_col = [c for c in vectorizer.get_feature_names() if c[0] == 'B']
sub_sub_cat_col = [c for c in vectorizer.get_feature_names() if c[0] == 'C']
prod_col = [c for c in vectorizer.get_feature_names() if c[0] == 'D']

print(len(cat_col) + len(sub_cat_col) + len(sub_sub_cat_col) + len(prod_col))


# In[ ]:


def feature_gen(data, train=True):
    data_clean = data[['session_id', 'startTime', 'endTime']].copy()
    
    data_vocab = pd.DataFrame(vectorizer.transform(data['ProductList']).toarray(), 
                              columns=vectorizer.get_feature_names())
    
    data_clean['wday'] = data_clean['startTime'].dt.weekday
    data_clean['month'] = data_clean['startTime'].dt.month

    data_clean['start_hour'] = data_clean['startTime'].dt.hour
    
    data_clean['end_hour'] = data_clean['endTime'].dt.hour
    

    data_clean['n_view']  = data_vocab[prod_col].sum(axis=1)
    data_clean['n_cat_view'] = (data_vocab[cat_col] > 0).sum(axis=1)
    data_clean['n_sub_cat_view'] = (data_vocab[sub_cat_col] > 0).sum(axis=1)
    data_clean['n_sub_sub_cat_view'] = (data_vocab[sub_sub_cat_col] > 0).sum(axis=1)
    data_clean['n_prod_view'] = (data_vocab[prod_col] > 0).sum(axis=1)

    data_clean['total_time_spend']  = (data_clean['endTime'] - data_clean['startTime'])/pd.Timedelta('1m')
    data_clean['avg_time__view']  = data_clean['total_time_spend']/data_clean['n_view']
    data_clean['avg_time__cat']  = data_clean['total_time_spend']/data_clean['n_cat_view']
    data_clean['avg_time__sub_cat']  = data_clean['total_time_spend']/data_clean['n_sub_cat_view']
    data_clean['avg_time__sub_sub_cat']  = data_clean['total_time_spend']/data_clean['n_sub_sub_cat_view']
    data_clean['avg_time__prod']  = data_clean['total_time_spend']/data_clean['n_prod_view']

    data_clean['n_view__cat'] = data_clean['n_view']/data_clean['n_cat_view']
    data_clean['n_view__sub_cat'] = data_clean['n_view']/data_clean['n_sub_cat_view']
    data_clean['n_view__sub_sub_cat'] = data_clean['n_view']/data_clean['n_sub_sub_cat_view']
    data_clean['n_view__prod'] = data_clean['n_view']/data_clean['n_prod_view']

    data_clean['n_prod__cat'] = data_clean['n_prod_view']/data_clean['n_cat_view']
    data_clean['n_prod__sub_cat'] = data_clean['n_prod_view']/data_clean['n_sub_cat_view']
    data_clean['n_prod__sub_sub_cat'] = data_clean['n_prod_view']/data_clean['n_sub_sub_cat_view']

    data_clean['n_sub_sub_cat__cat'] = data_clean['n_sub_sub_cat_view']/data_clean['n_cat_view']
    data_clean['n_sub_sub_cat__sub_cat'] = data_clean['n_sub_sub_cat_view']/data_clean['n_sub_cat_view']

    data_clean['n_sub_cat__cat'] = data_clean['n_sub_cat_view']/data_clean['n_cat_view']

    data_clean['std_cat_view'] = data_vocab[cat_col].std(axis=1)
    data_clean['std_sub_cat_view'] = data_vocab[sub_cat_col].std(axis=1)
    data_clean['std_sub_sub_cat_view'] = data_vocab[sub_sub_cat_col].std(axis=1)
    data_clean['std_prod_view'] = data_vocab[prod_col].std(axis=1)

    data_clean['std_cat_visited'] = (data_vocab[cat_col] > 0).std(axis=1)
    data_clean['std_sub_cat_visited'] = (data_vocab[sub_cat_col] > 0).std(axis=1)
    data_clean['std_sub_sub_cat_visited'] = (data_vocab[sub_sub_cat_col] > 0).std(axis=1)
    data_clean['std_prod_visited'] = (data_vocab[prod_col] > 0).std(axis=1)

    data_clean = pd.concat([data_clean, data_vocab], axis=1)
    
    if train:
        data_clean['gender'] = data['gender']
    else:
        data_clean['gender'] = np.nan
     
    return data_clean


# In[ ]:


reduce_train = feature_gen(train, train=True)
display(reduce_train.head())


# In[ ]:


reduce_test = feature_gen(test, train=False)
reduce_test.head()


# In[ ]:


class LightgbmModel:
   def __init__(self, train, n_splits, params, categorical):
       self.feature = [col for col in train.columns if col not in ['session_id', 'startTime', 'endTime', 'gender']]
       self.categorical = categorical
       self.target = 'gender'
       self.n_splits = n_splits
       self.params = params
       cv = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
       train = train.copy()
       if self.categorical is not None and len(self.categorical) > 0:
           train[self.categorical] = train[self.categorical].astype('category')
       self.models = []
       oof_pred = np.zeros((len(train), ))
       for fold, (train_idx, val_idx) in enumerate(cv.split(train, train[self.target])):
           x_train, x_val = train[self.feature].iloc[train_idx], train[self.feature].iloc[val_idx]
           y_train, y_val = train[self.target][train_idx], train[self.target][val_idx]
           train_set, val_set = self.convert_dataset(x_train, y_train, x_val, y_val)
           model = self.train_model(train_set, val_set)
           oof_pred[val_idx] = model.predict(x_val).reshape(-1)
           self.models.append(model)
           print('Partial accuracy score of fold {} is: {}'.format(fold, accuracy_score(y_val, self.get_class(oof_pred[val_idx]))))
       
       loss_score = accuracy_score(train[self.target], self.get_class(oof_pred))
       print('Our accuracy score is: ', loss_score)

       loss_score = roc_auc_score(train[self.target], oof_pred)
       print('Our roc auc score is: ', loss_score)
   
   def convert_dataset(self, x_train, y_train, x_val, y_val):
       train_set = lgb.Dataset(x_train, y_train)
       val_set = lgb.Dataset(x_val, y_val)
       return train_set, val_set
   
   def train_model(self, train_set, val_set):
       verbosity = 100
       return lgb.train(self.params, train_set, valid_sets=[train_set, val_set], 
                        categorical_feature=self.categorical, verbose_eval=verbosity)
   
   def get_class(self, predict_proba, cutoff=0.5):
       return (predict_proba >= cutoff).astype('int')
   
   def predict(self, test_df):
       x_test = test_df[self.feature].copy()
       if self.categorical is not None and len(self.categorical) > 0:
           x_test[self.categorical] = x_test[self.categorical].astype('category')
       y_pred = np.zeros((len(test_df), ))
       for model_ in self.models:
           y_pred += model_.predict(x_test).reshape(-1) / self.n_splits
       return y_pred


# In[ ]:


categorical = ['wday']
categorical


# In[ ]:


params_lgb_1 = {'n_estimators':1500,
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': ['binary_logloss', 'auc'],
            'subsample': 0.75,
            'subsample_freq': 1,
            'learning_rate': 0.02,
            'feature_fraction': 0.8,
            'max_depth': 15,
            'lambda_l1': 1.2,  
            'lambda_l2': 1,
            'early_stopping_rounds': 100
            }

model_lgb_1 = LightgbmModel(reduce_train, n_splits=3, params=params_lgb_1, categorical=categorical)


# In[ ]:


np.all(submission['session_id'] == reduce_test['session_id'])


# In[ ]:


submission['gender'] = (pd.Series(model_lgb_1.predict(reduce_test)) >= 0.5).astype('int').map(rev_gender_map).values
submission.head()


# In[ ]:


submission['gender'].value_counts(normalize=True)


# In[ ]:


submission['gender'].value_counts()


# In[ ]:


submission.to_csv('submission.csv', index=False)

