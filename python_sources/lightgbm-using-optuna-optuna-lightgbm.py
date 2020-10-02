#!/usr/bin/env python
# coding: utf-8

# # Summary
# ## 1. [eda&easy_data_confirmation](#jump1)
# ## 2. [make models as first_model using optuna](#jump2)

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


#!/usr/bin/env python
# coding: utf-8
"""
@author: nakayama.s
"""
import os
import warnings
import gc
import time
from tqdm import tqdm
import functools


# <a name="jump1"></a>
# ## eda & easy_data_confirmation
# 

# In[ ]:


def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

def eda(data):
    # print(data)
    print("----------Top-5- Record----------")
    print(data.head(5))
    print("-----------Information-----------")
    print(data.info())
    print("-----------Data Types------------")
    print(data.dtypes)
    print("----------Missing value----------")
    print(data.isnull().sum())
    print("----------Null value-------------")
    print(data.isna().sum())
    print("----------Shape of Data----------")
    print(data.shape)
    print("----------describe---------------")
    print(data.describe())
    print("----------tail-------------------")
    print(data.tail())
    
def read_csv(path):
  # logger.debug('enter')
  df = pd.read_csv(path)
  # logger.debug('exit')
  return df

def load_train_data():
  # logger.debug('enter')
  df = read_csv(SALES_TRAIN_V2)
  # logger.debug('exit')
  return df

def load_test_data():
  # logger.debug('enter')
  df = read_csv(TEST_DATA)
  # logger.debug('exit')
  return df

def graph_insight(data):
    print(set(data.dtypes.tolist()))
    df_num = data.select_dtypes(include = ['float64', 'int64'])
    df_num.hist(figsize=(16, 16), bins=50, xlabelsize=8, ylabelsize=8);

def drop_duplicate(data, subset):
    print('Before drop shape:', data.shape)
    before = data.shape[0]
    data.drop_duplicates(subset,keep='first', inplace=True) #subset is list where you have to put all column for duplicate check
    data.reset_index(drop=True, inplace=True)
    print('After drop shape:', data.shape)
    after = data.shape[0]
    print('Total Duplicate:', before-after)

def unresanable_data(data):
    print("Min Value:",data.min())
    print("Max Value:",data.max())
    print("Average Value:",data.mean())
    print("Center Point of Data:",data.median())

SAMPLE_SUBMISSION    = '../input/sample_submission.csv'
TRAIN_DATA           = '../input/train.csv'
TEST_DATA            = '../input/test.csv'

sample          = read_csv(SAMPLE_SUBMISSION)
train           = read_csv(TRAIN_DATA)
test            = read_csv(TEST_DATA)


# In[ ]:


eda(train)


# In[ ]:


import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,accuracy_score,confusion_matrix,f1_score,classification_report


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


X_train_all = train.drop(['target','id'], axis=1)


# In[ ]:


X_train_all = X_train_all.reset_index()


# In[ ]:


Y_train_all = train['target']


# In[ ]:


X_train_all.head()


# In[ ]:


Y_train_all.head()


# <a name="jump2"></a>
# ## make Lightgbm as first_model using optuna
# * [Lightgbm](https://lightgbm.readthedocs.io/en/latest/_modules/lightgbm/sklearn.html)
# 
# ## prameter_tuning using [Oputuna](https://optuna.readthedocs.io/en/stable/)

# In[ ]:


import lightgbm as lgb


# In[ ]:


def lb_opt(X_train_all,Y_train_all,trial):
    (X_train,X_test,y_train,y_test) = train_test_split(X_train_all,Y_train_all,test_size=0.2,random_state=0)
    #paramter_tuning using optuna
    bagging_freq =  trial.suggest_int('bagging_freq',1,10),
    min_data_in_leaf =  trial.suggest_int('min_data_in_leaf',2,100),
    max_depth = trial.suggest_int('max_depth',1,20),
    learning_rate = trial.suggest_loguniform('learning_rate',0.001,0.1),
    num_leaves = trial.suggest_int('num_leaves',2,70),
    num_threads = trial.suggest_int('num_threads',1,10),
    min_sum_hessian_in_leaf = trial.suggest_int('min_sum_hessian_in_leaf',1,10),
    
    lightgbm_tuna = lgb.LGBMClassifier(
        random_state = 0,
        verbosity = 1,
        bagging_seed = 0,
        boost_from_average = 'true',
        boost = 'gbdt',
        metric = 'auc',
        bagging_freq = bagging_freq ,
        min_data_in_leaf = min_data_in_leaf,
        max_depth = max_depth,
        learning_rate = learning_rate,
        num_leaves = num_leaves,
        num_threads = num_threads,
        min_sum_hessian_in_leaf = min_sum_hessian_in_leaf
    )
    
    lightgbm_tuna.fit(X_train,y_train)
    lb_predict_test = lightgbm_tuna.predict(X_test)
    #print('accuracy_score is {} '.format(accuracy_score(y_test,lb_predict_test)))
    
    return (1 - (accuracy_score(y_test,lb_predict_test)) )


# In[ ]:


X_test_all = test.drop(['id'], axis=1)


# In[ ]:


X_test_all = X_test_all.reset_index()


# In[ ]:


optuna.logging.disable_default_handler()


# In[ ]:


X_test_all.shape


# In[ ]:


lb_best_para_list =[]
pred_list = []
pred_prob_list = []
index_list = []
for i in tqdm(range(512)): 
    x = X_train_all[X_train_all['wheezy-copper-turtle-magic']==i]
    y = Y_train_all[X_train_all['wheezy-copper-turtle-magic']==i]
    x_test = X_test_all[X_test_all['wheezy-copper-turtle-magic']==i]
    lb_study = optuna.create_study()
    lb_study.optimize(functools.partial(lb_opt,x,y),n_trials = 50)
    lb_best_para_list.append(lb_study.best_params)
    lgbm =  lgb.LGBMClassifier(**lb_study.best_params)
    lgbm.fit(x,y)
    index_list.append(x_test['index'].values)
    pred_list.append(lgbm.predict(x_test))
    pred_prob_list.append(lgbm.predict_proba(x_test)[:,1])


# In[ ]:


lb_best_para_list[:5]


# In[ ]:


pred =[]
index = []


# In[ ]:


for i in range(512):
    for j in range(len(pred_list[i])):
        pred.append(pred_list[i][j])


# In[ ]:


for i in range(512):
    for j in range(len(index_list[i])):
        index.append(index_list[i][j])


# In[ ]:


pred_prpbb =[]
for i in range(512):
    for j in range(len(pred_prob_list[i])):
        pred_prpbb.append(pred_prob_list[i][j])


# In[ ]:


len(pred_prpbb)


# In[ ]:


submission_nosort = pd.DataFrame({
    'id':index,
    'target':pred_prpbb
})


# In[ ]:


submission_nosort.head()


# In[ ]:


submission_nosort = submission_nosort.sort_values(by=['id'], ascending=True)


# In[ ]:


sample['target'] = submission_nosort['target']


# In[ ]:


sample.head()


# In[ ]:


sample.to_csv('submission.csv',index=False)

