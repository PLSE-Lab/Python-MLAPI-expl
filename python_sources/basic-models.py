#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Imports
import os
import numpy as np
import keras
import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score


# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


train_data = data.iloc[:,2:203]


# In[ ]:


train_data.shape


# In[ ]:


y = data.iloc[:,1]


# In[ ]:


y.shape


# In[ ]:


y.head()


# In[ ]:


train_data.head()


# In[ ]:


#Checking Incomplete Data Rows
sample_incomplete_rows = data[train_data.isnull().any(axis=1)].head()
sample_incomplete_rows


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(train_data, y, test_size=0.2, random_state=142)


# In[ ]:


X_train.head()


# In[ ]:


y_train.head()


# In[ ]:


#Basic Models
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# Gaussian NB

# In[ ]:


gnb = GaussianNB()
gnb.fit(X_train, y_train)


# In[ ]:


y_preds_gnb = gnb.predict(X_test)
print(accuracy_score(y_test, y_preds_gnb))


# In[ ]:


test_data = pd.read_csv('../input/test.csv')


# In[ ]:


test_data.shape


# In[ ]:


test_data.head()


# In[ ]:


X_test_data = test_data.iloc[:,1:202]


# In[ ]:


X_test_data.shape


# In[ ]:


y_preds_test_data_gnb = gnb.predict(X_test_data)


# In[ ]:


my_submission_gnb = pd.DataFrame({'ID_code': test_data.ID_code, 'target': y_preds_test_data_gnb})
my_submission_gnb.to_csv('submission_gnb.csv', index=False)


# CatBoost Classifier

# In[ ]:


from catboost import CatBoostClassifier


# In[ ]:


cat = CatBoostClassifier(iterations=3000, learning_rate=0.03, objective="Logloss", eval_metric='AUC')
cat.fit(X_train, y_train)


# In[ ]:


y_preds_cat = cat.predict(X_test)
print(accuracy_score(y_test, y_preds_cat))


# In[ ]:


y_preds_test_data_cat = cat.predict(X_test_data)


# In[ ]:


my_submission_cat = pd.DataFrame({'ID_code': test_data.ID_code, 'target': y_preds_test_data_cat})
my_submission_cat.to_csv('submission_cat.csv', index=False)


# LightGBM

# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
import time


# In[ ]:


def augment(x,y,t=2):
    xs,xn = [],[]
    for i in range(t):
        mask = y>0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xs.append(x1)

    for i in range(t//2):
        mask = y==0
        x1 = x[mask].copy()
        ids = np.arange(x1.shape[0])
        for c in range(x1.shape[1]):
            np.random.shuffle(ids)
            x1[:,c] = x1[ids][:,c]
        xn.append(x1)

    xs = np.vstack(xs)
    xn = np.vstack(xn)
    ys = np.ones(xs.shape[0])
    yn = np.zeros(xn.shape[0])
    x = np.vstack([x,xs,xn])
    y = np.concatenate([y,ys,yn])
    return x,y


# In[ ]:


n_fold = 5
folds = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=42)


# In[ ]:


lgbm_params = {'bagging_freq': 5,
               'bagging_fraction': 0.335,
               'boost_from_average':'false',
               'boost': 'gbdt',
               'feature_fraction': 0.041,
               'learning_rate': 0.0083,
               'max_depth': -1,
               'metric':'auc',
               'min_data_in_leaf': 80,
               'min_sum_hessian_in_leaf': 10.0,
               'num_leaves': 13,
               'num_threads': 8,
               'tree_learner': 'serial',
               'objective': 'binary', 
               'verbosity': -1
}


# In[ ]:


prediction_lgb_new = np.zeros(len(X_test_data))

for fold_n, (train_index, valid_index) in enumerate(folds.split(train_data,y)):
    print('Fold', fold_n)
    X_training, X_validation = train_data.iloc[train_index], train_data.iloc[valid_index]
    y_training, y_validation = y.iloc[train_index], y.iloc[valid_index]
    
    X_training, y_training = augment(X_training.values, y_training.values)
    X_training = pd.DataFrame(X_training)
    
    training_data = lgb.Dataset(X_training, label=y_training)
    validation_data = lgb.Dataset(X_validation, label=y_validation)
        
    model = lgb.train(lgbm_params,training_data,num_boost_round=1000000,
                    valid_sets = [training_data, validation_data],verbose_eval=1000,early_stopping_rounds = 3000)
            
    #y_pred_valid = model.predict(X_valid)
    prediction_lgb_new += model.predict(X_test_data, num_iteration=model.best_iteration)/folds.n_splits


# In[ ]:


prediction_lgb_new


# In[ ]:


my_submission_lgb_new = pd.DataFrame({'ID_code': test_data.ID_code, 'target': prediction_lgb_new})
my_submission_lgb_new.to_csv('submission_lgb_new.csv', index=False)


# In[ ]:


my_submission_cat_lgb_gnb = pd.DataFrame({'ID_code': test_data.ID_code,
                                          'target': (y_preds_test_data_cat + y_preds_test_data_gnb)/2})
my_submission_cat_lgb_gnb.to_csv('my_submission_cat_gnb.csv', index=False)


# In[ ]:




