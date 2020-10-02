#!/usr/bin/env python
# coding: utf-8

# # Overview
# I investigated the difference between train and test by Adversarial Validation.  
# We can trust CV and public lb. Leaks are excluded...

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train = pd.read_csv('/kaggle/input/killer-shrimp-invasion/train.csv')
test = pd.read_csv('/kaggle/input/killer-shrimp-invasion/test.csv')
submission = pd.read_csv('/kaggle/input/killer-shrimp-invasion/temperature_submission.csv')
print('train: ', train.shape)
print('test: ', test.shape)
print('submission: ', submission.shape)


# In[ ]:


train.query('Presence==0').sample(5).head()


# In[ ]:


train.query('Presence==1').sample(5).head()


# In[ ]:


train['is_test'] = 0
test['is_test'] = 1
data = pd.concat([train, test], sort=False)
data.reset_index(drop=True, inplace=True)

# excluded pointid
feats = ['Salinity_today', 'Temperature_today', 'Substrate', 'Depth', 'Exposure'] 

data[feats].head()


# In[ ]:


def post_preparation_importance(feats, model, n_fold):
    imp_df = pd.DataFrame()
    imp_df["feature"] = feats
    imp_df["importance"] = model.feature_importance(importance_type='gain', iteration=model.best_iteration)
    imp_df["fold"] = n_fold + 1
    return imp_df

def post_display_importances(feature_importance_df_):
    cols = feature_importance_df_[["feature", "importance"]]        .groupby("feature").mean().sort_values(by="importance", ascending=False)[:40].index
        
    best_features = feature_importance_df_.loc[feature_importance_df_.feature.isin(cols)]

    plt.figure(figsize=(8, 10))
    sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
    plt.title('LightGBM Features (avg over folds)')
    plt.show()


# In[ ]:


lgb_params = {
    'boost': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'verbosity': 1,
}


# In[ ]:


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=42)
feature_importance_df = pd.DataFrame()

for i_fold, (train_idx, valid_idx) in enumerate(kfold.split(data)):
    print(f"--------fold {i_fold}-------")
    
    ## train data
    x_tr = data.loc[train_idx, feats]
    y_tr = data.loc[train_idx, 'is_test']

    ## valid data
    x_va = data.loc[valid_idx, feats]
    y_va = data.loc[valid_idx, 'is_test']
    
    lgb_train = lgb.Dataset(x_tr, label=y_tr)
    lgb_test = lgb.Dataset(x_va, label=y_va)

    model = lgb.train(
        lgb_params,
        lgb_train,
        valid_sets=[lgb_train, lgb_test],
        valid_names=['train', 'test'],
        num_boost_round=1000,
        early_stopping_rounds=100,
        verbose_eval=100
    )

    valid_preds = model.predict(x_va, num_iteration=model.best_iteration)
    valid_metric = roc_auc_score(y_va, valid_preds)
    
    fold_importance_df = post_preparation_importance(feats, model, n_split)
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('Valid Metric: {:.5f}'.format(valid_metric))


# In[ ]:


post_display_importances(feature_importance_df)


# In[ ]:


data.query('is_test == 0')['Depth'].describe()


# In[ ]:


data.query('is_test == 1')['Depth'].describe()


# In[ ]:


data.query('is_test == 1')['Exposure'].describe()


# In[ ]:


data.query('is_test == 0')['Exposure'].describe()

