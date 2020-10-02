#!/usr/bin/env python
# coding: utf-8

# In this notebook, I made a simple lightgbm code. 
# This code takes **less than 15min** to finish all process. 

# In[ ]:


import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import random
random.seed(1)

import os
import sys
import gc
print(os.listdir("../input"))

import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

import lightgbm as lgb
from sklearn.feature_extraction.text import CountVectorizer
import re
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn import metrics


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 1min in Kernel\nmeta_train = pd.read_csv('../input/metadata_train.csv')\nsubset_train = pq.read_pandas('../input/train.parquet', columns=[str(i) for i in range(8712)]).to_pandas()")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# 20s in Kernel\ntrain_length = 8712 #max 8712\npositive_length = len(meta_train[meta_train[\'target\']==1])\ntrain_df = pd.DataFrame()\nrow_index = 0\nfor i in range(train_length):\n    # downsampling\n    if meta_train.loc[i,\'target\'] == 1 or random.random() < positive_length / train_length:\n        subset_train_row = subset_train[str(i)]\n        train_df.loc[row_index, \'signal_min\'] = subset_train_row.min()\n        train_df.loc[row_index, \'signal_max\'] = subset_train_row.max()\n        train_df.loc[row_index, \'signal_mean\'] = subset_train_row.mean()\n        # *** Add your feature here ***\n        train_df.loc[row_index, \'signal_id\'] = i\n        row_index += 1\nprint("positive length: " + str(positive_length))\n# positive length 525\nprint("train length: " + str(len(train_df)))\n# train length 1038  example\n# This will be about 1050')


# In[ ]:


train_df = pd.merge(train_df, meta_train, on='signal_id')
train_df.to_csv("train.csv", index=False)
train_df.head()


# In[ ]:


# From https://www.kaggle.com/delayedkarma/lightgbm-cv-matthews-correlation-coeff
# Thank you!!
# If you have preprocessed data, input here and delete process method.
# x_train = pd.read_csv('../input/***/train.csv')
x_train = train_df
target = x_train['target']
input_target = x_train['target']
x_train.drop('target', axis=1, inplace=True)
x_train.drop('signal_id', axis=1, inplace=True)
features = x_train.columns
param = {'num_leaves': 80,
         'min_data_in_leaf': 60, 
         'objective':'binary',
         'max_depth': -1,
         'learning_rate': 0.1,
         "boosting": "gbdt",
         "feature_fraction": 0.8,
         "bagging_freq": 1,
         "bagging_fraction": 0.8 ,
         "bagging_seed": 11,
         "metric": 'auc',
         "lambda_l1": 0.1,
         "random_state": 133,
         "verbosity": -1}
max_iter=5


# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(x_train))
feature_importance_df = pd.DataFrame()
score = [0 for _ in range(folds.n_splits)]
for fold_, (trn_idx, val_idx) in enumerate(folds.split(x_train.values, target.values)):
    print("Fold No.{}".format(fold_+1))
    trn_data = lgb.Dataset(x_train.iloc[trn_idx][features],
                           label=target.iloc[trn_idx])
    val_data = lgb.Dataset(x_train.iloc[val_idx][features],
                           label=target.iloc[val_idx])
    num_round = 10000
    clf = lgb.train(param,
                    trn_data,
                    num_round,
                    valid_sets = [trn_data, val_data],
                    verbose_eval=100,
                    early_stopping_rounds = 100)
    
    oof[val_idx] = clf.predict(x_train.iloc[val_idx][features], num_iteration=clf.best_iteration)
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance(importance_type='gain')
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    score[fold_] = metrics.roc_auc_score(target.iloc[val_idx], oof[val_idx])
    if fold_ == max_iter - 1: break
if (folds.n_splits == max_iter):
    print("CV score: {:<8.5f}".format(metrics.roc_auc_score(target, oof)))
else:
     print("CV score: {:<8.5f}".format(sum(score) / max_iter))


# In[ ]:


cols = (feature_importance_df[["feature", "importance"]]
        .groupby("feature")
        .mean()
        .sort_values(by="importance", ascending=False)[:1000].index)
best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(10,10))
sns.barplot(x="importance",
            y="feature",
            data=best_features.sort_values(by="importance",ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()


# Check your CV carefully because
# > You may submit a maximum of 2 entries per day.

# In[ ]:


gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# 25ms in Kernel\nmeta_test = pd.read_csv('../input/metadata_test.csv')")


# In[ ]:


get_ipython().run_cell_magic('time', '', '# About 10min in Kernel\ntest_df = pd.DataFrame()\nrow_index = 0\nfor i in range(10):\n    subset_test = pq.read_pandas(\'../input/test.parquet\', columns=[str(i*2000 + j + 8712) for j in range(2000)]).to_pandas()\n    for j in range(2000):\n        subset_test_row = subset_test[str(i*2000 + j + 8712)]\n        test_df.loc[row_index, \'signal_min\'] = subset_test_row.min()\n        test_df.loc[row_index, \'signal_max\'] = subset_test_row.max()\n        test_df.loc[row_index, \'signal_mean\'] = subset_test_row.mean()\n        # *** Add your feature here ***\n        test_df.loc[row_index, \'signal_id\'] = i*2000 + j + 8712\n        row_index += 1\nsubset_test = pq.read_pandas(\'../input/test.parquet\', columns=[str(i + 28712) for i in range(337)]).to_pandas()\nfor i in tqdm(range(337)):\n    subset_test_row = subset_test[str(i + 28712)]\n    test_df.loc[row_index, \'signal_min\'] = subset_test_row.min()\n    test_df.loc[row_index, \'signal_max\'] = subset_test_row.max()\n    test_df.loc[row_index, \'signal_mean\'] = subset_test_row.mean()\n    # *** Add your feature here ***\n    test_df.loc[row_index, \'signal_id\'] = i + 28712\n    row_index += 1\ntest_df = pd.merge(test_df, meta_test, on=\'signal_id\')\ntest_df.to_csv("test.csv", index=False)\ntest_df.head()')


# In[ ]:


# If you have preprocessed data, input here and delete process method.
# x_test = pd.read_csv('../input/***/test.csv')
x_test = test_df
x_filename = x_test['signal_id']
x_test = x_test.drop('signal_id', axis=1)

predictions = clf.predict(x_test, num_iteration=clf.best_iteration)

sub_df = pd.DataFrame({"signal_id":x_filename.values})
sub_df["target"] = pd.Series(predictions).round()
sub_df['signal_id'] = sub_df['signal_id'].astype(np.int64)
sub_df['target'] = sub_df['target'].astype(np.int64)
sub_df.to_csv("submission.csv", index=False)
sub_df


# In[ ]:


positive = len(sub_df[sub_df["target"] == 1])
print(positive)
print(str(positive/len(sub_df)*100) + "%")


# 
