#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import xgboost as xgb

import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

from sklearn.model_selection import train_test_split
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


train_df = pd.read_csv("../input/training_set.csv", dtype={"object_id": "object"})
train_meta_df = pd.read_csv("../input/training_set_metadata.csv", dtype={"object_id": "object"})

sample_submission_df = pd.read_csv("../input/sample_submission.csv", dtype={"object_id": "object"})


# In[ ]:


train_df.head()


# In[ ]:


train_meta_df.head()


# In[ ]:


print("--------------Shape------------------")
print(" Train: {}\n Train meta: {}".format(train_df.shape, train_meta_df.shape))


# In[ ]:


ID_col = "object_id"
target_col = "target"
ts_index_col = "mjd"
ts_cols = ["passband", "flux", "flux_err", "detected"]
static_cols = ["ra", "decl", "gal_l", "gal_b", "ddf", "hostgal_specz", 
               "hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv"]


# In[ ]:


def preprocess(df):
    df[target_col] = df[target_col].astype("category").cat.codes
    return df


# In[ ]:


train_meta_df = preprocess(train_meta_df)


# In[ ]:


# xgb
params = {'eta': 0.02, 'max_depth': 4, 'subsample': 0.9, 'colsample_bytree': 0.9, 
          'objective': 'multi:softprob', 'eval_metric': 'mlogloss', 'silent': True, 
          'num_class':14}

X = train_meta_df.drop([ID_col, target_col], axis=1)
features = X.columns
y = train_meta_df[target_col].values
nrounds = 2000

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, stratify=y,random_state=123) 

d_train = xgb.DMatrix(X_train, y_train) 
d_valid = xgb.DMatrix(X_valid, y_valid) 
watchlist = [(d_train, 'train'), (d_valid, 'valid')]
xgb_model = xgb.train(params, d_train, nrounds, watchlist, early_stopping_rounds=100, 
                      maximize=False, verbose_eval=100)


# In[ ]:


test_meta_df = pd.read_csv("../input/test_set_metadata.csv")


# In[ ]:


## Setting up submission

sub = xgb_model.predict(xgb.DMatrix(test_meta_df[features]), ntree_limit=xgb_model.best_ntree_limit+50)
sub = pd.DataFrame(sub)
sub['14'] = 0


# In[ ]:


sub["max_prob"] = sub.values.argmax(axis=1)
sub["max_prob"] = sub["max_prob"] < 0.5
sub['14'] = sub["max_prob"].astype('int')


# In[ ]:


sub2 = sub.div(sub.sum(axis=1), axis=0)
sub2.drop(["max_prob"], axis=1, inplace=True)


# In[ ]:


sub2.rename(columns= dict(zip(sub2.columns, sample_submission_df.iloc[:,1:].columns)), inplace=True)
sub2[ID_col] = test_meta_df[ID_col]
sub2 = sub2[sample_submission_df.columns]


# In[ ]:


sub2.to_csv("submission2.csv", index=False)


# In[ ]:


sub2.head()


# In[ ]:




