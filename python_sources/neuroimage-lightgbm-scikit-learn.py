#!/usr/bin/env python
# coding: utf-8

# ## Import libraries

# In[ ]:


import numpy as np 
import pandas as pd 

import os


# ## Import data

# In[ ]:


loading = pd.read_csv('/kaggle/input/trends-assessment-prediction/loading.csv')
features = pd.read_csv('/kaggle/input/trends-assessment-prediction/train_scores.csv')
submission = pd.read_csv('/kaggle/input/trends-assessment-prediction/sample_submission.csv')
fnc = pd.read_csv("/kaggle/input/trends-assessment-prediction/fnc.csv")
score = pd.read_csv("/kaggle/input/trends-assessment-prediction/train_scores.csv")


# ## Transform data

# In[ ]:


fnc_features, loading_features = list(fnc.columns[1:]), list(loading.columns[1:])


# In[ ]:


df = fnc.merge(loading, on="Id")
score["is_train"] = True
df = df.merge(score, on="Id", how="left")
test_df = df[df["is_train"] != True].copy()
df = df[df["is_train"] == True].copy()


# In[ ]:


FNC_SCALE = 1/400

df[fnc_features] *= FNC_SCALE
test_df[fnc_features] *= FNC_SCALE


# In[ ]:


features = loading_features + fnc_features


# In[ ]:


from sklearn.model_selection import KFold

def metric(y_true, y_pred):
    return np.mean(np.sum(np.abs(y_true - y_pred), axis=0)/np.sum(y_true, axis=0))


# 
# 
# *************************************
# 
# 
# # Scikit-Learn
# 
# 
# ************************************
# 
# 
# ## Import Scikit-Learn, create Support Vector Machine model 

# In[ ]:


import sklearn
from sklearn.svm import SVR


# ## Train SVR

# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 10\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 50, 0.3), ("domain1_var1", 5, 0.175), ("domain1_var2", 5, 0.175), ("domain2_var1", 5, 0.175), ("domain2_var2", 5, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = SVR(C=c, cache_size=3000.0)\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score\n    print(target, np.round(score, 4))\n    print()\n    \nprint("Overal score:", np.round(overal_score, 4))')


# In[ ]:


sub_df4 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df4["Id"] = sub_df4["Id"].astype("str") + "_" +  sub_df4["variable"].astype("str")

sub_df4 = sub_df4.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df4.to_csv("submission50.csv", index=False, float_format='%.6f')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 10\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 80, 0.3), ("domain1_var1", 8, 0.175), ("domain1_var2", 8, 0.175), ("domain2_var1", 8, 0.175), ("domain2_var2", 8, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = SVR(C=c, cache_size=3000.0)\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score\n    print(target, np.round(score, 4))\n    print()\n    \nprint("Overal score:", np.round(overal_score, 4))')


# In[ ]:


sub_df4 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df4["Id"] = sub_df4["Id"].astype("str") + "_" +  sub_df4["variable"].astype("str")

sub_df4 = sub_df4.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df4.to_csv("submission80.csv", index=False, float_format='%.6f')

