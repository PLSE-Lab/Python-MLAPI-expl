#!/usr/bin/env python
# coding: utf-8

# # TReNDS Neuroimaging Prediction with Scikit-Learn:
# 
# 
# 
# 
#  
# #### Multiscanner normative age and assessments prediction with brain function, structure, and connectivity
# 
# 
# 
# 
# 
# ***********************************
# 
# 
# 
# 
# Human brain research is among the most complex areas of study for scientists.
# 
# The goal of this project is to predict multiple assessments and age from neuroimaging data and multimodal brain MRI features.
# 
# In particular, I look for measurable markers of behavior & health to identify relevant brain regions & their contribution to typical effects.
# 
# 
# 

# 
# ## Core Ideas
# 
# 
# 
# * Cross-Validation using Scikit-Learn KFold 
# 
# * Support Vector Machine Regressors - 3 different models
# 
# * Weighted Averaging Ensemble  
# 
# 
# *************************
# 
# 
# 
# 
# ![](https://github.com/s-gladysh/trends-neuroimaging/raw/master/trends_sa1.png)
# 
# 
# 
# 
# 
# 
# 
# ************************
# 

# 
# 
# ### Models in the Ensemble:  
# 
# 
# 
# 1. Scikit-Learn SVR
# 
# 2. Scikit-Learn NuSVR
# 
# 3.  Scikit-Learn LinearSVR
# 
# 
# 

# 
# 
# 
# ### Notebooks: 
# 
# 
# 
# 
# ****************************
# 
# 
# 
# 
# 
# https://www.kaggle.com/sgladysh/trends-neuroimaging-with-catboost-xgboost-lightgbm
# 
# 
# 
# https://www.kaggle.com/sgladysh/trends-neuroimaging-prediction-with-scikit-learn
# 
# 
# 
# 
# *************************
# 
# 
# 
# ### See also: 
# 
# 
# 
# 
# ****************************
# 
# 
# 
# 
# https://www.kaggle.com/sgladysh/sklearn-gb-bioresponse
# 
# 
# 
# 
# **************************
# 
# 
# 
# 

# 
# 
# 
# 
# ### References: 
# 
# 
# 
# *************************
# 
# 
# 
# 
# 
# *************************
# 
# 
# 
# 
# #### Scikit-Learn
# 
# * https://scikit-learn.org/
# 
# * https://github.com/scikit-learn/scikit-learn
# 
# 
# 
# *************************
# 
# 
# 
# 
# 
# #### Thanks to: 
# 
# 
# 
# https://www.kaggle.com/roshan03/svm-model
# 
# 
# 
# https://www.kaggle.com/hamditarek/trends-neuroimaging-xgbregressor
# 
# 
# 
# 
# **************************
# 
# 
# 

# ## Import libraries

# In[ ]:


import numpy as np 
import pandas as pd 

import os

import matplotlib.pyplot as plt
import matplotlib.image as mpimg    # to check images
get_ipython().run_line_magic('matplotlib', 'inline')


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


FNC_SCALE = 1/450

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
# ## Support Vector Regression  
# 
# 

# The method of Support Vector Machine (SVM) for Classification can be extended to solve regression problems. 
# 
# This method is called Support Vector Regression.
# 
# There are three different implementations of Support Vector Regression: 
# 
# 
# * SVR 
# 
# * NuSVR 
# 
# * LinearSVR
# 
# 
# 
# 
# https://scikit-learn.org/stable/modules/svm.html#regression

# In[ ]:


import sklearn


# *******************************
# 
# 
# 
# 
# 
# ## Train SVR
# 
# 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR
# 
# 

# In[ ]:


from sklearn.svm import SVR


# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 4\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = SVR(C=c)\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score')


# In[ ]:


sub_df4 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df4["Id"] = sub_df4["Id"].astype("str") + "_" +  sub_df4["variable"].astype("str")

sub_df4 = sub_df4.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df4.to_csv("submission4.csv", index=False)


# *****************************
# 
# 
# 
# ## Train NuSVR
# 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.NuSVR.html#sklearn.svm.NuSVR
# 
# 

# In[ ]:


from sklearn.svm import NuSVR


# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 4\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = NuSVR(C=c)\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score')


# ## Make predictions

# In[ ]:


sub_df5 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df5["Id"] = sub_df5["Id"].astype("str") + "_" +  sub_df5["variable"].astype("str")

sub_df5 = sub_df5.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df5.to_csv("submission5.csv", index=False)


# ***********************
# 
# 
# 
# ## Train LinearSVR
# 
# 
# 
# https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVR.html#sklearn.svm.LinearSVR
# 

# In[ ]:


from sklearn.svm import LinearSVR


# In[ ]:



get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 4\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = LinearSVR(C=c)\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score\n\n')


# ## Make predictions

# In[ ]:


sub_df6 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df6["Id"] = sub_df6["Id"].astype("str") + "_" +  sub_df6["variable"].astype("str")

sub_df6 = sub_df6.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df6.to_csv("submission6.csv", index=False)


# 
# 
# 
# ********************************
# 
# 
# 
# 
# ## Create a submission from our weighted ensemble: 
# 
# 

# In[ ]:


sub_df4['Predicted'] = 0.48 * sub_df4['Predicted'] + 0.48 * sub_df5['Predicted']  + 0.04 * sub_df6['Predicted']


# In[ ]:


sub_df4.to_csv('submission.csv', index=False, float_format='%.6f')

