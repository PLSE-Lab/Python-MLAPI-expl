#!/usr/bin/env python
# coding: utf-8

# ## # TReNDS Neuroimaging Prediction with CatBoost, XGBoost, lightGBM:
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
# 
# 
# 
# 
# *********************************
# 
# 
# Human brain research is among the most complex areas of study for scientists.
# 
# The goal of this project is to predict multiple assessments and age from neuroimaging data and multimodal brain MRI features.
# 
# In particular, I look for measurable markers of behavior & health to identify relevant brain regions & their contribution to typical effects.
# 
# 
# ************************
# 

# ## Core Ideas:
# 
# 
# * Cross-Validation using Scikit-Learn KFold 
# 
# * Gradient Boosting Regressor models built using different Machine Learning libraries
# 
# * Weighted Averaging Ensemble  
# 
# 
# ***************
# 
# 
# ![](https://github.com/s-gladysh/trends-neuroimaging/raw/master/trends_sa.png)
# 
# ***************
# 

# *************************
# 
# 
# 
# 
# 
# ### Models in the Weighted Averaging Ensemble:  
# 
# 
# 1. CatBoost Regressor
# 
# 2. XGBoost Regressor   
# 
# 3. lightGBM Regressor  
# 
# 4. Scikit-Learn Gradient Boosting Regressor
# 
# 
# 
# 
#  
# *******************
# 
# 
# 
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
# *********************
# 
# 
# ### See also:
# 
# 
# https://www.kaggle.com/sgladysh/covid19-catboost
# 
# 
# 
# 
# https://www.kaggle.com/sgladysh/bioresponse-catboost
# 
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
# 
# 
# **************************
# 
# 
# ##### CatBoost 
# 
# * https://catboost.ai/
# 
# * https://github.com/catboost/catboost
# 
# 
# 
# ***************************
# 
# 
# ##### XGBoost 
# 
# * https://xgboost.ai/
# 
# * https://github.com/dmlc/xgboost
# 
# 
# **************************
# 
# 
# 
# ##### lightGBM
# 
# * https://www.microsoft.com/en-us/research/project/lightgbm/
# 
# * https://github.com/microsoft/LightGBM
# 
# 
# *************************
# 
# 
# ##### Scikit-Learn 
# 
# 
# * https://scikit-learn.org/
# 
# 
# * https://github.com/scikit-learn/scikit-learn
# 
# 
# 
# ***************************
# 
# 
# 
# 
# 
# ### Thanks to: 
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
# **************************************
# 
# 
# 
# # 1. CatBoost 
# 
# 
# 
# 
# ************************************
# 
# 
# 
# *******************
# 
# * CatBoost is a machine learning algorithm that uses gradient boosting on decision trees. 
# 
# * It is available as an open source library.
# 
# 
# ******************************
# 
# 
# ![](https://avatars.mds.yandex.net/get-bunker/56833/dba868860690e7fe8b68223bb3b749ed8a36fbce/orig)
# 
# 
# *****************************
# 
# 
# 
# ### See some videos about CatBoost:
# 
# 
# 
# <div class="alert alert-block alert-info">
# <img src='https://i.imgur.com/H6AnLaj.png' width='90' align='left'></img>
# <p><a href='https://www.youtube.com/watch?v=s8Q_orF4tcI'>Catboost: Open-source Gradient Boosting Library!</a></p>
# <p>Yandex Research</p>
# </div>
# 
# 
# ******************************
# 
# 
# 
# 
# 
# <div class="alert alert-block alert-info">
# <img src='https://i.imgur.com/H6AnLaj.png' width='90' align='left'></img>
# <p><a href='https://www.youtube.com/watch?v=dvZLk7LxGzc'>CatBoost VS XGboost - It's Modeling Cat Fight Time! Welcome to 5 Minutes for Data Science!</a></p>
# <p>Manuel Amunategui</p>
# </div>
# 
# 
# ******************************
# 
# 
# 
# ### References: 
# 
# 
# https://catboost.ai/
# 
# https://github.com/catboost/catboost
# 
# 
# 
# Anna Veronika Dorogush, Andrey Gulin, Gleb Gusev, Nikita Kazeev, Liudmila Ostroumova Prokhorenkova, Aleksandr Vorobev 
# "Fighting biases with dynamic boosting". arXiv:1706.09516, 2017.
# https://arxiv.org/abs/1706.09516
# 
# 
# Anna Veronika Dorogush, Vasily Ershov, Andrey Gulin 
# "CatBoost: gradient boosting with categorical features support". Workshop on ML Systems at NIPS 2017.
# http://learningsys.org/nips17/assets/papers/paper_11.pdf
# 
# 
# 
# **************************************
# 
# 
# 

# 
# 
# 
# 
# ## Import CatBoost

# In[ ]:


import catboost
from catboost import CatBoostRegressor


# ## Train CatBoost Regressor model
# 
# 
# 
# ***************************
# 
# 
# 
# 

# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 4\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = CatBoostRegressor(silent=True)\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score\n    print(target, np.round(score, 4))\n    print()\n    \nprint("Overal score:", np.round(overal_score, 4))')


# ## Make predictions

# In[ ]:


sub_df1 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df1["Id"] = sub_df1["Id"].astype("str") + "_" +  sub_df1["variable"].astype("str")

sub_df1 = sub_df1.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df1.to_csv("submission1.csv", index=False)


# 
# *************************
# 
# 
# 
# # 2. XGBoost
# 
# 
# 
# 
# 
# ************************
# 
# 
# 
# 
# ************************
# 
# 
# 
# 
# ## Import XGBoost

# In[ ]:


import xgboost
from xgboost import XGBRegressor


# ## Train XGBoost Regressor model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 4\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = XGBRegressor()\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score\n    print(target, np.round(score, 4))\n    print()\n    \nprint("Overal score:", np.round(overal_score, 4))')


# ## Make predictions

# In[ ]:


sub_df2 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df2["Id"] = sub_df2["Id"].astype("str") + "_" +  sub_df2["variable"].astype("str")

sub_df2 = sub_df2.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df2.to_csv("submission2.csv", index=False)


# 
# 
# ********************************
# 
# 
# # 3. lightGBM
# 
# 
# 
# 
# *******************************
# 
# 
# 
# *********************************
# 
# 
# 
# ## Import lightGBM

# In[ ]:


import lightgbm
from lightgbm import LGBMRegressor


# ## Train lightGBM Regressor model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 4\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = LGBMRegressor()\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score\n    print(target, np.round(score, 4))\n    print()\n    \nprint("Overal score:", np.round(overal_score, 4))')


# ## Make predictions

# In[ ]:


sub_df3 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df3["Id"] = sub_df3["Id"].astype("str") + "_" +  sub_df3["variable"].astype("str")

sub_df3 = sub_df3.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df3.to_csv("submission3.csv", index=False)


# # 4. Scikit-Learn Gradient Boosting Regressor
# 
# 
# 
# 
# 
# 
# *****************************
# 
# 
# 
# 
# ******************************
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# In[ ]:


from sklearn.ensemble import GradientBoostingRegressor


# 
# ## Train Scikit-Learn Gradient Boosting Regressor

# In[ ]:


get_ipython().run_cell_magic('time', '', 'NUM_FOLDS = 4\nkf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=0)\n\n\nfeatures = loading_features + fnc_features\n\noveral_score = 0\nfor target, c, w in [("age", 100, 0.3), ("domain1_var1", 10, 0.175), ("domain1_var2", 10, 0.175), ("domain2_var1", 10, 0.175), ("domain2_var2", 10, 0.175)]:    \n    y_oof = np.zeros(df.shape[0])\n    y_test = np.zeros((test_df.shape[0], NUM_FOLDS))\n    \n    for f, (train_ind, val_ind) in enumerate(kf.split(df, df)):\n        train_df, val_df = df.iloc[train_ind], df.iloc[val_ind]\n        train_df = train_df[train_df[target].notnull()]\n\n        model = GradientBoostingRegressor()\n        model.fit(train_df[features], train_df[target])\n\n        y_oof[val_ind] = model.predict(val_df[features])\n        y_test[:, f] = model.predict(test_df[features])\n        \n    df["pred_{}".format(target)] = y_oof\n    test_df[target] = y_test.mean(axis=1)\n    score = metric(df[df[target].notnull()][target].values, df[df[target].notnull()]["pred_{}".format(target)].values)\n    overal_score += w*score')


# In[ ]:


sub_df5 = pd.melt(test_df[["Id", "age", "domain1_var1", "domain1_var2", "domain2_var1", "domain2_var2"]], id_vars=["Id"], value_name="Predicted")
sub_df5["Id"] = sub_df5["Id"].astype("str") + "_" +  sub_df5["variable"].astype("str")

sub_df5 = sub_df5.drop("variable", axis=1).sort_values("Id")


# In[ ]:


sub_df5.to_csv("submission5.csv", index=False)


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


sub_df1['Predicted'] = 0.86 * sub_df1['Predicted'] + 0.02 * sub_df2['Predicted'] + 0.1 * sub_df3['Predicted'] + 0.02 * sub_df5['Predicted'] 


# In[ ]:


sub_df1.to_csv('submission.csv', index=False, float_format='%.6f')


# ### Thank you!
