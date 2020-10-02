#!/usr/bin/env python
# coding: utf-8

# * Below you may notice the effect of using MAE/RMSE as objective functions of the resulting MAE for the competition
# * snippet from https://www.kaggle.com/tocha4/lanl-master-s-approach

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import scipy as sc
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
from tqdm import tqdm_notebook
import datetime
import time
import random
from joblib import Parallel, delayed


import lightgbm as lgb
from tensorflow import keras
from gplearn.genetic import SymbolicRegressor
from catboost import Pool, CatBoostRegressor

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.feature_selection import RFECV, SelectFromModel

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor


# In[ ]:


train_X_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_X_features_865.csv")
train_X_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_X_features_865.csv")
y_0 = pd.read_csv("../input/lanl-master-s-features-creating-0/train_y.csv", index_col=False,  header=None)
y_1 = pd.read_csv("../input/lanl-master-s-features-creating-1/train_y.csv", index_col=False,  header=None)


# In[ ]:


train_X = pd.concat([train_X_0, train_X_1], axis=0)
train_X = train_X.reset_index(drop=True)
print(train_X.shape)
train_X.head()


# In[ ]:


y = pd.concat([y_0, y_1], axis=0)
y = y.reset_index(drop=True)
y[0].shape


# In[ ]:


train_y = pd.Series(y[0].values)


# In[ ]:


test_X = pd.read_csv("../input/lanl-master-s-features-creating-0/test_X_features_10.csv")
# del X["seg_id"], test_X["seg_id"]


# In[ ]:


scaler = StandardScaler()
train_columns = train_X.columns

train_X[train_columns] = scaler.fit_transform(train_X[train_columns])
test_X[train_columns] = scaler.transform(test_X[train_columns])


# # CatBoost

# In[ ]:


train_columns = train_X.columns
n_fold = 5


# MAE

# In[ ]:


get_ipython().run_cell_magic('time', '', 'folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)\n\noof = np.zeros(len(train_X))\ntrain_score = []\nfold_idxs = []\n# if PREDICTION: \npredictions = np.zeros(len(test_X))\n\nfeature_importance_df = pd.DataFrame()\n#run model\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y.values)):\n    strLog = "fold {}".format(fold_)\n    print(strLog)\n    fold_idxs.append(val_idx)\n\n    X_tr, X_val = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]\n    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]\n\n    model = CatBoostRegressor(n_estimators=30000, verbose=-1, objective="MAE", loss_function="MAE", boosting_type="Ordered", task_type="GPU")\n    model.fit(X_tr, \n              y_tr, \n              eval_set=[(X_val, y_val)], \n#               eval_metric=\'mae\',\n              verbose=2500, \n              early_stopping_rounds=600)\n    oof[val_idx] = model.predict(X_val)\n\n    #feature importance\n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["Feature"] = train_columns\n    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    #predictions\n#     if PREDICTION:\n\n    predictions += model.predict(test_X[train_columns]) / folds.n_splits\n    train_score.append(model.best_score_[\'learn\']["MAE"])\n\ncv_score = mean_absolute_error(train_y, oof)\nprint(f"After {n_fold} test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")\n\ntoday = str(datetime.date.today())\nsubmission = pd.read_csv(\'../input/LANL-Earthquake-Prediction/sample_submission.csv\')\n\nsubmission["time_to_failure"] = predictions\nsubmission.to_csv(f\'CatBoost_MAE_{today}_test_{cv_score:.3f}_train_{np.mean(train_score):.3f}.csv\', index=False)\nsubmission.head()')


# RMSE

# In[ ]:


get_ipython().run_cell_magic('time', '', 'folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)\n\noof = np.zeros(len(train_X))\ntrain_score = []\nfold_idxs = []\n# if PREDICTION: \npredictions = np.zeros(len(test_X))\n\nfeature_importance_df = pd.DataFrame()\n#run model\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y.values)):\n    strLog = "fold {}".format(fold_)\n    print(strLog)\n    fold_idxs.append(val_idx)\n\n    X_tr, X_val = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]\n    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]\n\n    model = CatBoostRegressor(n_estimators=30000, verbose=-1, objective="RMSE", loss_function="RMSE", boosting_type="Ordered", task_type="GPU")\n    model.fit(X_tr, \n              y_tr, \n              eval_set=[(X_val, y_val)], \n#               eval_metric=\'mae\',\n              verbose=2500, \n              early_stopping_rounds=600)\n    oof[val_idx] = model.predict(X_val)\n\n    #feature importance\n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["Feature"] = train_columns\n    fold_importance_df["importance"] = model.feature_importances_[:len(train_columns)]\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)\n    #predictions\n#     if PREDICTION:\n\n    predictions += model.predict(test_X[train_columns]) / folds.n_splits\n    train_score.append(model.best_score_[\'learn\']["RMSE"])\n\ncv_score = mean_absolute_error(train_y, oof)\nprint(f"After {n_fold} test_CV = {cv_score:.3f} | train_CV = {np.mean(train_score):.3f} | {cv_score-np.mean(train_score):.3f}", end=" ")\n\ntoday = str(datetime.date.today())\nsubmission = pd.read_csv(\'../input/LANL-Earthquake-Prediction/sample_submission.csv\')\n\nsubmission["time_to_failure"] = predictions\nsubmission.to_csv(f\'CatBoost_RMSE_{today}_test_{cv_score:.3f}_train_{np.mean(train_score):.3f}.csv\', index=False)\nsubmission.head()')

