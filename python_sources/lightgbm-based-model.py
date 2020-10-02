#!/usr/bin/env python
# coding: utf-8

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


train_X_0 = pd.read_csv("../input/feature1/train_X_features_865.csv")
train_X_1 = pd.read_csv("../input/feature2/train_X_features_865_1.csv")
y_0 = pd.read_csv("../input/feature1/train_y.csv", index_col=False,  header=None)
y_1 = pd.read_csv("../input/feature2/train_y1.csv", index_col=False,  header=None)


# In[ ]:





# In[ ]:


train_X = pd.concat([train_X_0, train_X_1], axis=0)
train_X = train_X.reset_index(drop=True)
print(train_X.shape)
print(train_X_0.shape)
train_X.head()


# In[ ]:


y = pd.concat([y_0, y_1], axis=0)
y = y.reset_index(drop=True)
y[0].shape


# In[ ]:


train_y = pd.Series(y[0].values)


# In[ ]:


test_X = pd.read_csv("../input/feature2/test_X_features_10.csv")
# del X["seg_id"], test_X["seg_id"]
test_X.shape


# In[ ]:


scaler = StandardScaler()
train_columns = train_X.columns

train_X[train_columns] = scaler.fit_transform(train_X[train_columns])
test_X[train_columns] = scaler.transform(test_X[train_columns])


# # lightgbm

# In[ ]:


train_columns = train_X.columns
n_fold = 5


# In[ ]:


get_ipython().run_cell_magic('time', '', 'folds = KFold(n_splits=n_fold, shuffle=True, random_state=42)\n\noof = np.zeros(len(train_X))\ntrain_score = []\nfold_idxs = []\n# if PREDICTION: \npredictions = np.zeros(len(test_X))\n\nfeature_importance_df = pd.DataFrame()\n#run model\nfor fold_, (trn_idx, val_idx) in enumerate(folds.split(train_X,train_y.values)):\n    strLog = "fold {}".format(fold_)\n    print(strLog)\n    fold_idxs.append(val_idx)\n\n    X_tr, X_val = train_X[train_columns].iloc[trn_idx], train_X[train_columns].iloc[val_idx]\n    y_tr, y_val = train_y.iloc[trn_idx], train_y.iloc[val_idx]\n\n    model = lgb.LGBMRegressor(n_estimators = 30000, n_jobs = -1,random_state=40)\n    model.fit(X_tr, \n              y_tr, \n              eval_set=[(X_val, y_val)], \n              eval_metric=\'mae\',\n              verbose=True,\n              early_stopping_rounds=200)\n    oof[val_idx] = model.predict(X_val,num_iteration=model.best_iteration_)\n    predictions += model.predict(test_X[train_columns],num_iteration=model.best_iteration_) / folds.n_splits\n\ncv_score = mean_absolute_error(train_y, oof)\nprint(cv_score)')


# In[ ]:


submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
cat_saved=pd.DataFrame(oof,columns=['oof'])
cat_saved.to_csv('lgb_oof.csv',index=False)
submission["time_to_failure"] = predictions
submission.to_csv(f'lgb_submission_{cv_score:.3f}.csv', index=False)
submission.head()

