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


train_y = pd.concat([y_0, y_1], axis=0)
train_y = train_y.reset_index(drop=True)
train_y.columns = ['time_to_failure']
train_y['time_to_failure'].shape


# In[ ]:


test_X = pd.read_csv("../input/lanl-master-s-features-creating-0/test_X_features_10.csv")
test_X.drop('seg_id',inplace=True,axis=1)


# In[ ]:


scaler = StandardScaler()
train_columns = train_X.columns

train_X[train_columns] = scaler.fit_transform(train_X[train_columns])
test_X[train_columns] = scaler.transform(test_X[train_columns])


# # TPOT

# In[ ]:


from tpot import TPOTRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

tpot = TPOTRegressor(generations=5, population_size=15, verbosity=2,scoring='neg_mean_absolute_error',cv=5)
tpot.fit(train_X, train_y)
print(tpot.score(train_X, train_y))
# tpot.export('tpot_boston_pipeline.py')


# In[ ]:


preds = tpot.predict(test_X)

sample_submission = pd.read_csv('../input/LANL-Earthquake-Prediction/sample_submission.csv')
sample_submission['time_to_failure'] = preds.flatten()
sample_submission.to_csv('submission.csv', index=False)

