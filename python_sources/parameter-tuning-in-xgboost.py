#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
import os
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
rcParams['figure.figsize'] = 12, 4


# In[ ]:


import xgboost as xgb
from xgboost import plot_importance
from xgboost.sklearn import XGBClassifier
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# In[ ]:


X_train = pd.read_csv('../input/X_train.csv')
y_train = pd.read_csv('../input/y_train.csv')


# **Step1: Determine the number of estimators for learning rate and tree_based parameter tuning.**

# In[ ]:


param_test1 = { 'n_estimators': range(10,300,10)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,
                                                  min_child_weight=1,gamma=0, subsample=0.8,             
                                                  colsample_bytree=0.8, objective= 'binary:logistic',
                                                  scale_pos_weight=1,seed=27), 
                        param_grid = param_test1,scoring='accuracy',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(X_train.values,y_train.values)
gsearch1.best_params_, gsearch1.best_score_


# ({'n_estimators': 80}, 0.8306494426706699)

# n_estimators: 30

# **Step2: Max_depth and min_weight parameters are tuned.**

# In[ ]:


param_test1 = { 'max_depth': range(1,10,1),'min_child_weight':np.arange(1,10,1)}
gsearch1 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,
                                                  min_child_weight=3,gamma=0, subsample=0.8,             
                                                  colsample_bytree=0.8, objective= 'binary:logistic',
                                                  scale_pos_weight=1,seed=27), 
                        param_grid = param_test1,scoring='accuracy',n_jobs=-1,iid=False, cv=5)
gsearch1.fit(X_train.values,y_train.values)
gsearch1.best_params_, gsearch1.best_score_


# ({'max_depth': 5, 'min_child_weight': 3}, 0.8351439665478277)

# **Gamma parameter tuning.**

# In[ ]:


param_test3 = { 'gamma':[i/10.0 for i in range(0,5)]}
gsearch3 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5, 
                                                  min_child_weight=3, gamma=0, subsample=0.8,
                                                  colsample_bytree=0.8, objective= 'binary:logistic', 
                                                  scale_pos_weight=1,seed=27), 
                        param_grid = param_test3, scoring='accuracy',n_jobs=-1,iid=False, cv=5)

gsearch3.fit(X_train.values,y_train.values)
gsearch3.best_params_, gsearch3.best_score_


# 'gamma': 0.0

# **Step 4: Adjust the subsample and colsample_bytree parameters.**

# In[ ]:


param_test4 = {'subsample':[0.8,0.9,1.0], 'colsample_bytree':[0.8,0.9,1.0]}
gsearch4 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,
                                                  min_child_weight=3, gamma=0, subsample=1.0, 
                                                  colsample_bytree=0.9, objective= 'binary:logistic', 
                                                  scale_pos_weight=1,seed=27),
                        param_grid = param_test4, scoring='accuracy',n_jobs=-1,iid=False, cv=5)

gsearch4.fit(X_train.values,y_train.values)
gsearch4.best_params_, gsearch4.best_score_


# ({'colsample_bytree': 0.9, 'subsample': 1.0}, 0.8373596303550587)

# **Step5: Regularization parameter tuning.**

# In[ ]:


param_test5 = { 'reg_alpha':[0.005,0.01,0.015,0.02]}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,
                                                  min_child_weight=3, gamma=0, subsample=1.0, 
                                                  colsample_bytree=0.9,reg_alpha = 0.01
                                                  objective= 'binary:logistic', 
                                                  scale_pos_weight=1,seed=27), 
                        param_grid = param_test5, scoring='accuracy',n_jobs=-1,iid=False, cv=5)

gsearch5.fit(X_train.values,y_train.values)
gsearch5.best_params_, gsearch5.best_score_


# ({'reg_alpha': 0.01}, 0.8396068922936377)

# In[ ]:


param_test5 = { 'reg_alpha':[i/1000 for i in range(9,12)]}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,
                                                  min_child_weight=3, gamma=0, subsample=1.0, 
                                                  colsample_bytree=0.9,reg_alpha = 0.011,
                                                  objective= 'binary:logistic', 
                                                  scale_pos_weight=1,seed=27), 
                        param_grid = param_test5, scoring='accuracy',n_jobs=-1,iid=False, cv=5)

gsearch5.fit(X_train.values,y_train.values)
gsearch5.best_params_, gsearch5.best_score_


# ({'reg_alpha': 0.011}, 0.8407304877992556)

# **Step6: Reduce the learning rate.**

# In[ ]:


param_test5 = {'learning_rate': [0.05,0.1]}
gsearch5 = GridSearchCV(estimator = XGBClassifier(learning_rate =0.1, n_estimators=80, max_depth=5,
                                                  min_child_weight=3, gamma=0, subsample=1.0, 
                                                  colsample_bytree=0.9,reg_alpha = 0.011,
                                                  objective= 'binary:logistic', 
                                                  scale_pos_weight=1,seed=27), 
                        param_grid = param_test5, scoring='accuracy',n_jobs=-1,iid=False, cv=5)

gsearch5.fit(X_train.values,y_train.values)
gsearch5.best_params_, gsearch5.best_score_


# ({'learning_rate': 0.1}, 0.8407304877992556)
