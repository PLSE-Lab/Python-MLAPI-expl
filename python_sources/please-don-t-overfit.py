#!/usr/bin/env python
# coding: utf-8

# # General information
# 
# In Don't Overfit! II competition we have a binary classification task. 300 columns, 250 training samples and 79 times more samples in test data! We need to be able to build a model without overfitting.
# 
# In this kernel I'll try the following things:
# 
# 
# * EDA on the features;
# * Feature Selection and Feature Engineering using Principle Components Analysis;
# * Find the number of efficient features for Feature selection and for Feature Engineering;
# * Do crossvalidation on Random Forest for Hyperparameter optimization;
# * Save the model;
# 
# 
# ![](https://cdn-images-1.medium.com/max/1600/1*vuZxFMi5fODz2OEcpG-S1g.png)

# In[ ]:


# Libraries
import numpy as np
import pandas as pd
pd.set_option('max_columns', None)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

import datetime
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score, GridSearchCV, RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
import os
import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
from sklearn import model_selection
from sklearn.metrics import accuracy_score, roc_auc_score
import json
import ast
import time
import eli5
from eli5.sklearn import PermutationImportance

from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_selection import GenericUnivariateSelect, SelectPercentile, SelectKBest, f_classif, mutual_info_classif, RFE
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')
from catboost import CatBoostClassifier


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train.shape


# ## Data exploration

# In[ ]:


train.head()


# # Try to see if a data point represents a picture but failed

# In[ ]:


# row_one = train.loc[0,'0':]
# row_one = np.array(row_one)
# row_one = np.absolute(row_one)


# In[ ]:


# from PIL import Image
# img = Image.fromarray(row_one)
# img.show()


# In[ ]:


train[train.columns[2:]].std().plot('hist');
plt.title('Distribution of stds of all columns');


# In[ ]:


train[train.columns[2:]].mean().plot('hist');
plt.title('Distribution of means of all columns');


# In[ ]:


# we have no missing values
train.isnull().any().any()


# In[ ]:


print('Distributions of first 28 columns')
plt.figure(figsize=(26, 24))
for i, col in enumerate(list(train.columns)[2:30]):
    plt.subplot(7, 4, i + 1)
    plt.hist(train[col])
    plt.title(col)


# In[ ]:


train['target'].value_counts()


# From this overview we can see the following things:
# * target is binary and has some disbalance: 36% of samples belong to 0 class;
# * values in columns are more or less similar;
# * columns have std of 1 +/- 0.1 (min and max values are 0.889, 1.117 respectively);
# * columns have mean of 0 +/- 0.15 (min and max values are -0.2, 0.1896 respectively);

# Let's have a look at correlations now!

# In[ ]:


corrs = train.corr().abs().unstack().sort_values(kind="quicksort").reset_index()
corrs = corrs[corrs['level_0'] != corrs['level_1']]
corrs.tail(10)


# We can see that correlations between features are lower that 0.3 and the most correlated feature with target has correlation of 0.37. So we have no highly correlated features which we could drop, on the other hand we could drop some columns with have little correlation with the target.

# ## Basic modelling

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
get_ipython().run_line_magic('pinfo', 'RandomForestClassifier')


# # Above is to see the default settings of RandomForest

# In[ ]:


X_train = train.drop(['id', 'target'], axis=1)
y_train = train['target']
X_test = test.drop(['id'], axis=1)
n_fold = 20
folds = StratifiedKFold(n_splits=n_fold, shuffle=True, random_state=42)
repeated_folds = RepeatedStratifiedKFold(n_splits=10, n_repeats=20, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


def train_model(X, X_test, y, params, folds=folds, model_type='lgb', plot_feature_importance=False, averaging='usual', model=None):
    oof = np.zeros(len(X))
    prediction = np.zeros(len(X_test))
    scores = []
    feature_importance = pd.DataFrame()
    for fold_n, (train_index, valid_index) in enumerate(folds.split(X, y)):
        # print('Fold', fold_n, 'started at', time.ctime())
        X_train, X_valid = X[train_index], X[valid_index]
        y_train, y_valid = y[train_index], y[valid_index]
            
        if model_type == 'sklearn':
            model = model
            model.fit(X_train, y_train)
            y_pred_valid = model.predict(X_valid).reshape(-1,)
            score = roc_auc_score(y_valid, y_pred_valid)
            # print(f'Fold {fold_n}. AUC: {score:.4f}.')
            # print('')
            
            y_pred = model.predict_proba(X_test)[:, 1]
            
        oof[valid_index] = y_pred_valid.reshape(-1,)
        scores.append(roc_auc_score(y_valid, y_pred_valid))

        if averaging == 'usual':
            prediction += y_pred
        elif averaging == 'rank':
            prediction += pd.Series(y_pred).rank().values  
        
    prediction /= n_fold
    print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
    
    if model_type == 'lgb':
        feature_importance["importance"] /= n_fold
        if plot_feature_importance:
            cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
                by="importance", ascending=False)[:50].index

            best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

            plt.figure(figsize=(16, 12));
            sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
            plt.title('LGB Features (avg over folds)');
        
            return oof, prediction, feature_importance
        return oof, prediction, scores
    
    else:
        return oof, prediction, scores


# # Random Forest Basic Model Creation and Cross Validation on the whole dataset

# # Cross Validation

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

parameter_grid = {'n_estimators': [10,100,500,1000],
                  'max_depth': [None,2,3,5]
                 }

grid_search = GridSearchCV(rfc, param_grid=parameter_grid, cv=folds, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train, y_train)
print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters: {}'.format(grid_search.best_params_))
rfc = RandomForestClassifier(**grid_search.best_params_)
oof_rfc, prediction_rfc, scores_rfc = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=rfc)


# In[ ]:


model = rfc


# ## ELI5 for Feature Selection
# 
# ELI5 is a package with provides explanations for ML models. It can do this not only for linear models, but also for tree based like Random Forest or lightgbm.

# In[ ]:


perm = PermutationImportance(model, random_state=1).fit(X_train, y_train)
eli5.show_weights(perm, top=50)


# # Select Top 30 features

# In[ ]:


top_features = [i[1:] for i in eli5.formatters.as_dataframe.explain_weights_df(model).feature][:30]
top_features


# 

# In[ ]:


X_train = train[top_features]
X_test = test[top_features]
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[ ]:


oof_lr, prediction_lr, scores = train_model(X_train, X_test, y_train, params=None, model_type='sklearn', model=model)
print(scores)


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')
submission['target'] = prediction_lr
submission.to_csv('submission_top30.csv', index=False)
submission.head()


# In[ ]:




