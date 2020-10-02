#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')
import os
# Any results you write to the current directory are saved as output.


# In[ ]:


train_application_df = pd.read_csv('../input/application_train.csv')
test_application_df = pd.read_csv('../input/application_test.csv')
previous_application_df = pd.read_csv('../input/previous_application.csv')
all_application_df = pd.concat([train_application_df, test_application_df], axis=0)


# # Introduction
# * All Features from application csv + 10 top Reduced Components
# * PCA: Principal Component Analysis (unsupervised)
# * LDA: Linear Discriminant Analysis (supervised)
# * PLS: Partial Least Square SVD (supervised)
# 
# # TLDR/Result
# ## [V1] Used all features & fillna with -1 for reduction
# | Method        | 5-fold mean AUC |
# | ------------- :-------------|
# | Baseline     |  0.75982 std: 0.0030946 | 
# | MinMaxScaler, PCA     | 0.75852 std: 0.0030524  |
# | StandardScaler, PCA |   0.75834 std: 0.0029477  |
# | RobustScaler, PCA |  0.75826 std: 0.0034292 |
# | MinMaxScaler, LDA  | 0.76070 std: 0.0040072  |
# | StandardScaler, LDA | 0.76043 std: 0.0040373  |
# | RobustScaler, LDA |  **0.76080**  std: 0.0042677 |
# | MinMaxScaler, PLS  | 0.75966 std: 0.0033254  |
# | StandardScaler, PLS  | 0.75901 std: 0.0033388 |
# | RobustScaler, PLS |  0.75932 std: 0.0038315 |
# 
# ## [V2] Used only numerical features & fillna with mean for reduction
# | Method        | 5-fold mean AUC |
# | ------------- :-------------|
# | Baseline     |  0.75982 std: 0.0030946 | 
# | MinMaxScaler, PCA     | 0.75849 std: 0.0031068  |
# | StandardScaler, PCA |   0.75888 std: 0.0031630  |
# | RobustScaler, PCA |   0.75903 std: 0.0026790 |
# | MinMaxScaler, LDA  |  **0.76058** std: 0.0030256  |
# | StandardScaler, LDA | 0.76042 std: 0.0031176  |
# | RobustScaler, LDA |  0.76051 std: 0.0029084 |
# | MinMaxScaler, PLS  | 0.75941 std: 0.0036380  |
# | StandardScaler, PLS  | 0.75944 std: 0.0034895 |
# | RobustScaler, PLS |  0.75905 std: 0.0034893  |
# 
# ## [V3] Used all features & fillna with mean for reduction
# | Method        | 5-fold mean AUC |
# | ------------- :-------------|
# | Baseline     |  0.75982 std: 0.0030946 | 
# | MinMaxScaler, PCA     |  0.75857 std: 0.0023189  |
# | StandardScaler, PCA |   0.75842 std: 0.0028206  |
# | RobustScaler, PCA |   0.75857 std: 0.0030156  |
# | MinMaxScaler, LDA  |   0.76049 std: 0.0034916  |
# | StandardScaler, LDA | 0.76042 std: 0.0033096 |
# | RobustScaler, LDA |  **0.76052** std: 0.0035620  |
# | MinMaxScaler, PLS  | 0.75990 std: 0.0034749  |
# | StandardScaler, PLS  | 0.75934 std: 0.0032347 |
# | RobustScaler, PLS |  0.75955 std: 0.0032429  |

# # Label encoding

# In[ ]:


categorical_features = []

for column in train_application_df.columns:
    if column == 'TARGET' or column == 'SK_ID_CURR':
        continue
    if train_application_df[column].dtype == 'object':
        all_application_df[column], _ = pd.factorize(all_application_df[column])
        categorical_features.append(column)
categorical_features = list(set(categorical_features))
print(len(categorical_features), categorical_features)


# # 1. Baseline

# In[ ]:


# Code for cross validation
def cross_validation(df):
    X = df[df['TARGET'].notna()]
    Y = X.pop('TARGET')
    if 'SK_ID_CURR' in df.columns:
        X.pop('SK_ID_CURR')
    num_fold = 5
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=2018)
    valid_scores = []
    train_scores = []
    
    for train_index, test_index in skf.split(X, Y):
        X_train, X_validation = X.iloc[train_index], X.iloc[test_index]
        y_train, y_validation = Y.iloc[train_index], Y.iloc[test_index]
        
        clf = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            n_estimators=1000,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            subsample=0.8,
            max_depth=8,
            reg_alpha=1,
            reg_lambda=1,
            min_child_weight=40,
            random_state=2018,
            nthread=-1
            )
                   
        clf.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_validation, y_validation)], 
                eval_metric='auc',
                verbose = False,
                early_stopping_rounds=100
                )
        
        train_prediction = clf.predict_proba(X_train)[:, 1]
        train_score = roc_auc_score(y_train, train_prediction)
        train_scores.append(train_score)
        
        valid_prediction = clf.predict_proba(X_validation)[:, 1]
        valid_score = roc_auc_score(y_validation, valid_prediction)
        valid_scores.append(valid_score)
        
        print('Fold', train_score, valid_score, clf.best_iteration_)
    print('AUC mean:', np.mean(valid_scores), 'std:',np.std(valid_scores))
    


# In[ ]:


cross_validation(all_application_df)


# # Filter Features for Dimension Reduction

# In[ ]:


all_features = list(all_application_df.columns)
all_features.remove('TARGET')
all_features.remove('SK_ID_CURR')

print(len(all_features))
print(all_features)


# # 2. Dimensionality Reduction
# ## 2.1 PCA

# In[ ]:


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
import matplotlib.pyplot as plt


# Code for cross validation
def cross_validation_with_reduction(df, reducer, scaler):
    X = df[df['TARGET'].notna()]
    Y = X.pop('TARGET')
    if 'SK_ID_CURR' in df.columns:
        X.pop('SK_ID_CURR')
    num_fold = 5
    
     # fill na
    for feature in all_features:
        X[feature] = X[feature].fillna(X[feature].mean())
    
    # scaling
    X[all_features] = pd.DataFrame(scaler.fit_transform(X[all_features]))
    
    skf = StratifiedKFold(n_splits=num_fold, shuffle=True, random_state=2018)
    valid_scores = []
    train_scores = []
    
    for train_index, test_index in skf.split(X, Y):
        X_train, X_validation = X.iloc[train_index], X.iloc[test_index]
        y_train, y_validation = Y.iloc[train_index], Y.iloc[test_index]
        
        
        reducer.fit(X_train[all_features], y_train)
        train_reduced_samples = pd.DataFrame(reducer.transform(X_train[all_features]))
        valid_reduced_samples = pd.DataFrame(reducer.transform(X_validation[all_features]))
        
        for feature in train_reduced_samples.columns:
            X_train[feature] = train_reduced_samples[feature].values
        
        for feature in valid_reduced_samples.columns:
            X_validation[feature] = valid_reduced_samples[feature].values
        
        clf = LGBMClassifier(
            boosting_type='gbdt',
            objective='binary',
            n_estimators=1000,
            learning_rate=0.1,
            num_leaves=31,
            feature_fraction=0.8,
            subsample=0.8,
            max_depth=8,
            reg_alpha=1,
            reg_lambda=1,
            min_child_weight=40,
            random_state=2018,
            nthread=-1
            )
                   
        clf.fit(X_train, y_train, 
                eval_set=[(X_train, y_train), (X_validation, y_validation)], 
                eval_metric='auc',
                verbose = False,
                early_stopping_rounds=100
                )
        
        train_prediction = clf.predict_proba(X_train)[:, 1]
        train_score = roc_auc_score(y_train, train_prediction)
        train_scores.append(train_score)
        
        valid_prediction = clf.predict_proba(X_validation)[:, 1]
        valid_score = roc_auc_score(y_validation, valid_prediction)
        valid_scores.append(valid_score)
        
        print('Fold', train_score, valid_score, clf.best_iteration_)
    print('AUC mean:', np.mean(valid_scores), 'std:',np.std(valid_scores))


# In[ ]:


from sklearn.decomposition import PCA


# ### 2.1.1 MinMaxScaler, PCA

# In[ ]:


cross_validation_with_reduction(all_application_df, PCA(n_components=10), MinMaxScaler())


# ### 2.1.2 StandardScaler, PCA

# In[ ]:


cross_validation_with_reduction(all_application_df, PCA(n_components=10), MinMaxScaler())


# ### 2.1.3 RobustScaler, PCA

# In[ ]:


cross_validation_with_reduction(all_application_df, PCA(n_components=10), RobustScaler())


# ## 2.2 LDA
# 

# In[ ]:


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


# ### 2.2.1 MinMaxScaler , LDA

# In[ ]:


cross_validation_with_reduction(all_application_df, LDA(n_components=10), MinMaxScaler())


# ### 2.2.2 StandardScaler , LDA

# In[ ]:


cross_validation_with_reduction(all_application_df, LDA(n_components=10), StandardScaler())


# ### 2.2.3 RobustScaler , LDA

# In[ ]:


cross_validation_with_reduction(all_application_df, LDA(n_components=10), RobustScaler())


# ## 2.3 PLS

# In[ ]:


from sklearn.cross_decomposition import PLSSVD


# ### 2.3.1 MinMaxScaler , PLS

# In[ ]:


cross_validation_with_reduction(all_application_df, PLSSVD(n_components=10), MinMaxScaler())


# ### 2.3.2 StandardScaler , PLS

# In[ ]:


cross_validation_with_reduction(all_application_df, PLSSVD(n_components=10), StandardScaler())


# ### 2.3.3 RobustScaler , PLS

# In[ ]:


cross_validation_with_reduction(all_application_df, PLSSVD(n_components=10), RobustScaler())


# In[ ]:




