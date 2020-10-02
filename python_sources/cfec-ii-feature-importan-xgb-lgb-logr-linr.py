#!/usr/bin/env python
# coding: utf-8

# <a class="anchor" id="0"></a>
# 
# # Building the feature importance diagrams in 6 ways on the example of competition "Titanic: Machine Learning from Disaster":
# * XGB
# * LGBM
# * LogisticRegression (with Eli5)
# * LinearRegression (with Eli5)
# * Mean
# * Total with different weights

# <a class="anchor" id="0.1"></a>
# 
# ## Table of Contents
# 
# 1. [Import libraries](#1)
# 1. [Download datasets](#2)
# 1. [FE & EDA](#3)
# 1. [Preparing to modeling](#4)
# 1. [Tuning models and building the feature importance diagrams](#5)
#     -  [LGBM](#5.1)
#     -  [XGB](#5.2)
#     -  [Logistic Regression](#5.3)
#     -  [Linear Regression](#5.4)
# 1. [Comparison of the all feature importance diagrams](#6)

# ## 1. Import libraries <a class="anchor" id="1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import eli5

import lightgbm as lgbm
import xgboost as xgb

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import warnings
warnings.filterwarnings("ignore")


# ## 2. Download datasets <a class="anchor" id="2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


train = pd.read_csv('../input/cat-in-the-dat-ii/train.csv')


# In[ ]:


train.head(3)


# In[ ]:


train.info()


# ## 3. FE & EDA <a class="anchor" id="3"></a>
# 
# [Back to Table of Contents](#0.1)

# While we take all the features

# In[ ]:


pd.set_option('max_columns',100)


# In[ ]:


#train.head(5)


# In[ ]:


#train.info()


# ## 4. Preparing to modeling <a class="anchor" id="4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Thanks to : https://www.kaggle.com/aantonova/some-new-risk-and-clusters-features
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


# In[ ]:


numerics = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
categorical_columns = []
features = train.columns.values.tolist()
for col in features:
    if train[col].dtype in numerics: continue
    categorical_columns.append(col)
indexer = {}
for col in categorical_columns:
    if train[col].dtype in numerics: continue
    _, indexer[col] = pd.factorize(train[col])
    
for col in categorical_columns:
    if train[col].dtype in numerics: continue
    train[col] = indexer[col].get_indexer(train[col])


# In[ ]:


target = train.pop('target')


# In[ ]:


train = train.fillna(0)


# In[ ]:


train = reduce_mem_usage(train)


# In[ ]:


train.info()


# ## 5. Tuning models and building the feature importance diagrams<a class="anchor" id="5"></a>
# 
# [Back to Table of Contents](#0.1)

# ### 5.1 LGBM <a class="anchor" id="5.1"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


X = train
z = target


# In[ ]:


#%% split training set to validation set
Xtrain, Xval, Ztrain, Zval = train_test_split(X, z, test_size=0.2, random_state=0)
train_set = lgbm.Dataset(Xtrain, Ztrain, silent=False)
valid_set = lgbm.Dataset(Xval, Zval, silent=False)


# In[ ]:


params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,        
    }

modelL = lgbm.train(params, train_set = train_set, num_boost_round=1000,
                   early_stopping_rounds=50,verbose_eval=10, valid_sets=valid_set)


# In[ ]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
lgbm.plot_importance(modelL,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


feature_score = pd.DataFrame(train.columns, columns = ['feature']) 
feature_score['score_lgb'] = modelL.feature_importance()


# ### 5.2 XGB<a class="anchor" id="5.2"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#%% split training set to validation set 
data_tr  = xgb.DMatrix(Xtrain, label=Ztrain)
data_cv  = xgb.DMatrix(Xval   , label=Zval)
evallist = [(data_tr, 'train'), (data_cv, 'valid')]


# In[ ]:


parms = {'max_depth':8, #maximum depth of a tree
         'objective':'reg:logistic',
         'eta'      :0.3,
         'subsample':0.8,#SGD will use this percentage of data
         'lambda '  :4, #L2 regularization term,>1 more conservative 
         'colsample_bytree ':0.9,
         'colsample_bylevel':1,
         'min_child_weight': 10}
modelx = xgb.train(parms, data_tr, num_boost_round=200, evals = evallist,
                  early_stopping_rounds=30, maximize=False, 
                  verbose_eval=10)

print('score = %1.5f, n_boost_round =%d.'%(modelx.best_score,modelx.best_iteration))


# In[ ]:


fig =  plt.figure(figsize = (15,15))
axes = fig.add_subplot(111)
xgb.plot_importance(modelx,ax = axes,height = 0.5)
plt.show();plt.close()


# In[ ]:


feature_score['score_xgb'] = feature_score['feature'].map(modelx.get_score(importance_type='weight'))
feature_score


# ### 5.3 Logistic Regression <a class="anchor" id="5.3"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Standardization for regression models
train = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(train),
    columns=train.columns,
    index=train.index
)


# In[ ]:


# Logistic Regression

logreg = LogisticRegression()
logreg.fit(train, target)
coeff_logreg = pd.DataFrame(train.columns.delete(0))
coeff_logreg.columns = ['feature']
coeff_logreg["score_logreg"] = pd.Series(logreg.coef_[0])
coeff_logreg.sort_values(by='score_logreg', ascending=False)


# In[ ]:


len(coeff_logreg)


# In[ ]:


# the level of importance of features is not associated with the sign
coeff_logreg["score_logreg"] = coeff_logreg["score_logreg"].abs()
feature_score = pd.merge(feature_score, coeff_logreg, on='feature')


# In[ ]:


# Eli5 visualization
eli5.show_weights(logreg)


# ### 5.4 Linear Regression <a class="anchor" id="5.4"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


# Linear Regression

linreg = LinearRegression()
linreg.fit(train, target)
coeff_linreg = pd.DataFrame(train.columns.delete(0))
coeff_linreg.columns = ['feature']
coeff_linreg["score_linreg"] = pd.Series(linreg.coef_)
coeff_linreg.sort_values(by='score_linreg', ascending=False)


# In[ ]:


# Eli5 visualization
eli5.show_weights(linreg)


# In[ ]:


# the level of importance of features is not associated with the sign
coeff_linreg["score_linreg"] = coeff_linreg["score_linreg"].abs()


# In[ ]:


feature_score = pd.merge(feature_score, coeff_linreg, on='feature')
feature_score = feature_score.fillna(0)
feature_score = feature_score.set_index('feature')
feature_score


# ### 6. Comparison of the all feature importance diagrams <a class="anchor" id="6"></a>
# 
# [Back to Table of Contents](#0.1)

# In[ ]:


#Thanks to https://www.kaggle.com/nanomathias/feature-engineering-importance-testing
# MinMax scale all importances
feature_score = pd.DataFrame(
    preprocessing.MinMaxScaler().fit_transform(feature_score),
    columns=feature_score.columns,
    index=feature_score.index
)

# Create mean column
feature_score['mean'] = feature_score.mean(axis=1)

# Plot the feature importances
feature_score.sort_values('mean', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


feature_score.sort_values('mean', ascending=False)


# In[ ]:


# Create total column with different weights
feature_score['total'] = 0.5*feature_score['score_lgb'] + 0.3*feature_score['score_xgb']                        + 0.1*feature_score['score_logreg'] + 0.1*feature_score['score_linreg']

# Plot the feature importances
feature_score.sort_values('total', ascending=False).plot(kind='bar', figsize=(20, 10))


# In[ ]:


feature_score.sort_values('total', ascending=False)

