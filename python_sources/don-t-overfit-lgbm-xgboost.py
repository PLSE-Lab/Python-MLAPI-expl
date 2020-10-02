#!/usr/bin/env python
# coding: utf-8

# Playing around with LightGBM and XGBoost on the dont overfit II dataset:

# In[ ]:



import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import seaborn as sns

import xgboost as xgb
import lightgbm as lgb

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


import warnings
warnings.filterwarnings('ignore')

from time import time 

train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

target= train['target']
train = train.drop(['target','id'], axis = 1)
test = test.drop(['id'], axis = 1)


# How not to overfit with gradient boosted trees: 
# * Reduce tree complexity by setting num_leaves or max_depth small.
# * Use a small learning rate
# * Use feature_fraction and bagging_fraction/bagging_freq. 
# * Try to penalize L1 and L2
# * By default min_data_in_leaf is 20, this needs to be reduced in this small dataset, otherwise you will struggle to predict a 0. 

# In[ ]:


params_LGBM = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 4,
    'learning_rate': 0.012,
    'feature_fraction': 0.6,
    'bagging_fraction': 0.5,
    'bagging_freq': 1,
    'verbose': 0,
    'lambda_l1':0.4,
    'lambda_l2':0.9,
    'min_data_in_leaf': 2,
    'max_bin': 25,
    'min_data_in_bin':2    
}
params_XGB =  {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'max_depth': 2,
    'eta': 0.012,
    'subsample': 0.7,    
    'verbosity': 0,
    'alpha':0.4,
    'lambda':0.9,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.8,
    'colsample_bynode': 0.8,
   # 'three_method':'hist'    
}  


# 
# 

# In[ ]:


def cv_LGBM(train,target,params,rounds):      
    lgb_train = lgb.Dataset(train, target)
    hist = lgb.cv(
                params,
                lgb_train,
                num_boost_round=rounds
                   )   
    scores = pd.DataFrame.from_dict(hist)
    return(scores)

def train_LGBM(train,target,params,rounds):
    lgb_train = lgb.Dataset(train,target)
    booster = lgb.train(
        params,lgb_train,
        num_boost_round = rounds,
        verbose_eval = False
    )
    return(booster)

def cv_XGB(train,target,params,rounds):  
    xgb_train = xgb.DMatrix(train, label = target)       
    hist = xgb.cv(
        params,
        xgb_train,
        num_boost_round=rounds,
        stratified = True)  
    scores = pd.DataFrame.from_dict(hist)
    return(scores)

def train_XGB(train,target,params,rounds):  
        xgb_train = xgb.DMatrix(train, label = target)    
        booster = xgb.train(
            params,
            xgb_train,
            num_boost_round = rounds, )
        return(booster)
    


# 

# In[ ]:


scores_LGBM = cv_LGBM(train,target,params_LGBM,rounds = 3000)
    
scores_XGB = cv_XGB(train,target,params_XGB, rounds = 3000)

scores = pd.DataFrame()
scores['XGB'],scores['LGBM'] = scores_XGB['test-auc-mean'],scores_LGBM['auc-mean']
plt.plot(scores)
plt.legend(labels = ('XGB','LGMB'))
print(scores[-10:])


# LightGMB scoring 0.792 in public leaderboard with these parameters and all features. 
# XGBoost scoring 0.795 on public leaderboard with these parameters and all features. 
# 
# Generally; CV is more conservative than the public leaderboard.. 

# 

# Trying if PCA gives improvement:

# In[ ]:


scaler = StandardScaler().fit(train+test) #Shoud only train be fitted? Using train+test gives improvement.

train_scaled = scaler.transform(train)
test_scaled = scaler.transform(test)

pca = PCA(n_components = 50)
train_pca = pca.fit_transform(train_scaled)
test_pca = pca.transform(test_scaled)

scores_LGBM = cv_LGBM(train_pca,target,params_LGBM,rounds = 1000)    
scores_XGB = cv_XGB(train_pca,target,params_XGB, rounds = 1000)

scores = pd.DataFrame()
scores['XGB'],scores['LGBM'] = scores_XGB['test-auc-mean'],scores_LGBM['auc-mean']

plt.plot(scores)
plt.legend(labels = ('XGB','LGMB'))
print(scores[-10:])


# PCA is not very effective..
# 
# Trying feature removal:

# In[ ]:


# Stolen from https://www.kaggle.com/tboyle10/feature-selection
booster = train_LGBM(train,target,params = params_LGBM, rounds = 1000)

feature_importance = booster.feature_importance(importance_type = 'gain')
sorted_idx = np.argsort(feature_importance)
#print(sorted_idx)
plot_idx = sorted_idx[-20:]

pos = np.arange(plot_idx.shape[0]) + .5
plt.barh(pos,feature_importance[plot_idx])
plt.yticks(pos,plot_idx)

plt.title('Feature Importance', fontsize=20)

plt.show()


# In[ ]:


reduced_params_LGBM = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'num_leaves': 6,
    'learning_rate': 0.012,
    'feature_fraction': 0.4,
    'bagging_fraction': 0.35,
    'bagging_freq': 2,
    'verbose': 0,
    'lambda_l1':0.5,
    'lambda_l2':0.9,
    'min_data_in_leaf': 2,
    'max_bin': 200,
    'min_data_in_bin':2    
}


# In[ ]:


#Returns a booster with n less features
def removeAndRun(n_features,train,target,test):
    #Train boosters
    lgbm = train_LGBM(train,target, params = params_LGBM, rounds = 1000)
    xgbm = train_XGB(train,target, params = params_XGB, rounds = 100)
    
    #Finding and removing least important features in lightgbm 
    feature_importance = lgbm.feature_importance(importance_type = 'gain')
    sorted_idx = np.argsort(feature_importance)
    remove = sorted_idx[:n_features]
   
    train_reduced = train
    test_reduced = test
    for index in remove:
        train_reduced = train_reduced.drop(str(index),axis = 1)
        test_reduced = test_reduced.drop(str(index),axis = 1)
        
    #Re-train
    lgbm = train_LGBM(train_reduced,target, params = reduced_params_LGBM, rounds = 5000)
    xgbm = train_XGB(train_reduced,target, params = params_XGB, rounds = 5000)
    
    return lgbm,xgbm,train_reduced,test_reduced

lgbm,xgbm,train_reduced,test_reduced= removeAndRun(250, train, target, test)
scores_LGBM = cv_LGBM(train_reduced,target,reduced_params_LGBM,rounds = 3000)   
scores_XGB = cv_XGB(train_reduced,target,params_XGB, rounds = 3000)

scores = pd.DataFrame()
scores['XGB'],scores['LGBM'] = scores_XGB['test-auc-mean'],scores_LGBM['auc-mean']
print('shape of training data: ' ,np.shape(train_reduced.ix[0]))
plt.plot(scores)
plt.legend(labels = ('XGB','LGMB'))
print(scores[-10:])


# Both XGB and LGB scores the same on public leaderboard after feature removal woth the "old" parameters. 
# 
# LGBM scores 0.81 in public leaderboard with the new reduced_parameters. 
# 
# So the CV-function is now overfitting. LGB seems to underperform more than XGB when comparing to CV. LGB and XGB seems equal on public leaderboard. 

# In[ ]:


#test_xgb = xgb.DMatrix(test_reduced)
#test_lgbm = lgb.Dataset(test_reduced) no need. 
final = lgbm.predict(test_reduced)

plt.scatter(range(300),(final[:300]))
plt.legend(['Float'])


# In[ ]:


submission = pd.read_csv('../input/sample_submission.csv')

submission['target'] = final
submission.to_csv('submission.csv', index=False)


# In[ ]:


lgb.plot_tree(lgbm,tree_index = 0,figsize = (20,20))

