#!/usr/bin/env python
# coding: utf-8

# The following single model scores 0.87169 on the private leaderboard, between 12th & 13th (private).

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import lightgbm as lgb
from sklearn import metrics
from sklearn.metrics import roc_auc_score
import pickle
import os
import gc
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
gc.enable()


# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


def pre_pro(df):
    df = df.astype('float32')
    col = df.columns
    for i in range(len(col)):
        m = df.loc[df[col[i]] != -np.inf, col[i]].min()
        df[col[i]].replace(-np.inf,m,inplace=True)
        M = df.loc[df[col[i]] != np.inf, col[i]].max()
        df[col[i]].replace(np.inf,M,inplace=True)
    
    df.fillna(0, inplace = True)
    return df 


# In[ ]:


def feat_eng(df):
    df.replace(0, 0.001)
    
    df['follower_diff'] = (df['A_follower_count'] > df['B_follower_count'])
    df['following_diff'] = (df['A_following_count'] > df['B_following_count'])
    df['listed_diff'] = (df['A_listed_count'] > df['B_listed_count'])
    df['ment_rec_diff'] = (df['A_mentions_received'] > df['B_mentions_received'])
    df['rt_rec_diff'] = (df['A_retweets_received'] > df['B_retweets_received'])
    df['ment_sent_diff'] = (df['A_mentions_sent'] > df['B_mentions_sent'])
    df['rt_sent_diff'] = (df['A_retweets_sent'] > df['B_retweets_sent'])
    df['posts_diff'] = (df['A_posts'] > df['B_posts'])

    df['A_pop_ratio'] = df['A_mentions_sent']/df['A_listed_count']
    df['A_foll_ratio'] = df['A_follower_count']/df['A_following_count']
    df['A_ment_ratio'] = df['A_mentions_sent']/df['A_mentions_received']
    df['A_rt_ratio'] = df['A_retweets_sent']/df['A_retweets_received']
    
    df['B_pop_ratio'] = df['B_mentions_sent']/df['B_listed_count']
    df['B_foll_ratio'] = df['B_follower_count']/df['B_following_count']
    df['B_ment_ratio'] = df['B_mentions_sent']/df['B_mentions_received']
    df['B_rt_ratio'] = df['B_retweets_sent']/df['B_retweets_received']
    
    df['A/B_foll_ratio'] = (df['A_foll_ratio'] > df['B_foll_ratio'])
    df['A/B_ment_ratio'] = (df['A_ment_ratio'] > df['B_ment_ratio'])
    df['A/B_rt_ratio'] = (df['A_rt_ratio'] > df['B_rt_ratio'])

    df['nf1_diff'] = (df['A_network_feature_1'] > df['B_network_feature_1'])
    df['nf2_diff'] = (df['A_network_feature_2'] > df['B_network_feature_2'])
    df['nf3_diff'] = (df['A_network_feature_3'] > df['B_network_feature_3'])
    
    df['nf3_ratio'] = df['A_network_feature_3'] / df['B_network_feature_3']
    df['nf2_ratio'] = df['A_network_feature_2'] / df['B_network_feature_2']
    df['nf1_ratio'] = df['A_network_feature_1'] / df['B_network_feature_1']
    
    return(pre_pro(df))
# # # # # # # # # # # # # # # # # # # # # #


# In[ ]:


fe_train = feat_eng(train.copy())
fe_test = feat_eng(test.copy())


# In[ ]:


train_df = fe_train
test_df = fe_test
y_train = np.array(train_df['Choice'])


# In[ ]:


target = 'Choice'
predictors = train_df.columns.values.tolist()[1:]


# Parameters came from a Bayesian Optimized Parameter Search

# In[ ]:


bayesian_tr_index, bayesian_val_index  = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=1).split(train_df, train_df.Choice.values))[0]

def LGB_bayesian(
     num_leaves,  # int
     min_data_in_leaf,  # int
     learning_rate,
     min_sum_hessian_in_leaf,    # int  
     feature_fraction,
     lambda_l1,
     lambda_l2,
     min_gain_to_split,
     max_depth):
    
     # LightGBM expects next three parameters need to be integer. So we make them integer
     num_leaves = int(round(num_leaves))
     min_data_in_leaf = int(round(min_data_in_leaf))
     max_depth = int(round(max_depth))

     assert type(num_leaves) == int
     assert type(min_data_in_leaf) == int
     assert type(max_depth) == int

     param = {
         'num_leaves': num_leaves,
         'max_bin': 63,
         'min_data_in_leaf': min_data_in_leaf,
         'learning_rate': learning_rate,
         'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
         'bagging_fraction': 1.0,
         'bagging_freq': 5,
         'feature_fraction': feature_fraction,
         'lambda_l1': lambda_l1,
         'lambda_l2': lambda_l2,
         'min_gain_to_split': min_gain_to_split,
         'max_depth': max_depth,
         'save_binary': True, 
         'seed': 1337,
         'feature_fraction_seed': 1337,
         'bagging_seed': 1337,
         'drop_seed': 1337,
         'data_random_seed': 1337,
         'objective': 'binary',
         'boosting_type': 'gbdt',
         'verbose': 1,
         'metric': 'auc',
         'is_unbalance': True,
         'boost_from_average': False,   

     }    
    
    
     xg_train = lgb.Dataset(train_df.iloc[bayesian_tr_index][predictors].values,
                            label=train_df.iloc[bayesian_tr_index][target].values,
                            feature_name=predictors,
                            free_raw_data = False
                            )
     xg_valid = lgb.Dataset(train_df.iloc[bayesian_val_index][predictors].values,
                            label=train_df.iloc[bayesian_val_index][target].values,
                            feature_name=predictors,
                            free_raw_data = False
                            )   

     num_round = 5000
     clf = lgb.train(param, xg_train, num_round, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 50)
    
     predictions = clf.predict(train_df.iloc[bayesian_val_index][predictors].values, num_iteration=clf.best_iteration)   
    
     score = metrics.roc_auc_score(train_df.iloc[bayesian_val_index][target].values, predictions)
    
     return score


# In[ ]:


# # Bounded region of parameter space
bounds_LGB = {
     'num_leaves': (2, 5), 
     'min_data_in_leaf': (1, 10),  
     'learning_rate': (0.03, 0.07),
     'min_sum_hessian_in_leaf': (0.1, 0.5),    
     'feature_fraction': (0.2, 0.5),
     'lambda_l1': (0, 1), 
     'lambda_l2': (0, 1), 
     'min_gain_to_split': (0.1, 1.0),
     'max_depth':(2,10),
 }

from bayes_opt import BayesianOptimization

LGB_BO = BayesianOptimization(LGB_bayesian, bounds_LGB, random_state=13)

print(LGB_BO.space.keys)

init_points = 10
n_iter = 10

target = 'Choice'
predictors = train_df.columns.values.tolist()[1:]

print('-' * 130)

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    LGB_BO.maximize(init_points=init_points, n_iter=n_iter, acq='ucb', xi=0.0, alpha=1e-6)


# In[ ]:


LGB_BO.max
## used to updat first 9 parameters following


# In[ ]:


param_lgb = {
        'feature_fraction': 0.4647875434283183,
        'lambda_l1': 0.14487098904632512,
        'lambda_l2': 0.9546002933329684,
        'learning_rate': 0.050592093295320606,
        'max_depth': int(round(7.696194993998026)),
        'min_data_in_leaf': int(round(9.879507661608065)),
        'min_gain_to_split': 0.7998292013880356,
        'min_sum_hessian_in_leaf': 0.24962103361366683,
        'num_leaves': int(round(2.854239951949671)),
        'max_bin': 63,
        'bagging_fraction': 1.0, 
        'bagging_freq': 5, 
        'save_binary': True,
        'seed': 1965,
        'feature_fraction_seed': 1965,
        'bagging_seed': 1965,
        'drop_seed': 1965,
        'data_random_seed': 1965,
        'objective': 'binary',
        'boosting_type': 'gbdt',
        'verbose': 1,
        'metric': 'auc',
        'is_unbalance': True,
        'boost_from_average': False}


# In[ ]:


nfold = 20

skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=2019)

oof = np.zeros(len(fe_train))
predictions = np.zeros((len(fe_test),nfold))

i = 1
for train_index, valid_index in skf.split(fe_train, fe_train.Choice.values):
    print("\nfold {}".format(i))

    xg_train = lgb.Dataset(fe_train.iloc[train_index][predictors].values,
                           label=fe_train.iloc[train_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )
    xg_valid = lgb.Dataset(fe_train.iloc[valid_index][predictors].values,
                           label=fe_train.iloc[valid_index][target].values,
                           feature_name=predictors,
                           free_raw_data = False
                           )   

    
    clf = lgb.train(param_lgb, xg_train, 10000000, valid_sets = [xg_valid], verbose_eval=250, early_stopping_rounds = 100)
    oof[valid_index] = clf.predict(fe_train.iloc[valid_index][predictors].values, num_iteration=clf.best_iteration) 
    
    predictions[:,i-1] += clf.predict(fe_test[predictors], num_iteration=clf.best_iteration)
    i = i + 1

print("\n\nCV AUC: {:<0.8f}".format(metrics.roc_auc_score(fe_train.Choice.values, oof)))


# In[ ]:


lgb_bay = []

for i in range(len(predictions)):
    lgb_bay.append(predictions[i][-1])


# In[ ]:


submission = pd.read_csv('../input/sample_predictions.csv')
submission['Choice'] = lgb_bay
submission.to_csv('sub.csv', index = False, header = True)

