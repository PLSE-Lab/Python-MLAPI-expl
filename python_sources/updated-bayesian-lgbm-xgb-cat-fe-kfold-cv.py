#!/usr/bin/env python
# coding: utf-8

# ---
# ## Introduction
# ---
# #### Over All Strategy
# * **First** Train a Model with normal **LightGBM +Catboost+ XGB **
# * In **Second Part** applied **Feature engineering** and Again Applied **CV+GroupKFOLD+LightGBM +Catboost+ XGB**
# ![](https://cdn-images-1.medium.com/max/2000/1*A0b_ahXOrrijazzJengwYw.png)
# ---
# 
# ### **Notebook Workflow**
# 
# ---
# 
# * [**1.Get the extracted data**](#1.Get-the-extracted-data)
# * [**2.Data Fold using GroupKfold**](#2.Data-Fold-using-GroupKfold)
# * [**3.Target Define transaction**](#3.Target-Define-transaction)
# * [**4.Add date features**](#4.Add-date-features)
# * [**5.Create features list**](#5.Create-features-list)
# * [**6.Factorize categoricals**](#6.Factorize-categoricals)
# * [**7.Cross Validation for Hyperparameter Tuning With Bayesian optimization**](#7.Cross-Validation-for-Hyperparameter-Tuning)
# * [**8.Model Training with Kfold Validation LightGBM**](#8.Model-Training-with-Kfold-Validation-LightGBM)
# * [**9.Display feature importances**](#9.Display-feature-importances)
# * [**10.Create user level predictions**](#10.Create-user-level-predictions)
# * [**11.Create target and Cross Validation**](#11.Create-target-and-Cross-Validation)
# * [**12.Train a model at Visitor level**](#12.Train-a-model-at-Visitor-level)
# * [**13.Display feature importances**](#13.Display-feature-importances)
# * [**14.Save Result**](#14.Save-Result)
# 
# ---

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error
import gc
import time
from pandas.core.common import SettingWithCopyWarning
import warnings
import lightgbm as lgb
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GroupKFold, GridSearchCV

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### 1.Get the extracted data

# In[ ]:


def load_data():
    train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                        dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                       dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
    return train,test
train, test = load_data()
print("Train Shape:", train.shape)
print("Test Shape:",test.shape)


# ### 2.Data Fold using GroupKfold

# In[ ]:


def get_folds(df=None, n_splits=5):
    """Returns dataframe indices corresponding to Visitors Group KFold"""
    # Get sorted unique visitors
    unique_vis = np.array(sorted(df['fullVisitorId'].unique()))

    # Get folds
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for trn_vis, val_vis in folds.split(X=unique_vis, y=unique_vis, groups=unique_vis):
        fold_ids.append(
            [
                ids[df['fullVisitorId'].isin(unique_vis[trn_vis])],
                ids[df['fullVisitorId'].isin(unique_vis[val_vis])]
            ]
        )

    return fold_ids


# ### 3.Target Define transaction

# In[ ]:


y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']


# ### 4.Add date features
# 
# Only add the one I think can ganeralize

# In[ ]:


display(train.columns)


# In[ ]:


for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day
    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    df['next_session_1'] = (
    df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['nb_pageviews'] = df['date'].map(
        df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum()
    )
    
    df['ratio_pageviews'] = df['totals.pageviews'] / df['nb_pageviews']


# In[ ]:


train['target'] = y_reg
y_reg = train['target']
del train['target']


# In[ ]:


# https://www.kaggle.com/prashantkikani/teach-lightgbm-to-sum-predictions-fe
def browser_mapping(x):
    browsers = ['chrome','safari','firefox','internet explorer','edge','opera','coc coc','maxthon','iron']
    if x in browsers:
        return x.lower()
    elif  ('android' in x) or ('samsung' in x) or ('mini' in x) or ('iphone' in x) or ('in-app' in x) or ('playstation' in x):
        return 'mobile browser'
    elif  ('mozilla' in x) or ('chrome' in x) or ('blackberry' in x) or ('nokia' in x) or ('browser' in x) or ('amazon' in x):
        return 'mobile browser'
    elif  ('lunascape' in x) or ('netscape' in x) or ('blackberry' in x) or ('konqueror' in x) or ('puffin' in x) or ('amazon' in x):
        return 'mobile browser'
    elif '(not set)' in x:
        return x
    else:
        return 'others'
    
    
def adcontents_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('placement' in x) | ('placememnt' in x):
        return 'placement'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'ad' in x:
        return 'ad'
    else:
        return 'others'
    
def source_mapping(x):
    if  ('google' in x):
        return 'google'
    elif  ('youtube' in x):
        return 'youtube'
    elif '(not set)' in x or 'nan' in x:
        return x
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'bing' in x:
        return 'bing'
    elif 'quora' in x:
        return 'quora'
    elif 'outlook' in x:
        return 'outlook'
    elif 'linkedin' in x:
        return 'linkedin'
    elif 'pinterest' in x:
        return 'pinterest'
    elif 'ask' in x:
        return 'ask'
    elif 'siliconvalley' in x:
        return 'siliconvalley'
    elif 'lunametrics' in x:
        return 'lunametrics'
    elif 'amazon' in x:
        return 'amazon'
    elif 'mysearch' in x:
        return 'mysearch'
    elif 'qiita' in x:
        return 'qiita'
    elif 'messenger' in x:
        return 'messenger'
    elif 'twitter' in x:
        return 'twitter'
    elif 't.co' in x:
        return 't.co'
    elif 'vk.com' in x:
        return 'vk.com'
    elif 'search' in x:
        return 'search'
    elif 'edu' in x:
        return 'edu'
    elif 'mail' in x:
        return 'mail'
    elif 'ad' in x:
        return 'ad'
    elif 'golang' in x:
        return 'golang'
    elif 'direct' in x:
        return 'direct'
    elif 'dealspotr' in x:
        return 'dealspotr'
    elif 'sashihara' in x:
        return 'sashihara'
    elif 'phandroid' in x:
        return 'phandroid'
    elif 'baidu' in x:
        return 'baidu'
    elif 'mdn' in x:
        return 'mdn'
    elif 'duckduckgo' in x:
        return 'duckduckgo'
    elif 'seroundtable' in x:
        return 'seroundtable'
    elif 'metrics' in x:
        return 'metrics'
    elif 'sogou' in x:
        return 'sogou'
    elif 'businessinsider' in x:
        return 'businessinsider'
    elif 'github' in x:
        return 'github'
    elif 'gophergala' in x:
        return 'gophergala'
    elif 'yandex' in x:
        return 'yandex'
    elif 'msn' in x:
        return 'msn'
    elif 'dfa' in x:
        return 'dfa'
    elif '(not set)' in x:
        return '(not set)'
    elif 'feedly' in x:
        return 'feedly'
    elif 'arstechnica' in x:
        return 'arstechnica'
    elif 'squishable' in x:
        return 'squishable'
    elif 'flipboard' in x:
        return 'flipboard'
    elif 't-online.de' in x:
        return 't-online.de'
    elif 'sm.cn' in x:
        return 'sm.cn'
    elif 'wow' in x:
        return 'wow'
    elif 'baidu' in x:
        return 'baidu'
    elif 'partners' in x:
        return 'partners'
    else:
        return 'others'

train['device.browser'] = train['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
train['trafficSource.adContent'] = train['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
train['trafficSource.source'] = train['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

test['device.browser'] = test['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
test['trafficSource.adContent'] = test['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
test['trafficSource.source'] = test['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')

def process_device(data_df):
    print("process device ...")
    data_df['source.country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '_' + data_df['trafficSource.medium']
    data_df['browser.category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '_' + data_df['device.operatingSystem']
    return data_df

train = process_device(train)
test = process_device(test)

def custom(data):
    print('custom..')
    data['device_deviceCategory_channelGrouping'] = data['device.deviceCategory'] + "_" + data['channelGrouping']
    data['channelGrouping_browser'] = data['device.browser'] + "_" + data['channelGrouping']
    data['channelGrouping_OS'] = data['device.operatingSystem'] + "_" + data['channelGrouping']
    
    for i in ['geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country','geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region','geoNetwork.subContinent']:
        for j in ['device.browser','device.deviceCategory', 'device.operatingSystem', 'trafficSource.source']:
            data[i + "_" + j] = data[i] + "_" + data[j]
    
    data['content.source'] = data['trafficSource.adContent'] + "_" + data['source.country']
    data['medium.source'] = data['trafficSource.medium'] + "_" + data['source.country']
    return data

train = custom(train)
test = custom(test)


# ### 5.Create features list

# In[ ]:


excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime','nb_sessions', 'max_visits'
]
# excluded_features = [
#     'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
#     'visitId', 'visitStartTime', 'vis_date', 'nb_sessions', 'max_visits'
# ]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]


# ### 6.Factorize categoricals

# In[ ]:


for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])


# In[ ]:


print("Train Shape:", train.shape)
print("Test Shape:",test.shape)


# In[ ]:


from sklearn.model_selection import train_test_split
train_features = [_f for _f in train.columns if _f not in excluded_features]
# X_train, X_test, y_train, y_test = train_test_split(train[train_features], y_reg, test_size=0.20, random_state=42)
X_train, y_train = train[train_features], y_reg


# ### 7.Cross Validation for Hyperparameter Tuning
# 
# * Getting Results by Below Codes
# ![](https://www.kaggle.com/ashishpatel26/mlimage/downloads/f1.JPG/4)
# 

# In[ ]:


# # https://www.kaggle.com/qwe1398775315/eda-lgbm-bayesianoptimization
# from bayes_opt import BayesianOptimization

# def lgb_eval(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples,bagging_fraction,feature_fraction):
#     params = {
#     "objective" : "regression",
#     "metric" : "rmse", 
#     "num_leaves" : int(num_leaves),
#     "max_depth" : int(max_depth),
#     "lambda_l2" : lambda_l2,
#     "lambda_l1" : lambda_l1,
#     "num_threads" : 4,
#     "min_child_samples" : int(min_child_samples),
#     "learning_rate" : 0.03,
#     "bagging_fraction" : bagging_fraction,
#     "feature_fraction" : feature_fraction,
#     "subsample_freq" : 5,
#     "bagging_seed" : 42,
#     "verbosity" : -1
#     }
#     lgtrain = lgb.Dataset(X_train, label=np.log1p(y_train.apply(lambda x : 0 if x < 0 else x)))
#     cv_result = lgb.cv(params,
#                        lgtrain,
#                        1500,
# #                        categorical_feature=category_features,
#                        early_stopping_rounds=100,
#                        stratified=False,
#                        nfold=5)
#     return -cv_result['rmse-mean'][-1]

# def lgb_train(num_leaves,max_depth,lambda_l2,lambda_l1,min_child_samples,bagging_fraction,feature_fraction):
#     params = {
#     "objective" : "regression",
#     "metric" : "rmse", 
#     "num_leaves" : int(num_leaves),
#     "max_depth" : int(max_depth),
#     "lambda_l2" : lambda_l2,
#     "lambda_l1" : lambda_l1,
#     "num_threads" : 4,
#     "min_child_samples" : int(min_child_samples),
#     "learning_rate" : 0.03,
#     "bagging_fraction" : bagging_fraction,
#     "feature_fraction" : feature_fraction,
#     "subsample_freq" : 5,
#     "bagging_seed" : 42,
#     "verbosity" : -1
#     }
#     t_x,v_x,t_y,v_y = train_test_split(X_train,y_train,test_size=0.2)
#     lgtrain = lgb.Dataset(t_x, label=np.log1p(t_y.apply(lambda x : 0 if x < 0 else x)))
#     lgvalid = lgb.Dataset(v_x, label=np.log1p(v_y.apply(lambda x : 0 if x < 0 else x)))
#     model = lgb.train(params, lgtrain, 5000, valid_sets=[lgvalid], early_stopping_rounds=100, verbose_eval=100)
#     pred_test_y = model.predict(test_x, num_iteration=model.best_iteration)
#     return pred_test_y, model
    
# def param_tuning(init_points,num_iter,**args):
#     lgbBO = BayesianOptimization(lgb_eval, {'num_leaves': (25, 65),
#                                                 'max_depth': (5, 15),
#                                                 'lambda_l2': (0.0, 0.5),
#                                                 'lambda_l1': (0.0, 0.5),
#                                                 'bagging_fraction': (0.1, 0.99),
#                                                 'feature_fraction': (0.1, 0.99),
#                                                 'min_child_samples': (20, 50),
#                                                 })

#     lgbBO.maximize(init_points=init_points, n_iter=num_iter,**args)
#     return lgbBO

# result = param_tuning(5,15)
# result.res['max']['max_params']


# In[ ]:


params={'learning_rate': 0.03,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.9,
        "random_state":42,
        'max_depth': 4,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "bagging_frequency" : 5,
        'lambda_l2': 0.5,
        'lambda_l1': 0.5,
        'min_child_samples': 36
       }
xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

cat_param = {
    'learning_rate' :0.03,
    'depth' :10,
    'eval_metric' :'RMSE',
    'od_type' :'Iter',
    'metric_period ' : 50,
    'od_wait' : 20,
    'seed' : 42
    
}


# ### 8.Model Training with Kfold Validation LightGBM

# In[ ]:


folds = get_folds(df=train, n_splits=5)

train_features = [_f for _f in train.columns if _f not in excluded_features]
print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=1000,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100,
        eval_metric='rmse'
    )
    imp_df = pd.DataFrame()
    imp_df['feature'] = train_features
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    _preds = reg.predict(test[train_features], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_reg_preds += np.expm1(_preds) / len(folds)
    

print("RMSE: ", mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5)


# ### 9.Display feature importances

# In[ ]:


import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))


# ### 10.Create user level predictions

# In[ ]:


train['predictions'] = np.expm1(oof_reg_preds)
test['predictions'] = sub_reg_preds


# In[ ]:


# Aggregate data at User level
trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()


# In[ ]:


get_ipython().run_cell_magic('time', '', "# Create a list of predictions for each Visitor\ntrn_pred_list = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\\\n    .apply(lambda df: list(df.predictions))\\\n    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})")


# In[ ]:


# Create a DataFrame with VisitorId as index
# trn_pred_list contains dict 
# so creating a dataframe from it will expand dict values into columns
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns
trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions
gc.collect()
full_data.shape


# In[ ]:


get_ipython().run_cell_magic('time', '', "sub_pred_list = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\\\n    .apply(lambda df: list(df.predictions))\\\n    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})")


# In[ ]:


sub_data = test[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)
for f in trn_feats:
    if f not in sub_all_predictions.columns:
        sub_all_predictions[f] = np.nan
sub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))
sub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))
sub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)
sub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))
sub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)
sub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)
del sub_data, sub_all_predictions
gc.collect()
sub_full_data.shape


# ### 11.Create target and Cross Validation

# In[ ]:


train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()


# In[ ]:


params={'learning_rate': 0.03,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        'bagging_fraction': 0.9,
        'feature_fraction': 0.9,
        "random_state":42,
        'max_depth': 4,
        "bagging_seed" : 42,
        "verbosity" : -1,
        "bagging_frequency" : 5,
        'lambda_l2': 0.5,
        'lambda_l1': 0.5,
        'min_child_samples': 36
       }
xgb_params = {
        'objective': 'reg:linear',
        'booster': 'gbtree',
        'learning_rate': 0.02,
        'max_depth': 22,
        'min_child_weight': 57,
        'gamma' : 1.45,
        'alpha': 0.0,
        'lambda': 0.0,
        'subsample': 0.67,
        'colsample_bytree': 0.054,
        'colsample_bylevel': 0.50,
        'n_jobs': -1,
        'random_state': 456
    }

cat_param = {
    'learning_rate' :0.03,
    'depth' :10,
    'eval_metric' :'RMSE',
    'od_type' :'Iter',
    'metric_period ' : 50,
    'od_wait' : 20,
    'seed' : 42
    
}


# ### 12.Train a model at Visitor level

# In[ ]:


folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

oof_reg_preds = np.zeros(full_data.shape[0])
oof_reg_preds1 = np.zeros(full_data.shape[0])
oof_reg_preds2 = np.zeros(full_data.shape[0])
merge_pred = np.zeros(full_data.shape[0])
sub_preds = np.zeros(sub_full_data.shape[0])
vis_importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = full_data.iloc[trn_], trn_user_target['target'].iloc[trn_]
    val_x, val_y = full_data.iloc[val_], trn_user_target['target'].iloc[val_]
    
    reg = lgb.LGBMRegressor(**params,n_estimators=1100)
    xgb = XGBRegressor(**xgb_params, n_estimators=1000)
    cat = CatBoostRegressor(iterations=1000,learning_rate=0.03,
                            depth=10,
                            eval_metric='RMSE',
                            random_seed = 42,
                            bagging_temperature = 0.2,
                            od_type='Iter',
                            metric_period = 50,
                            od_wait=20)
    print("-"* 20 + "LightGBM Training" + "-"* 20)
    reg.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,verbose=100,eval_metric='rmse')
    print("-"* 20 + "XGboost Training" + "-"* 20)
    xgb.fit(trn_x, np.log1p(trn_y),eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,eval_metric='rmse',verbose=100)
    print("-"* 20 + "Catboost Training" + "-"* 20)
    cat.fit(trn_x, np.log1p(trn_y), eval_set=[(val_x, np.log1p(val_y))],early_stopping_rounds=50,use_best_model=True,verbose=100)
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = trn_x.columns
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)
    
    # LightGBM
    oof_reg_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_reg_preds[oof_reg_preds < 0] = 0
    lgb_preds = reg.predict(sub_full_data[full_data.columns], num_iteration=reg.best_iteration_)
    lgb_preds[lgb_preds < 0] = 0
    
    
    # Xgboost
    oof_reg_preds1[val_] = xgb.predict(val_x)
    oof_reg_preds1[oof_reg_preds1 < 0] = 0
    xgb_preds = xgb.predict(sub_full_data[full_data.columns])
    xgb_preds[xgb_preds < 0] = 0
    
    # catboost
    oof_reg_preds2[val_] = cat.predict(val_x)
    oof_reg_preds1[oof_reg_preds2 < 0] = 0
    cat_preds = cat.predict(sub_full_data[full_data.columns])
    cat_preds[xgb_preds < 0] = 0
        
    #merge all prediction
    merge_pred[val_] = oof_reg_preds[val_] * 0.6 + oof_reg_preds1[val_] * 0.3 + oof_reg_preds2[val_] * 0.1
    
    sub_preds += (lgb_preds / len(folds)) * 0.6 + (xgb_preds / len(folds)) * 0.3 + (cat_preds / len(folds)) * 0.1
    
print("LGBM Result ", mean_squared_error(np.log1p(trn_user_target['target']), oof_reg_preds) ** .5)
print("XGBoost Result", mean_squared_error(np.log1p(trn_user_target['target']), oof_reg_preds1) ** .5)
print("CatBoost Result", mean_squared_error(np.log1p(trn_user_target['target']), oof_reg_preds2) ** .5)
print("Combine  ", mean_squared_error(np.log1p(trn_user_target['target']), merge_pred) ** .5)


# ### 13.Display feature importances

# In[ ]:


vis_importances['gain_log'] = np.log1p(vis_importances['gain'])
mean_gain = vis_importances[['gain', 'feature']].groupby('feature').mean()
vis_importances['mean_gain'] = vis_importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 25))
sns.barplot(x='gain_log', y='feature', data=vis_importances.sort_values('mean_gain', ascending=False).iloc[:300])


# ### 14.Save Result

# In[ ]:


sub_full_data['PredictedLogRevenue'] = sub_preds
sub_full_data[['PredictedLogRevenue']].to_csv('new_test.csv', index=True)

