#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# You can refer the data analysis with this kernel : 
# https://www.kaggle.com/super13579/basic-feature-analysis-date-categorical-revenue
# * Keep try different feature to find the key feature on training
# * Use K cross validate to find the best score
# * After find the best score , try to do ensemble learning
# 

# ## Feature process in this kernel
# * Process Date feature (add day, week, hour, "revisit time")
# * Totals feature don't do anything, only fillna(0)
# * Do one hot encoding for categorical feature that have unique value <15
# * Do ranking encoding for categorical feature that have unique value >15

# ## Model used in this kernel
# LightGBM with K-Cross validation, K=5

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
from sklearn.model_selection import GroupKFold

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Get the extracted data
# * preprocessed dataset by olivier https://www.kaggle.com/ogrellier/create-extracted-json-fields-dataset

# In[ ]:


train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, "visitId":str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, "visitId":str, 'sessionId':str}, nrows=None)

train.shape, test.shape


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


# In[ ]:


y_reg = train['totals.transactionRevenue'].fillna(0)


# ## Add Date feature
# * add revisit features
# * add data extract feature

# In[ ]:


train['target'] = y_reg
def extract_new_feature(df): 
    print("Start extract date...")
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['day_of_week'] = df['date'].dt.dayofweek
    df['hour'] = df['date'].dt.hour
    df['day'] = df['date'].dt.day
    df['month'] = df['date'].dt.month
    print("Finished extract date...")
extract_new_feature(train)
extract_new_feature(test)

def add_time_period_of_same_ID(df): 
    print("Start add time period feature...")
    df.sort_values(['fullVisitorId', 'date'], ascending=True, inplace=True)
    df['next_revisit_time'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['prev_revisit_time'] = (
        df['date'] - df[['fullVisitorId', 'date']].groupby('fullVisitorId')['date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60
    print("Finished time periodfeature...")
    
add_time_period_of_same_ID(train)
add_time_period_of_same_ID(test)
y_reg = train['target']
del train['target']


# ## Find categorical features

# In[ ]:


categorical_features_train = train.select_dtypes(include=[np.object])
categorical_features_test = test.select_dtypes(include=[np.object])
categorical_features_train.columns


# ## Process totals features
# * fill nan feature to 0

# In[ ]:


train['totals.pageviews']=train['totals.pageviews'].astype('float64')
train['totals.hits']=train['totals.hits'].astype('float64')
test['totals.pageviews']=test['totals.pageviews'].astype('float64')
test['totals.hits']=test['totals.hits'].astype('float64')


# ## Do one hot encoding for categorical unique count <10

# In[ ]:


df_combine=pd.concat([train,test],ignore_index=True)
print(df_combine.shape)
#Find One_hot features that unique count <15
one_hot_features = df_combine[list(categorical_features_test)].nunique().reset_index()
one_hot_features.columns = ['features','unique_count']
one_hot_features = one_hot_features.loc[one_hot_features['unique_count'] < 10,"features"]
one_hot_features = list(one_hot_features)


# In[ ]:


#Process one_hot_features
for i in one_hot_features:
    print("Process feature =====>"+str(i))
    df_combine["one_hot_feature"] = df_combine[i]
    df_combine["one_hot_feature"] =  str(i) + "." + df_combine["one_hot_feature"].astype('str')
    one_hot_combine = pd.get_dummies(df_combine["one_hot_feature"])
    print(one_hot_combine.shape)
    df_combine = df_combine.join(one_hot_combine)
    del df_combine["one_hot_feature"]
    del df_combine[i]
    del one_hot_combine
    print(df_combine.shape)


# In[ ]:


train = df_combine[:len(train)]
print(train.shape)
test = df_combine[len(train):]
print(test.shape)
del df_combine


# ## Do ranking encoding  for categorical unique count >10
# * Idea comes from rahal's kernel : https://www.kaggle.com/rahullalu/gstore-eda-lgbm-baseline-1-4260

# In[ ]:


excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime']
categorical_larger_15_feature = [i for i in categorical_features_train if i not in one_hot_features and i not in excluded_features]
print(categorical_larger_15_feature)


# * Cause referral Path and networkDomain have many unique count, let's bypass it 

# In[ ]:


not_do_ranking =['trafficSource.referralPath','geoNetwork.networkDomain']
ranking_feature = [i for i in categorical_larger_15_feature if i not in not_do_ranking]
print(ranking_feature)


# In[ ]:


def Ranking_process(df,df_test):
    for col in ranking_feature:
        if df[col].dtype=='object':
            print("Process Ranking of "+str(col)+" feature...")
            df[col].fillna('others',inplace=True)
            col_list=[col,'totals.transactionRevenue']
            df_gropby=df[col_list].fillna(0).groupby(col).mean().reset_index()
            df_gropby.columns = col_list
            df_gropby['rank']=df_gropby['totals.transactionRevenue'].rank(ascending=1)
            replace_dict={}
            final_dict={}
            for k,col_val in enumerate(df_gropby[col].values):
                replace_dict[col_val]=df_gropby.iloc[k,2]
            final_dict[col]=replace_dict
            df[col]=df[col].map(replace_dict)
            df_test[col]=df_test[col].map(replace_dict)
            #df.replace(final_dict,inplace=True)
            del df_gropby,replace_dict,final_dict
            gc.collect()
            print("Finished process Ranking of "+str(col)+" feature")


# In[ ]:


Ranking_process(train,test)


# ## Factorize other categoricals features

# In[ ]:


excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]
print(categorical_features)


# In[ ]:



for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])


# In[ ]:


train.shape, test.shape


# ## Training model with K cross validation (Light GBM)
# * reference this kernel : https://www.kaggle.com/ogrellier/i-have-seen-the-future

# In[ ]:


folds = get_folds(df=train, n_splits=5)

train_features = [_f for _f in train.columns if _f not in excluded_features]


# In[ ]:


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
    
mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5


# ### Display feature importances

# In[ ]:


import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))


# ### Create user level predictions

# In[ ]:


train['predictions'] = oof_reg_preds
test['predictions'] = np.log1p(sub_reg_preds)


# In[ ]:


test_result = test[['fullVisitorId','predictions']].groupby('fullVisitorId').sum().reset_index()
train_result = train[['fullVisitorId','predictions']].groupby('fullVisitorId').sum().reset_index()
test_result.columns = ['fullVisitorId','PredictedLogRevenue']
train_result.columns = ['fullVisitorId','PredictedLogRevenue']
test_result.to_csv('Ranking_onehot_test.csv',index = False)
train_result.to_csv('Ranking_onehot_train.csv',index = False)


# In[ ]:





# ### Train a model at Visitor level

# In[ ]:




