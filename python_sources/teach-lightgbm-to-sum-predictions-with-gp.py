#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# In this kernel I demonstrate how to create predictions at Session level and then use them at User level so that LighGBM can learn how to better sum individual session prediction. 
# 
# It is sort of mini stacker and to avoid leakage, we use GroupKFold strategy.
# 

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

# In[ ]:


train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape


# ### Define folding strategy

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


# ### Get session target

# In[ ]:


y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']


# ### Add date features
# 
# Only add the one I think can ganeralize

# In[ ]:


train.columns


# In[ ]:


for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day


# ### Create features list

# In[ ]:


excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]


# ### Factorize categoricals

# In[ ]:


for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])


# ### Predict revenues at session level

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


# ### Create target at Visitor level

# In[ ]:


train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()


# In[ ]:


df = pd.concat([full_data,sub_full_data],sort=False)
df = df.reset_index(drop=False)
for c in df.columns[1:]:
    if((df[c].min()>=0)&(df[c].max()>=10)):
        df[c] = np.log1p(df[c])
    elif((df[c].min()<0)&((df[c].max()-df[c].min())>=10)):
        df.loc[df[c]!=0,c] = np.sign(df.loc[df[c]!=0,c])*np.log(np.abs(df.loc[df[c]!=0,c]))
from sklearn.preprocessing import StandardScaler
for c in df.columns[1:]:
    ss = StandardScaler()
    df.loc[~np.isfinite(df[c]),c] = np.nan
    df.loc[~df[c].isnull(),c] = ss.fit_transform(df.loc[~df[c].isnull(),c].values.reshape(-1,1))
df.fillna(-99999,inplace=True)
gp_trn_users = df[:full_data.shape[0]].copy().set_index('fullVisitorId')
gp_trn_users['target'] = np.log1p(trn_user_target['target'].values)
gp_trn_users.target /= gp_trn_users.target.max()
gp_sub_users = df[full_data.shape[0]:].copy().set_index('fullVisitorId')
newcols =  [x.replace('.','_') for x in gp_trn_users.columns]
gp_trn_users.columns = newcols
newcols =  [x.replace('.','_') for x in gp_sub_users.columns]
gp_sub_users.columns = newcols


# In[ ]:


def GP1(data):
    return ((((data["t_mean"]) + (data["totals_hits"]))) +
            (np.maximum((((((data["pred_2"]) + (data["pred_2"]))/2.0))), ((-3.0)))) +
            (np.maximum(((data["t_mean"])), ((-1.0)))) +
            (np.minimum(((1.0)), (((4.81583929061889648))))) +
            (np.maximum(((-2.0)), ((data["t_sum_act"])))) +
            (((1.0) * (data["geoNetwork_metro"]))) +
            (np.maximum((((((data["pred_0"]) + ((((data["t_median"]) + (np.maximum(((np.maximum(((data["t_sum_log"])), ((data["pred_0"]))))), ((data["t_mean"])))))/2.0)))/2.0))), ((data["t_sum_log"])))) +
            (((data["t_sum_log"]) + (np.minimum(((data["t_mean"])), ((data["t_sum_log"])))))) +
            ((((((-1.0) - (-2.0))) + (data["t_sum_act"]))/2.0)) +
            ((((0.0) > ((((data["trafficSource_referralPath"]) < (data["trafficSource_referralPath"]))*1.)))*1.)) +
            (((data["t_mean"]) + (1.0))) +
            (((data["t_sum_log"]) + (data["t_median"]))) +
            (((data["t_sum_act"]) - (0.0))) +
            (np.maximum(((data["pred_3"])), ((((3.0) * (data["pred_0"])))))) +
            ((((data["pred_2"]) + (-2.0))/2.0)) +
            (((data["t_sum_log"]) * (data["t_mean"]))) +
            (((data["t_sum_act"]) + (data["t_sum_log"]))) +
            ((((((data["totals_hits"]) - (3.0))) > (data["geoNetwork_metro"]))*1.)) +
            (((3.0) * (data["t_sum_log"]))) +
            ((((data["geoNetwork_continent"]) > (-1.0))*1.)) +
            ((((data["geoNetwork_subContinent"]) < (((data["device_browser"]) + (-2.0))))*1.)) +
            (np.minimum(((np.minimum(((2.0)), ((data["geoNetwork_metro"]))))), ((data["geoNetwork_metro"])))) +
            (np.minimum(((1.0)), ((np.tanh((((2.0) * 2.0))))))) +
            ((((data["geoNetwork_subContinent"]) > (-3.0))*1.)) +
            (((data["t_sum_log"]) - (-1.0))) +
            (np.minimum(((-2.0)), ((data["trafficSource_referralPath"])))) +
            (np.maximum(((data["t_sum_log"])), ((-3.0)))) +
            (np.minimum(((np.where((((data["pred_0"]) > (data["trafficSource_referralPath"]))*1.)<0, data["totals_hits"], -3.0 ))), ((((data["geoNetwork_subContinent"]) / 2.0))))) +
            ((((2.0) + (data["pred_1"]))/2.0)) +
            (np.tanh((np.tanh((2.0))))) +
            (np.tanh(((-1.0*((-3.0)))))) +
            (((-2.0) + (data["t_sum_log"]))) +
            ((((data["geoNetwork_country"]) < (data["geoNetwork_metro"]))*1.)) +
            ((((3.0) + ((-1.0*((-2.0)))))/2.0)) +
            (((((((data["t_sum_log"]) * (data["geoNetwork_continent"]))) * (((data["t_sum_log"]) - (data["visitNumber"]))))) * (np.minimum(((data["t_sum_act"])), ((data["t_sum_log"])))))) +
            ((((((data["totals_hits"]) / 2.0)) + (data["t_sum_log"]))/2.0)) +
            (((((np.tanh((data["pred_1"]))) + (data["t_sum_act"]))) + (((data["t_sum_act"]) + (((data["t_sum_log"]) + (-2.0))))))) +
            (((2.0) - (np.where(data["t_mean"]>0, data["t_nb_sess"], (((data["t_mean"]) < (data["geoNetwork_continent"]))*1.) )))) +
            (np.where(data["t_mean"]>0, data["t_sum_log"], -2.0 )) +
            ((((data["t_mean"]) + ((((data["geoNetwork_subContinent"]) + (((np.tanh(((12.19428730010986328)))) * 2.0)))/2.0)))/2.0)) +
            (np.where(((((((data["pred_0"]) * 2.0)) + (data["visitNumber"]))) / 2.0) < -99998, (((-1.0) > (2.0))*1.), np.tanh((data["totals_hits"])) )) +
            (np.where((((-1.0) > (1.0))*1.)<0, ((((1.0)) + (-1.0))/2.0), data["t_sum_act"] )) +
            ((((-1.0*((((((((((data["t_median"]) + (((data["t_sum_log"]) + (data["t_nb_sess"]))))/2.0)) + (data["geoNetwork_continent"]))/2.0)) + (data["t_sum_log"])))))) / 2.0)) +
            (np.minimum(((np.minimum(((np.where(data["trafficSource_referralPath"]<0, (((data["trafficSource_referralPath"]) + (np.tanh((data["t_sum_log"]))))/2.0), -3.0 ))), ((data["geoNetwork_country"]))))), ((data["t_sum_log"])))) +
            ((((((data["t_sum_log"]) + (np.where(data["visitNumber"]<0, -2.0, data["pred_1"] )))/2.0)) + (data["t_sum_act"]))) +
            (((((np.where(data["geoNetwork_country"]>0, data["t_sum_act"], data["geoNetwork_country"] )) + (((data["geoNetwork_country"]) / 2.0)))) + (-1.0))) +
            (np.where(data["t_mean"]<0, -2.0, np.where(((data["geoNetwork_metro"]) + (data["geoNetwork_continent"]))<0, -2.0, ((data["geoNetwork_continent"]) + (data["geoNetwork_continent"])) ) )) +
            ((((((data["geoNetwork_continent"]) < (data["t_sum_act"]))*1.)) + (((np.minimum(((((((data["totals_hits"]) / 2.0)) / 2.0))), ((data["pred_0"])))) + (data["t_sum_log"]))))) +
            (((data["t_sum_act"]) + (((-2.0) - (np.minimum(((data["geoNetwork_metro"])), ((((data["geoNetwork_metro"]) * (np.minimum(((data["t_sum_act"])), ((data["t_mean"]))))))))))))) +
            (np.where(-3.0<0, ((np.where(2.0 < -99998, data["t_sum_log"], (((0.0) < (data["totals_pageviews"]))*1.) )) * 2.0), (6.0) )) +
            ((((data["t_sum_log"]) + ((((((-1.0) - (((data["visitNumber"]) + (data["t_mean"]))))) + (np.tanh((data["t_sum_log"]))))/2.0)))/2.0)) +
            ((((((4.0)) * (np.where((((data["totals_pageviews"]) > (data["totals_hits"]))*1.)>0, data["t_sum_act"], -3.0 )))) * 2.0)) +
            (np.minimum(((((data["t_sum_log"]) + ((-1.0*((data["t_nb_sess"]))))))), ((np.where(np.tanh((data["t_mean"]))>0, (-1.0*((data["t_nb_sess"]))), -3.0 ))))) +
            (((((((-3.0) + ((((11.04013633728027344)) * (data["t_sum_log"]))))) + ((((-3.0) + (data["t_sum_log"]))/2.0)))) * 2.0)) +
            (np.minimum(((np.tanh((np.tanh((data["totals_hits"])))))), ((data["pred_1"])))) +
            ((((np.minimum(((np.tanh((((((data["pred_0"]) * 2.0)) / 2.0))))), (((4.91354322433471680))))) + (-2.0))/2.0)) +
            (((-3.0) + (((data["t_sum_act"]) * ((((13.85095119476318359)) - (-2.0))))))) +
            (((((((((data["t_sum_log"]) * 2.0)) * 2.0)) * 2.0)) * 2.0)) +
            ((((((np.maximum(((np.minimum(((data["t_median"])), ((data["geoNetwork_metro"]))))), (((((data["t_mean"]) > (data["t_sum_log"]))*1.))))) + (-3.0))/2.0)) + (data["t_mean"]))) +
            (np.minimum(((((1.0) * 2.0))), (((((-2.0) + (((0.0) + (data["trafficSource_referralPath"]))))/2.0))))) +
            (((np.minimum(((data["t_sum_log"])), ((0.0)))) + (np.minimum((((((data["t_mean"]) + (data["totals_pageviews"]))/2.0))), ((data["t_mean"])))))) +
            ((((3.0) > (data["t_sum_act"]))*1.)) +
            ((((data["trafficSource_referralPath"]) < (3.0))*1.)) +
            (((0.0) + (-1.0))) +
            (np.minimum(((data["geoNetwork_metro"])), (((13.91542816162109375))))) +
            (((data["t_sum_act"]) + (((-3.0) + (((data["t_nb_sess"]) + (((data["t_sum_act"]) + (data["t_sum_act"]))))))))) +
            ((-1.0*(((((data["t_nb_sess"]) + (((((data["pred_1"]) * (((((((data["t_mean"]) > (data["t_sum_log"]))*1.)) < (-1.0))*1.)))) * 2.0)))/2.0))))) +
            (((data["t_median"]) + (((((((data["t_median"]) + (((data["totals_pageviews"]) + (-1.0))))) - (data["trafficSource_referralPath"]))) + (-1.0))))) +
            (((((0.0) + (((((((data["t_mean"]) < (data["trafficSource_referralPath"]))*1.)) > (1.0))*1.)))) * (-3.0))) +
            (((data["totals_pageviews"]) - ((((data["visitNumber"]) + (((((data["totals_hits"]) - ((-1.0*((data["t_nb_sess"])))))) * 2.0)))/2.0)))) +
            (((((((data["geoNetwork_metro"]) - (((((10.93749332427978516)) > (((data["t_sum_log"]) * 2.0)))*1.)))) - (np.tanh((data["geoNetwork_metro"]))))) * (data["geoNetwork_metro"]))) +
            (((data["t_nb_sess"]) * (data["channelGrouping"]))) +
            (((-2.0) + (0.0))) +
            ((((-1.0) > (1.0))*1.)) +
            (((((-3.0) + (0.0))) + (-3.0))) +
            (((((data["totals_hits"]) - (np.where(data["pred_1"] < -99998, 1.0, data["totals_pageviews"] )))) + (((data["totals_hits"]) - (data["totals_pageviews"]))))) +
            (((((data["t_mean"]) * ((((data["geoNetwork_continent"]) + ((-1.0*((data["geoNetwork_subContinent"])))))/2.0)))) * (2.0))) +
            (np.maximum(((data["t_sum_act"])), ((data["visitNumber"])))) +
            ((((data["totals_hits"]) < (3.0))*1.)) +
            (np.minimum((((((data["t_median"]) + (data["visitNumber"]))/2.0))), ((np.minimum(((data["t_sum_act"])), ((data["totals_hits"]))))))) +
            (np.maximum(((data["totals_hits"])), ((((data["geoNetwork_continent"]) / 2.0))))) +
            (np.maximum(((-3.0)), ((np.maximum(((data["geoNetwork_continent"])), ((data["t_nb_sess"]))))))) +
            ((((2.0) > (data["visitNumber"]))*1.)) +
            ((((((data["channelGrouping"]) + (data["geoNetwork_continent"]))) < ((((data["geoNetwork_continent"]) > (((data["t_nb_sess"]) - (data["channelGrouping"]))))*1.)))*1.)) +
            ((((-2.0) > (np.maximum(((((data["t_sum_log"]) - ((2.97587943077087402))))), (((((data["pred_0"]) + ((3.84451842308044434)))/2.0))))))*1.)) +
            (np.minimum(((np.minimum(((data["geoNetwork_country"])), ((((((((1.0)) - (data["geoNetwork_country"]))) + ((((1.0)) - (data["geoNetwork_subContinent"]))))/2.0)))))), ((data["t_sum_act"])))) +
            (((((((10.0)) * (data["t_nb_sess"]))) > (np.minimum((((3.0))), ((((data["geoNetwork_continent"]) * ((0.13566258549690247))))))))*1.)) +
            ((((data["geoNetwork_continent"]) > ((3.89537310600280762)))*1.)) +
            (np.minimum(((data["t_mean"])), ((2.0)))) +
            (np.minimum(((np.minimum(((-3.0)), ((((data["t_sum_log"]) - (data["t_median"]))))))), ((((data["t_sum_log"]) + (data["t_sum_act"])))))) +
            (((data["totals_pageviews"]) - (data["t_mean"]))) +
            (np.maximum(((1.0)), ((data["t_sum_act"])))) +
            (np.where(0.0<0, data["geoNetwork_metro"], data["visitNumber"] )) +
            (((data["visitNumber"]) * (data["geoNetwork_continent"]))) +
            ((((0.0) < (data["geoNetwork_subContinent"]))*1.)) +
            (np.minimum((((((-3.0) + ((((data["t_sum_log"]) + (data["geoNetwork_subContinent"]))/2.0)))/2.0))), ((((data["t_sum_act"]) - (np.minimum(((data["t_sum_log"])), ((data["t_mean"]))))))))) +
            (np.minimum(((3.0)), ((data["geoNetwork_metro"])))) +
            (((-3.0) * (((data["t_median"]) * (-1.0))))) +
            (((2.0) * ((((9.0)) * ((9.0)))))) +
            (np.minimum(((np.minimum(((data["t_median"])), ((data["t_sum_log"]))))), (((((((data["totals_hits"]) - (data["visitNumber"]))) + (((data["totals_hits"]) + (data["totals_hits"]))))/2.0))))))

def GP2(data):
    return ((((((np.tanh((data["totals_hits"]))) + (data["pred_1"]))) * 2.0)) +
            ((((8.08461284637451172)) - (data["t_sum_log"]))) +
            ((((np.tanh((data["t_sum_log"]))) + (((data["t_sum_log"]) + (data["t_mean"]))))/2.0)) +
            (((((data["t_sum_log"]) + (0.0))) + (data["t_sum_act"]))) +
            (np.minimum(((data["t_nb_sess"])), (((((-2.0) + (3.0))/2.0))))) +
            (((data["geoNetwork_continent"]) + (data["t_sum_log"]))) +
            (np.minimum(((data["t_sum_act"])), ((np.maximum((((5.0))), ((data["t_sum_act"]))))))) +
            (((data["totals_hits"]) * (((data["t_sum_act"]) - ((8.14568805694580078)))))) +
            (((data["t_sum_act"]) + (data["pred_3"]))) +
            ((-1.0*(((-1.0*((data["t_sum_act"]))))))) +
            (np.maximum(((((3.0) - (-2.0)))), ((3.0)))) +
            (np.maximum(((-2.0)), ((data["t_sum_log"])))) +
            (((((data["t_sum_act"]) * 2.0)) * ((1.0)))) +
            (((data["t_sum_log"]) + (data["t_sum_log"]))) +
            (((data["pred_1"]) - ((5.53632402420043945)))) +
            ((((data["t_sum_log"]) < (data["geoNetwork_metro"]))*1.)) +
            (((data["pred_1"]) * (1.0))) +
            (np.maximum(((data["totals_pageviews"])), (((-1.0*((np.tanh((0.0))))))))) +
            (((((((0.0)) * (-3.0))) > (-1.0))*1.)) +
            (np.maximum(((data["totals_pageviews"])), ((-2.0)))) +
            (((data["t_sum_log"]) * (data["t_sum_log"]))) +
            (np.minimum(((3.0)), ((data["t_sum_log"])))) +
            (((np.maximum(((data["visitNumber"])), ((3.0)))) - (data["t_mean"]))) +
            (((data["t_sum_log"]) - ((14.92182254791259766)))) +
            (np.maximum(((data["t_median"])), ((3.0)))) +
            ((((3.0) > (data["t_mean"]))*1.)) +
            (((((data["geoNetwork_subContinent"]) * 2.0)) * 2.0)) +
            (((np.where(1.0<0, np.tanh((data["pred_1"])), data["t_nb_sess"] )) * (((data["t_sum_act"]) * 2.0)))) +
            (((data["t_sum_act"]) * (data["t_sum_act"]))) +
            (((-3.0) + (data["t_sum_log"]))) +
            ((((9.0)) * (((-2.0) + (data["t_sum_act"]))))) +
            (((((((7.23600816726684570)) * (data["geoNetwork_continent"]))) > ((1.63156664371490479)))*1.)) +
            ((((data["t_sum_log"]) < (1.0))*1.)) +
            (((data["t_sum_act"]) * ((((((((7.0)) - (data["t_sum_act"]))) * (((((7.0)) > (data["t_sum_act"]))*1.)))) * (data["t_sum_act"]))))) +
            (((((((3.0) > (data["pred_0"]))*1.)) > (((np.maximum(((np.minimum(((2.0)), ((data["visitNumber"]))))), ((((data["t_median"]) * 2.0))))) / 2.0)))*1.)) +
            (((((2.0) * (data["geoNetwork_subContinent"]))) + (((((data["t_sum_act"]) + (((data["t_sum_act"]) - (data["pred_0"]))))) * (data["t_sum_act"]))))) +
            (np.minimum((((((data["t_sum_act"]) < (data["pred_1"]))*1.))), ((data["geoNetwork_metro"])))) +
            (((data["geoNetwork_continent"]) + (data["t_mean"]))) +
            (np.maximum((((((((data["totals_hits"]) * 2.0)) + (-2.0))/2.0))), ((((data["visitNumber"]) + (1.0)))))) +
            ((((((data["t_sum_act"]) + ((((data["t_mean"]) + (data["pred_1"]))/2.0)))/2.0)) + ((((((data["pred_1"]) + (data["t_sum_act"]))/2.0)) - (data["visitNumber"]))))) +
            (np.minimum((((((10.0)) - (data["t_sum_log"])))), ((((((((data["geoNetwork_continent"]) * 2.0)) * (data["t_sum_act"]))) * (data["geoNetwork_continent"])))))) +
            (np.minimum(((data["geoNetwork_subContinent"])), (((((((((data["geoNetwork_subContinent"]) + (-3.0))) - (2.0))) + (data["t_sum_log"]))/2.0))))) +
            (((((((data["t_sum_log"]) / 2.0)) + (data["t_nb_sess"]))) + (((((-3.0) + (data["t_sum_act"]))) + (data["t_sum_act"]))))) +
            ((((np.tanh((((((np.tanh((np.tanh((np.minimum(((data["geoNetwork_country"])), ((data["geoNetwork_country"])))))))) * 2.0)) * 2.0)))) + ((7.0)))/2.0)) +
            (np.where(np.minimum(((2.0)), (((((-1.0) > (-2.0))*1.))))>0, data["geoNetwork_subContinent"], (-1.0*(((5.0)))) )) +
            (((data["geoNetwork_metro"]) + (np.where(data["geoNetwork_metro"]>0, data["geoNetwork_metro"], data["t_sum_act"] )))) +
            ((((((6.0)) / 2.0)) - ((((((-3.0) + (data["visitNumber"]))) < (np.tanh((data["totals_pageviews"]))))*1.)))) +
            (((data["t_nb_sess"]) + (((data["t_sum_log"]) + (((np.where(data["t_sum_log"] < -99998, data["t_mean"], data["t_sum_log"] )) - (3.0))))))) +
            ((((data["t_median"]) + (((np.maximum((((((data["geoNetwork_metro"]) + (-1.0))/2.0))), ((-2.0)))) * (data["t_sum_act"]))))/2.0)) +
            ((((((data["t_median"]) * (data["t_sum_act"]))) + (np.tanh((((np.minimum(((-1.0)), ((2.0)))) + (data["t_sum_act"]))))))/2.0)) +
            (np.maximum(((data["visitNumber"])), ((np.minimum(((data["totals_hits"])), ((data["t_mean"]))))))) +
            (np.where(data["t_median"]<0, ((data["t_median"]) * 2.0), ((data["t_sum_act"]) * 2.0) )) +
            ((((((np.minimum(((np.minimum(((3.0)), ((data["totals_pageviews"]))))), ((data["t_mean"])))) + (data["totals_pageviews"]))/2.0)) * 2.0)) +
            (np.maximum(((data["totals_hits"])), ((np.maximum((((((((data["totals_pageviews"]) + (data["t_mean"]))) + (data["t_sum_log"]))/2.0))), ((data["t_sum_act"]))))))) +
            (((((np.where(data["t_sum_log"]<0, data["pred_1"], data["geoNetwork_metro"] )) + (np.tanh((np.tanh((data["t_sum_log"]))))))) * 2.0)) +
            (((np.minimum(((((data["geoNetwork_metro"]) - (data["channelGrouping"])))), (((((13.51332950592041016)) * (data["t_sum_log"])))))) * 2.0)) +
            ((((-1.0*((data["geoNetwork_country"])))) / 2.0)) +
            ((((np.tanh((((data["pred_1"]) + ((((1.0) > (data["channelGrouping"]))*1.)))))) > ((((data["t_sum_act"]) + (data["t_nb_sess"]))/2.0)))*1.)) +
            (((data["t_sum_log"]) + (((data["t_sum_log"]) + (((data["t_sum_log"]) + (((data["t_sum_act"]) + (data["t_sum_act"]))))))))) +
            (((((data["totals_pageviews"]) + (((data["totals_pageviews"]) - (3.0))))) + (((data["t_sum_log"]) - (data["t_sum_act"]))))) +
            (((data["trafficSource_referralPath"]) - (np.maximum((((2.04302835464477539))), ((0.0)))))) +
            ((((data["geoNetwork_continent"]) + (data["totals_pageviews"]))/2.0)) +
            (np.minimum((((12.72106933593750000))), ((((data["geoNetwork_continent"]) - (data["device_browser"])))))) +
            (((np.minimum(((((data["t_sum_log"]) * ((4.04873085021972656))))), ((((data["totals_pageviews"]) - (np.minimum(((data["totals_hits"])), ((data["totals_hits"]))))))))) * ((4.04873085021972656)))) +
            (((3.0) + (data["geoNetwork_metro"]))) +
            ((((-2.0) < (np.minimum((((4.73788738250732422))), ((data["t_median"])))))*1.)) +
            (np.minimum(((-1.0)), ((np.minimum(((0.0)), ((3.0))))))) +
            (((np.maximum((((0.0))), ((-3.0)))) * (((((((2.0) > (data["t_median"]))*1.)) > (0.0))*1.)))) +
            (np.minimum((((((data["geoNetwork_metro"]) > ((4.0)))*1.))), ((np.minimum(((data["t_mean"])), ((data["t_sum_log"]))))))) +
            (((np.minimum((((((6.11185455322265625)) * (((np.tanh((data["t_median"]))) - ((-1.0*((data["t_sum_act"]))))))))), ((data["t_sum_log"])))) + (data["totals_hits"]))) +
            ((((np.minimum(((data["t_nb_sess"])), ((-3.0)))) > (1.0))*1.)) +
            (((((((np.tanh(((-1.0*((data["pred_1"])))))) + (data["trafficSource_referralPath"]))/2.0)) < (0.0))*1.)) +
            (((data["t_sum_act"]) * ((((data["t_sum_act"]) < ((7.0)))*1.)))) +
            (((data["t_sum_log"]) + (np.minimum(((data["t_mean"])), ((((data["channelGrouping"]) * (data["t_mean"])))))))) +
            (np.maximum((((((data["totals_hits"]) > (0.0))*1.))), ((data["totals_pageviews"])))) +
            ((((data["geoNetwork_continent"]) > (((data["device_browser"]) + (((data["trafficSource_referralPath"]) * ((((data["pred_0"]) > (data["t_nb_sess"]))*1.)))))))*1.)) +
            (np.where(data["t_mean"]<0, data["totals_pageviews"], data["geoNetwork_metro"] )) +
            (((data["totals_pageviews"]) - (np.maximum(((-1.0)), ((data["totals_hits"])))))) +
            ((((data["t_mean"]) < (data["totals_pageviews"]))*1.)) +
            ((((1.0)) + ((((data["t_median"]) + (1.0))/2.0)))) +
            ((((np.minimum(((((data["pred_0"]) * (((np.minimum(((data["t_median"])), (((-1.0*((data["device_browser"]))))))) * 2.0))))), ((-2.0)))) + (data["totals_hits"]))/2.0)) +
            (np.minimum(((data["pred_0"])), ((data["t_median"])))) +
            (np.minimum(((((data["t_sum_act"]) * (np.minimum(((((data["geoNetwork_subContinent"]) - (data["geoNetwork_country"])))), ((data["geoNetwork_subContinent"]))))))), ((((data["geoNetwork_country"]) - (data["trafficSource_referralPath"])))))) +
            (((3.0) - (data["geoNetwork_metro"]))) +
            (((np.maximum(((((data["pred_0"]) * (data["pred_0"])))), ((data["channelGrouping"])))) * (((data["t_nb_sess"]) * (data["t_nb_sess"]))))) +
            ((((-2.0) > ((((0.0) < (data["t_median"]))*1.)))*1.)) +
            (np.maximum(((np.maximum(((data["trafficSource_referralPath"])), ((np.maximum(((((data["geoNetwork_subContinent"]) + (((data["t_nb_sess"]) + (data["geoNetwork_continent"])))))), ((data["channelGrouping"])))))))), ((data["trafficSource_referralPath"])))) +
            (np.minimum(((np.maximum(((data["trafficSource_referralPath"])), ((data["t_nb_sess"]))))), (((((((data["t_sum_log"]) + (data["t_mean"]))/2.0)) + (data["t_sum_act"])))))) +
            ((((((data["totals_pageviews"]) + (data["geoNetwork_city"]))) < (2.0))*1.)) +
            (np.minimum(((data["t_sum_act"])), ((1.0)))) +
            (((-1.0) - (3.0))) +
            ((((((((((12.32176685333251953)) > ((((data["geoNetwork_metro"]) + (data["geoNetwork_metro"]))/2.0)))*1.)) + (data["channelGrouping"]))/2.0)) * (np.minimum(((data["t_mean"])), ((data["geoNetwork_metro"])))))) +
            (((np.minimum(((data["visitNumber"])), ((data["trafficSource_source"])))) * 2.0)) +
            (((1.0) * (data["device_browser"]))) +
            ((((((-1.0*((2.0)))) + (((-2.0) + (1.0))))) / 2.0)) +
            (((((((((((data["t_nb_sess"]) + ((6.0)))/2.0)) - (data["t_nb_sess"]))) + (data["geoNetwork_subContinent"]))/2.0)) + (data["t_nb_sess"]))) +
            ((((((data["t_sum_act"]) * (data["totals_pageviews"]))) < (data["pred_0"]))*1.)) +
            (((((((data["t_median"]) - (data["totals_hits"]))) + (data["geoNetwork_subContinent"]))) + (((data["geoNetwork_subContinent"]) + (((data["geoNetwork_subContinent"]) + (data["geoNetwork_country"]))))))) +
            (np.minimum(((data["geoNetwork_subContinent"])), ((data["geoNetwork_metro"])))) +
            ((((np.minimum(((data["t_median"])), ((data["t_sum_log"])))) + (data["totals_hits"]))/2.0)))


# In[ ]:


xtrain = GP1(gp_trn_users).values
ytrain = GP2(gp_trn_users).values
xtrain = np.sign(xtrain)*np.log1p(np.abs(xtrain))
ytrain = np.sign(ytrain)*np.log1p(np.abs(ytrain))
xtest = GP1(gp_sub_users).values
ytest = GP2(gp_sub_users).values
xtest = np.sign(xtest)*np.log1p(np.abs(xtest))
ytest = np.sign(ytest)*np.log1p(np.abs(ytest))
del gp_sub_users
del gp_trn_users
gc.collect()


# In[ ]:


cm = plt.cm.get_cmap('RdYlGn')
fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(xtrain,
                  ytrain,
                  alpha=.1,
                  c=np.log1p(trn_user_target.values.ravel()),
                  cmap=cm,
                  s=30)
cbar = fig.colorbar(sc, ax=axes)
cbar.set_label('Target')
_ = axes.set_title("Clustering colored by target")


# In[ ]:


fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(xtest,
                  ytest,
                  alpha=.1,
                  s=30)
_ = axes.set_title("Clustering test")


# In[ ]:


full_data['gp1'] = xtrain
full_data['gp2'] = ytrain
sub_full_data['gp1'] = xtest
sub_full_data['gp2'] = ytest


# ### Train a model at Visitor level

# In[ ]:


folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

oof_preds = np.zeros(full_data.shape[0])
sub_preds = np.zeros(sub_full_data.shape[0])
vis_importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = full_data.iloc[trn_], trn_user_target['target'].iloc[trn_]
    val_x, val_y = full_data.iloc[val_], trn_user_target['target'].iloc[val_]
    
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
        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
        eval_names=['TRAIN', 'VALID'],
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbose=100
    )
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = trn_x.columns
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)
    
    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_preds[oof_preds < 0] = 0
    
    # Make sure features are in the same order
    _preds = reg.predict(sub_full_data[full_data.columns], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    sub_preds += _preds / len(folds)
    
mean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5


# ### Display feature importances

# In[ ]:


vis_importances['gain_log'] = np.log1p(vis_importances['gain'])
mean_gain = vis_importances[['gain', 'feature']].groupby('feature').mean()
vis_importances['mean_gain'] = vis_importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 25))
sns.barplot(x='gain_log', y='feature', data=vis_importances.sort_values('mean_gain', ascending=False).iloc[:300])


# ### Save predictions

# In[ ]:


sub_full_data['PredictedLogRevenue'] = sub_preds
sub_full_data[['PredictedLogRevenue']].to_csv('new_test.csv', index=True)

