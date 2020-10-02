#https://www.kaggle.com/ogrellier/teach-lightgbm-to-sum-predictions
#https://www.kaggle.com/mukesh62/lgb-fe-groupkfold-cv-xgb

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

train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape

#Define folding strategy

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

#Get session target
y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']
#Add date features
#Only add the one I think can ganeralize

train.columns


for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
    #df['sess_date_month'] = df['date'].dt.month
    df['sess_date_hours'] = df['date'].dt.hour
    df['sess_date_dom'] = df['date'].dt.day

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

#Create features list
excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]
#Factorize categoricals
for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])
    
params={'learning_rate': 0.03,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        "subsample": 0.99,
        "colsample_bytree": 0.99,
        "random_state":42,
        'max_depth': 15,
        'lambda_l2': 0.02085548700474218,
        'lambda_l1': 0.004107624022751344,
        'bagging_fraction': 0.7934712636944741,
        'feature_fraction': 0.686612409641711,
        'min_child_samples': 21
       }

#Predict revenues at session level
#Model Training with Kfold Validation LightGBM

folds = get_folds(df=train, n_splits=5)

train_features = [_f for _f in train.columns if _f not in excluded_features]
# print(train_features)

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(folds):
    print("Fold:",fold_)
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    reg = lgb.LGBMRegressor(**params,
        n_estimators = 1000
        #objective = 'regression', 
        #boosting_type = 'gbdt', 
        #metric = 'rmse',
        #n_estimators = 20000, #10000
        #num_leaves = 100, #10
        #learning_rate = 0.01, #0.01
        #bagging_fraction = 0.9,
        #feature_fraction = 0.9,
        #bagging_seed = 0,
        #max_depth = 10
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

#Display feature importances
import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))


#Create user level predictions
train['predictions'] = np.expm1(oof_reg_preds)
test['predictions'] = sub_reg_preds

# Aggregate data at User level
trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()

# Create a list of predictions for each Visitor
trn_pred_list = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})


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

sub_pred_list = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\
    .apply(lambda df: list(df.predictions))\
    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})

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

#Create target at Visitor level
train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()

#Train a model at Visitor level
#folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

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

fig, axes = plt.subplots(1, 1, figsize=(15, 15))
sc = axes.scatter(xtest,
                  ytest,
                  alpha=.1,
                  s=30)
_ = axes.set_title("Clustering test")

full_data['gp1'] = xtrain
full_data['gp2'] = ytrain
sub_full_data['gp1'] = xtest
sub_full_data['gp2'] = ytest

params={'learning_rate': 0.03,
        'objective':'regression',
        'metric':'rmse',
        'num_leaves': 31,
        'verbose': 1,
        "subsample": 0.99,
        "colsample_bytree": 0.99,
        "random_state":42,
        'max_depth': 15,
        'lambda_l2': 0.02085548700474218,
        'lambda_l1': 0.004107624022751344,
        'bagging_fraction': 0.7934712636944741,
        'feature_fraction': 0.686612409641711,
        'min_child_samples': 21
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
cat_params = {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 10,
        'eval_metric': 'RMSE',
        'random_seed': 42,
        'bagging_temperature': 0.2,
        'od_type': 'Iter',
        'metric_period': 50,
        'od_wait': 20
        }

#train a model at visitor level
from xgboost import XGBRegressor
from catboost import CatBoostRegressor

folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)

oof_preds = np.zeros(full_data.shape[0])
oof_preds1 = np.zeros(full_data.shape[0])
oof_preds2 = np.zeros(full_data.shape[0])

both_oof = np.zeros(full_data.shape[0])
sub_preds = np.zeros(sub_full_data.shape[0])
vis_importances = pd.DataFrame()

for fold_, (trn_, val_) in enumerate(folds):
    trn_x, trn_y = full_data.iloc[trn_], trn_user_target['target'].iloc[trn_]
    val_x, val_y = full_data.iloc[val_], trn_user_target['target'].iloc[val_]
    
    xg = XGBRegressor(**xgb_params, n_estimators=1000)
    
    reg = lgb.LGBMRegressor(**params, n_estimators=1500)
    
    cat = CatBoostRegressor(**cat_params)
    
    print('XGB' + "-" * 50)
    xg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbose=100
    )
    
    print('LGB' + "-" * 50)
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
        eval_names=['TRAIN', 'VALID'],
        early_stopping_rounds=50,
        eval_metric='rmse',
        verbose=100
    )
    print('CatBoost' + "-" * 50)
    cat.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],
        #eval_names=['TRAIN', 'VALID'],
        early_stopping_rounds=50,
        #eval_metric='rmse',
        use_best_model=True,
        verbose=True
    )
    
    imp_df = pd.DataFrame()
    imp_df['feature'] = trn_x.columns
    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')
    
    imp_df['fold'] = fold_ + 1
    vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)
    
    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)
    oof_preds1[val_] = xg.predict(val_x)
    oof_preds2[val_] = cat.predict(val_x)
    
    oof_preds[oof_preds < 0] = 0
    oof_preds1[oof_preds1 < 0] = 0
    oof_preds2[oof_preds2 < 0] = 0
    
    both_oof[val_] = oof_preds[val_] * 0.7 + oof_preds1[val_] * 0.25 + oof_preds2[val_] * 0.05
    
    # Make sure features are in the same order
    _preds = reg.predict(sub_full_data[full_data.columns], num_iteration=reg.best_iteration_)
    _preds[_preds < 0] = 0
    
    pre = xg.predict(sub_full_data[full_data.columns])
    pre[pre<0]=0
    
    _pre = cat.predict(sub_full_data[full_data.columns])
    _pre[_pre<0]=0
    
    sub_preds += (_preds / len(folds)) * 0.7 + (pre / len(folds)) * 0.25 + (_pre / len(folds)) *0.05
    
print("LGB  ", mean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5)
print("XGB  ", mean_squared_error(np.log1p(trn_user_target['target']), oof_preds1) ** .5)
print("CAT  ", mean_squared_error(np.log1p(trn_user_target['target']), oof_preds2) ** .5)
print("Combine  ", mean_squared_error(np.log1p(trn_user_target['target']), both_oof) ** .5)

vis_importances['gain_log'] = np.log1p(vis_importances['gain'])
mean_gain = vis_importances[['gain', 'feature']].groupby('feature').mean()
vis_importances['mean_gain'] = vis_importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 25))
sns.barplot(x='gain_log', y='feature', data=vis_importances.sort_values('mean_gain', ascending=False).iloc[:300])

sub_full_data['PredictedLogRevenue'] = sub_preds
sub_full_data[['PredictedLogRevenue']].to_csv('LGB_XGB_CAT.csv', index=True)