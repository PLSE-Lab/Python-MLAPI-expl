#!/usr/bin/env python
# coding: utf-8

# Use dataleak + categorical features encoded with frequency encoding, mean encoding and OHE, lightgbm on session level, lightgbm+xgb on visitior level
# I use https://www.kaggle.com/ogrellier/i-have-seen-the-future, https://www.kaggle.com/satian/story-of-a-leak

# In[ ]:


import pandas as pd
from tqdm import tqdm_notebook
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import gc
from xgboost import XGBRegressor

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

def make_mapping(data_df):
    data_df['device.browser'] = data_df['device.browser'].map(lambda x:browser_mapping(str(x).lower())).astype('str')
    data_df['trafficSource.adContent'] = data_df['trafficSource.adContent'].map(lambda x:adcontents_mapping(str(x).lower())).astype('str')
    data_df['trafficSource.source'] = data_df['trafficSource.source'].map(lambda x:source_mapping(str(x).lower())).astype('str')
    return data_df


def process_device(data_df):
    print("process device ...")
    data_df['source.country'] = data_df['trafficSource.source'] + '_' + data_df['geoNetwork.country']
    data_df['campaign.medium'] = data_df['trafficSource.campaign'] + '_' + data_df['trafficSource.medium']
    data_df['browser.category'] = data_df['device.browser'] + '_' + data_df['device.deviceCategory']
    data_df['browser.os'] = data_df['device.browser'] + '_' + data_df['device.operatingSystem']
    return data_df


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

def process(data):
    return custom(process_device(make_mapping(data)))
    


# In[ ]:


target_key = 'totals.transactionRevenue'


# In[ ]:


train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape


# In[ ]:


train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0)


# In[ ]:


#Loading external data
train_store_1 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data.csv',
                            low_memory=False, skiprows=6, dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data_2.csv',
                            low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_1 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data.csv',
                           low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_2 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data_2.csv',
                           low_memory=False, skiprows=6, dtype={"Client Id":'str'})


# In[ ]:


for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(str)

train_exdata = pd.concat([train_store_1, train_store_2], sort=False)
test_exdata = pd.concat([test_store_1, test_store_2], sort=False)


# In[ ]:


for df in [train, test, train_exdata, test_exdata]:
    df["visitId"] = df["visitId"].astype(float).astype(int)

# Merge with train/test data
train_new = train.merge(train_exdata, how="left", on="visitId")
test_new = test.merge(test_exdata, how="left", on="visitId")

# Drop Client Id
for df in [train_new, test_new]:
    df.drop("Client Id", 1, inplace=True)
    
#Cleaning Revenue
for df in [train_new, test_new]:
    df["Revenue"].fillna('$', inplace=True)
    df["Revenue"] = df["Revenue"].apply(lambda x: x.replace('$', '').replace(',', ''))
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Revenue"].fillna(0.0, inplace=True)
    df["Revenue"] = np.log1p(df["Revenue"])
    df['is_high_hits'] = np.logical_or(df["totals.hits"]>4,df["totals.pageviews"]>4).astype(np.int32)
    df['views/hits']=df["totals.pageviews"]/df["totals.hits"].dropna(0)


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = process(train_new)\ntest = process(test_new)\ndel train_new, test_new')


# In[ ]:


excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]


# In[ ]:


def freq_encoding(column):
    counter = column.value_counts()
    counter /= counter.sum()
    return counter


# In[ ]:


def mean_encoding(column, target, alpha = 0.01):
    global_mean = target.mean()
    gb = target.groupby(column)
    mean = gb.mean()
    count = gb.count()
    mean_adj = (mean * count + global_mean * alpha) / (count + alpha)
    return mean_adj


# In[ ]:


from sklearn.model_selection import GroupKFold
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


def get_subfolds(train, fold_idx, n_subfolds):
    subfolds = get_folds(df=train.loc[fold_trn_idx], n_splits=n_subfolds)
    for i in range(len(subfolds)):
        for j in range(len(subfolds[i])):
            subfolds[i][j] = fold_trn_idx[subfolds[i][j]]
    return subfolds


# In[ ]:


n_folds = 5
n_subfolds = 3

subfolds = []

folds = get_folds(df=train, n_splits=n_folds)
for fold_trn_idx, fold_val_idx in tqdm_notebook(folds):
    subfolds_cur = get_subfolds(train, fold_val_idx, n_subfolds)
    subfolds.append(subfolds_cur)
    '''for subfold_train_idx, subfold_val_idx in subfolds_cur:
        for cat_name in categorical_features:
            target = train.loc[subfold_train_idx, target_key]
            column = train.loc[subfold_train_idx, cat_name].fillna(0)
            mapping = mean_encoding(column, target)
            column_to = train.loc[subfold_val_idx, cat_name].fillna(0)
            train.loc[subfold_val_idx, 'mean_' + cat_name] = mapping[column_to].values'''
            


# In[ ]:


'''for cat_name in tqdm_notebook(categorical_features, leave=False):            
    target = train[target_key]
    column = train[cat_name].fillna(0)
    mapping = mean_encoding(column, target)
    column_to = test[cat_name].fillna(0)
    test['mean_' + cat_name] = mapping[column_to].values'''


# In[ ]:


for column in tqdm_notebook(categorical_features):
    train_col = train[column].fillna(0)
    test_col = test[column].fillna(0)
    mapping =  freq_encoding(pd.concat([train_col,test_col]))
    train['freq_' + column] = mapping[train_col].values
    test['freq_' + column] = mapping[test_col].values


# In[ ]:


for f in tqdm_notebook(categorical_features):
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])


# In[ ]:


y_reg = train[target_key]


# In[ ]:


train['target'] = y_reg
for df in [train, test]:
    df['vis_date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['vis_date'].dt.dayofweek
    df['sess_date_hours'] = df['vis_date'].dt.hour
    df['sess_date_dom'] = df['vis_date'].dt.day
    df.sort_values(['fullVisitorId', 'vis_date'], ascending=True, inplace=True)
    df['next_session_1'] = (
        df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(1)
    ).astype(np.int64) // 1e9 // 60 // 60
    df['next_session_2'] = (
        df['vis_date'] - df[['fullVisitorId', 'vis_date']].groupby('fullVisitorId')['vis_date'].shift(-1)
    ).astype(np.int64) // 1e9 // 60 // 60
    
    df['nb_pageviews'] = df['date'].map(
        df[['date', 'totals.pageviews']].groupby('date')['totals.pageviews'].sum()
    )
        
y_reg = train['target']
del train['target']


# In[ ]:


del train['vis_date']


# In[ ]:


train_features = [_f for _f in train.columns if _f not in excluded_features]

importances = pd.DataFrame()
oof_reg_preds = np.zeros(train.shape[0])
sub_reg_preds = np.zeros(test.shape[0])
for fold_, (trn_, val_) in enumerate(tqdm_notebook(folds)):
    trn_x, trn_y = train[train_features].iloc[trn_], y_reg.iloc[trn_]
    val_x, val_y = train[train_features].iloc[val_], y_reg.iloc[val_]
    
    reg = lgb.LGBMRegressor(
        num_leaves=31,
        learning_rate=0.03,
        n_estimators=600,
        subsample=.9,
        colsample_bytree=.9,
        random_state=1
    )
    reg.fit(
        trn_x, np.log1p(trn_y),
        eval_set=[(val_x, np.log1p(val_y))],
        early_stopping_rounds=50,
        verbose=100
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


# In[ ]:


mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5


# In[ ]:


train['predictions'] = np.expm1(oof_reg_preds)
test['predictions'] = sub_reg_preds
trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()

trn_pred_list = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')    .apply(lambda df: list(df.predictions))    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})

trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns
trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
trn_all_predictions['t_std'] = np.log1p(trn_all_predictions[trn_feats].std(axis=1))
trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
del trn_data, trn_all_predictions
gc.collect()


# In[ ]:


get_ipython().run_cell_magic('time', '', "sub_pred_list = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')\\\n    .apply(lambda df: list(df.predictions))\\\n    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})\nsub_data = test[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()\ndel test\nsub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)\n\nfor f in trn_feats:\n    if f not in sub_all_predictions.columns:\n        sub_all_predictions[f] = np.nan\nsub_all_predictions['t_mean'] = np.log1p(sub_all_predictions[trn_feats].mean(axis=1))\nsub_all_predictions['t_std'] = np.log1p(sub_all_predictions[trn_feats].std(axis=1))\nsub_all_predictions['t_median'] = np.log1p(sub_all_predictions[trn_feats].median(axis=1))\nsub_all_predictions['t_sum_log'] = np.log1p(sub_all_predictions[trn_feats]).sum(axis=1)\nsub_all_predictions['t_sum_act'] = np.log1p(sub_all_predictions[trn_feats].fillna(0).sum(axis=1))\nsub_all_predictions['t_nb_sess'] = sub_all_predictions[trn_feats].isnull().sum(axis=1)\nsub_full_data = pd.concat([sub_data, sub_all_predictions], axis=1)\ndel sub_data, sub_all_predictions\ngc.collect()\nsub_full_data.shape")


# In[ ]:


train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()
del train
gc.collect()


# In[ ]:


len(full_data)


# In[ ]:


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


# In[ ]:


get_ipython().run_cell_magic('time', '', "#I don't use xgboost in kernel, because i have some problems with kernel commit\nfolds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=5)\n\noof_preds = np.zeros(full_data.shape[0])\noof_preds1 = np.zeros(full_data.shape[0])\nboth_oof = np.zeros(full_data.shape[0])\nsub_preds = np.zeros(sub_full_data.shape[0])\nvis_importances = pd.DataFrame()\n\nfor fold_, (trn_, val_) in enumerate(tqdm_notebook(folds)):\n    trn_x, trn_y = full_data.iloc[trn_], trn_user_target['target'].iloc[trn_]\n    val_x, val_y = full_data.iloc[val_], trn_user_target['target'].iloc[val_]\n    \n    #xg = XGBRegressor(**xgb_params, n_estimators=1000)\n    \n    reg = lgb.LGBMRegressor(\n        num_leaves=31,\n        learning_rate=0.03,\n        n_estimators=100,\n        subsample=.9,\n        colsample_bytree=.9,\n        random_state=1\n    )\n    reg.fit(\n        trn_x, np.log1p(trn_y),\n        eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],\n        eval_names=['TRAIN', 'VALID'],\n        early_stopping_rounds=50,\n        eval_metric='rmse',\n        verbose=1000\n    )\n    #xg.fit(\n    #    trn_x, np.log1p(trn_y),\n    #    eval_set=[(trn_x, np.log1p(trn_y)), (val_x, np.log1p(val_y))],\n    #    early_stopping_rounds=50,\n    #    eval_metric='rmse',\n    #    verbose=100\n    #)\n    \n    imp_df = pd.DataFrame()\n    imp_df['feature'] = trn_x.columns\n    imp_df['gain'] = reg.booster_.feature_importance(importance_type='gain')\n\n    imp_df['fold'] = fold_ + 1\n    vis_importances = pd.concat([vis_importances, imp_df], axis=0, sort=False)\n\n    oof_preds[val_] = reg.predict(val_x, num_iteration=reg.best_iteration_)\n    #oof_preds1[val_] = xg.predict(val_x)\n\n    oof_preds[oof_preds < 0] = 0\n    #oof_preds1[oof_preds1 < 0] = 0\n\n    both_oof[val_] = oof_preds[val_]# * 0.6 + oof_preds1[val_] * 0.4\n\n    # Make sure features are in the same order\n    _preds = reg.predict(sub_full_data[full_data.columns], num_iteration=reg.best_iteration_)\n    _preds[_preds < 0] = 0\n\n    #pre = xg.predict(sub_full_data[full_data.columns])\n    #pre[pre<0]=0\n\n    sub_preds += (_preds / len(folds))# * 0.6 + (pre / len(folds)) * 0.4\n    \nmean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5")


# In[ ]:


mean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5


# In[ ]:


sub_full_data['PredictedLogRevenue'] = sub_preds

idx = list(set(sub_full_data.index).intersection(set(full_data.index)))
a = full_data.loc[idx]['Revenue']
diff = np.expm1(a[a>-10]) - trn_user_target['target'].loc[idx][a>-10] / 5e5
diff[diff < 0] = 0
sub_full_data['PredictedLogRevenue'].loc[diff.index] = np.log1p(diff * 5e5)

sub_full_data[['PredictedLogRevenue']].to_csv('new_test_leak_xgb.csv', index=True)


# In[ ]:




