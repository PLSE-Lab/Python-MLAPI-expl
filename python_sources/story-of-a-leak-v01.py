#!/usr/bin/env python
# coding: utf-8

# ### I screwed up uploading the script as it is not visible to everyone, hence i am uploading it here and making the other one private.
# 

# In[ ]:


# As discussed in the discussion section, we have leak in the competition if we use external data.
# Discussed here: https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/68235#401950
 
#  I have downloaded the data and used it to climb to the 2nd position. Before using it I was at the 600th position. 
#  Most of the code used it from olivier's (https://www.kaggle.com/ogrellier) notebooks in this competition and I haven't done any feature engineering with the external data yet.
 
#  You can use the data that I downloaded as it would be in the data of this kernel. 
#  There are 4 files 2 for train and 2 for test.
#Can't upload the notebook as kaggle kernels keep on crashing but this script will get you same score as mine.

import warnings
warnings.simplefilter("ignore")
import os
import json
import pandas as pd
import numpy as np
from pandas.io.json import json_normalize
from IPython.display import display
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import gc
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
pd.set_option("display.max_rows", 1000, "display.max_columns", 1000)
pd.options.mode.chained_assignment = None

#using https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook
def load_df(csv_path='../input/ga-customer-revenue-prediction/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'}, # Important!!
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df
    
train = load_df()
test = load_df("../input/ga-customer-revenue-prediction/test.csv")

#Loading external data
train_store_1 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_1 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
test_store_2 = pd.read_csv('../input/exported-google-analytics-data/Test_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})

#Getting VisitId to Join with our train, test data
def get_visitid(id):
    bef_, af_ = str(x).split('.')
    return int(bef_), (int(af_)*10 if len(af_)==1 else int(af_))

for df in [train_store_1, train_store_2, test_store_1, test_store_2]:
    df["visitId"] = df["Client Id"].apply(lambda x: x.split('.', 1)[1]).astype(str)

train_exdata = pd.concat([train_store_1, train_store_2], sort=False)
test_exdata = pd.concat([test_store_1, test_store_2], sort=False)

for df in [train, test]:
    df["visitId"] = df["visitId"].astype(str)

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

#Imputing NaN
for df in [train_new, test_new]:
    df["Sessions"] = df["Sessions"].fillna(0)
    df["Avg. Session Duration"] = df["Avg. Session Duration"].fillna(0)
    df["Bounce Rate"] = df["Bounce Rate"].fillna(0)
    df["Revenue"] = df["Revenue"].fillna(0)
    df["Transactions"] = df["Transactions"].fillna(0)
    df["Goal Conversion Rate"] = df["Goal Conversion Rate"].fillna(0)
    df['trafficSource.adContent'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.slot'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.page'].fillna(0.0, inplace=True)
    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.adNetworkType'].fillna('N/A', inplace=True)
    df['trafficSource.adwordsClickInfo.gclId'].fillna('N/A', inplace=True)
    df['trafficSource.isTrueDirect'].fillna('N/A', inplace=True)
    df['trafficSource.referralPath'].fillna('N/A', inplace=True)
    df['trafficSource.keyword'].fillna('N/A', inplace=True)
    df['totals.bounces'].fillna(0.0, inplace=True)
    df['totals.newVisits'].fillna(0.0, inplace=True)
    df['totals.pageviews'].fillna(0.0, inplace=True)

del train
del test
train = train_new
test = test_new
del train_new
del test_new

for df in [train, test]:
    df['date_new'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['date'] = df['date_new'].dt.date
    df['sess_date_dow'] = df['date_new'].dt.dayofweek
    df['sess_date_hours'] = df['date_new'].dt.hour
    df['sess_month'] = df['date_new'].dt.month

# Dropping Constant Columns

const_cols = [col for col in train.columns if len(train[col].unique())==1]

for df in [train, test]:
    df.drop(const_cols, 1, inplace=True)

train.drop('trafficSource.campaignCode', 1, inplace=True)

# Modeling with Olivier's method
#https://www.kaggle.com/ogrellier/teach-lightgbm-to-sum-predictions

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

y_reg = train['totals.transactionRevenue'].fillna(0)
del train['totals.transactionRevenue']

if 'totals.transactionRevenue' in test.columns:
    del test['totals.transactionRevenue']

excluded_features = [
    'date', 'date_new', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]

for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])

for df in [train, test]:
    df['hits/pageviews'] = (df["totals.pageviews"]/(df["totals.hits"])).apply(lambda x: 0 if np.isinf(x) else x)
    df['is_high_hits'] = np.logical_or(df["totals.hits"]>4,df["totals.pageviews"]>4).astype(np.int32)
    df["Revenue"] = np.log1p(df["Revenue"])

warnings.simplefilter(action='ignore', category=FutureWarning)
folds = get_folds(df=train, n_splits=2) #changed from 5
y_reg = y_reg.astype(float)
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
        early_stopping_rounds=100,
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
    
print(mean_squared_error(np.log1p(y_reg), oof_reg_preds) ** .5)


# importances['gain_log'] = np.log1p(importances['gain'])
# mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
# importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

# plt.figure(figsize=(8, 12))
# plot = sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))
# fig = plot.get_figure()
# fig.savefig("feat_imp_1.png")
# plt.close()


train['predictions'] = np.expm1(oof_reg_preds)
test['predictions'] = sub_reg_preds

# Aggregate data at User level
trn_data = train[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()

# Create a list of predictions for each Visitor
trn_pred_list = train[['fullVisitorId', 'predictions']].groupby('fullVisitorId')    .apply(lambda df: list(df.predictions))    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})

# Create a DataFrame with VisitorId as index
# trn_pred_list contains dict 
# so creating a dataframe from it will expand dict values into columns
trn_all_predictions = pd.DataFrame(list(trn_pred_list.values), index=trn_data.index)
trn_feats = trn_all_predictions.columns

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    trn_all_predictions['t_mean'] = np.log1p(trn_all_predictions[trn_feats].mean(axis=1))
    trn_all_predictions['t_median'] = np.log1p(trn_all_predictions[trn_feats].median(axis=1))
    trn_all_predictions['t_sum_log'] = np.log1p(trn_all_predictions[trn_feats]).sum(axis=1)
    trn_all_predictions['t_sum_act'] = np.log1p(trn_all_predictions[trn_feats].fillna(0).sum(axis=1))
    trn_all_predictions['t_nb_sess'] = trn_all_predictions[trn_feats].isnull().sum(axis=1)
    full_data = pd.concat([trn_data, trn_all_predictions], axis=1)
    del trn_data, trn_all_predictions
    gc.collect()
    print(full_data.shape)


sub_pred_list = test[['fullVisitorId', 'predictions']].groupby('fullVisitorId')    .apply(lambda df: list(df.predictions))    .apply(lambda x: {'pred_'+str(i): pred for i, pred in enumerate(x)})

sub_data = test[train_features + ['fullVisitorId']].groupby('fullVisitorId').mean()
sub_all_predictions = pd.DataFrame(list(sub_pred_list.values), index=sub_data.index)


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
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
    print(sub_full_data.shape)

# Create target at Visitor level
train['target'] = y_reg
trn_user_target = train[['fullVisitorId', 'target']].groupby('fullVisitorId').sum()

# Train a model at Visitor level
warnings.simplefilter(action='ignore', category=FutureWarning)
folds = get_folds(df=full_data[['totals.pageviews']].reset_index(), n_splits=2)#changed from 5

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
        early_stopping_rounds=100,
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
    
print(mean_squared_error(np.log1p(trn_user_target['target']), oof_preds) ** .5)

# # Display feature importances
# vis_importances['gain_log'] = np.log1p(vis_importances['gain'])
# mean_gain = vis_importances[['gain', 'feature']].groupby('feature').mean()
# vis_importances['mean_gain'] = vis_importances['feature'].map(mean_gain['gain'])

# plt.figure(figsize=(8, 25))
# plot = sns.barplot(x='gain_log', y='feature', data=vis_importances.sort_values('mean_gain', ascending=False).iloc[:300])
# fig = plot.get_figure()
# fig.savefig("feat_imp_2.png")
# plt.close()

# Save predictions
sub_full_data['PredictedLogRevenue'] = sub_preds
sub_full_data[['PredictedLogRevenue']].to_csv('submission_new.csv', index=True)


# In[ ]:





# In[ ]:




