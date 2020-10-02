#!/usr/bin/env python
# coding: utf-8

# ## General information
# 
# This kernel continues my [previous one](https://www.kaggle.com/artgor/fork-of-eda-on-basic-data-and-lgb-in-progress) - you can see EDA and other things there.
# 
# This kernel is dedicated to feature generation. I'll generate features step by step and try to increase CV.

# In[ ]:


import numpy as np 
import pandas as pd 
import json
import bq_helper
from pandas.io.json import json_normalize
import seaborn as sns 
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('ggplot')
import lightgbm as lgb

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.metrics import mean_squared_error
import time


# In[ ]:


# https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields

def load_df(csv_path='../input/train.csv', JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']):

    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'})
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)

    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = load_df("../input/train.csv")')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test = load_df("../input/test.csv")')


# ## Data processing

# Some of columns aren't available in this dataset, let's drop them.

# In[ ]:


cols_to_drop = [col for col in train.columns if train[col].nunique(dropna=False) == 1]
train.drop(cols_to_drop, axis=1, inplace=True)
test.drop([col for col in cols_to_drop if col in test.columns], axis=1, inplace=True)

#only one not null value
train.drop(['trafficSource.campaignCode'], axis=1, inplace=True)

print(f'Dropped {len(cols_to_drop)} columns.')


# In[ ]:


train['totals.transactionRevenue'] = train['totals.transactionRevenue'].fillna(0).astype(int)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'])


# In[ ]:


def process_df(df):
    """Process df and create new features."""
    
    for col in ['visitNumber', 'totals.hits', 'totals.pageviews']:
        df[col] = df[col].astype(float)
    df['trafficSource.adwordsClickInfo.isVideoAd'].fillna(True, inplace=True)
    df['trafficSource.isTrueDirect'].fillna(False, inplace=True)
    
    df['date'] = pd.to_datetime(df['date'].apply(lambda x: str(x)[:4] + '-' + str(x)[4:6] + '-' + str(x)[6:]))
    
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['weekday'] = df['date'].dt.weekday
    df['weekofyear'] = df['date'].dt.weekofyear

    df['month_unique_user_count'] = df.groupby('month')['fullVisitorId'].transform('nunique')
    df['day_unique_user_count'] = df.groupby('day')['fullVisitorId'].transform('nunique')
    df['weekday_unique_user_count'] = df.groupby('weekday')['fullVisitorId'].transform('nunique')
    df['weekofyear_unique_user_count'] = df.groupby('weekofyear')['fullVisitorId'].transform('nunique')
    
    df['browser_category'] = df['device.browser'] + '_' + df['device.deviceCategory']
    df['browser_operatingSystem'] = df['device.browser'] + '_' + df['device.operatingSystem']
    df['source_country'] = df['trafficSource.source'] + '_' + df['geoNetwork.country']
    
    df['visitNumber'] = np.log1p(df['visitNumber'])
    df['totals.hits'] = np.log1p(df['totals.hits'])
    df['totals.pageviews'] = np.log1p(df['totals.pageviews'].fillna(0))

    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['mean_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('mean')
    df['sum_hits_per_day'] = df.groupby(['day'])['totals.hits'].transform('sum')
    
    df['sum_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.pageviews'].transform('mean')

    df['sum_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('sum')
    df['count_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('count')
    df['mean_pageviews_per_region'] = df.groupby('geoNetwork.region')['totals.pageviews'].transform('mean')
    
    df['sum_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('sum')
    df['count_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('count')
    df['mean_hits_per_network_domain'] = df.groupby('geoNetwork.networkDomain')['totals.hits'].transform('mean')

    df['sum_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('sum')
    df['count_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('count')
    df['mean_hits_per_region'] = df.groupby('geoNetwork.region')['totals.hits'].transform('mean')

    df['sum_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('sum')
    df['count_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('count')
    df['mean_hits_per_country'] = df.groupby('geoNetwork.country')['totals.hits'].transform('mean')

    df['user_pageviews_sum'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('sum')
    df['user_hits_sum'] = df.groupby('fullVisitorId')['totals.hits'].transform('sum')

    df['user_pageviews_count'] = df.groupby('fullVisitorId')['totals.pageviews'].transform('count')
    df['user_hits_count'] = df.groupby('fullVisitorId')['totals.hits'].transform('count')

    df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
    df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()
    df['user_pageviews_sum_to_mean'] = df['user_pageviews_sum'] / df['user_pageviews_sum'].mean()
    df['user_hits_sum_to_mean'] = df['user_hits_sum'] / df['user_hits_sum'].mean()
    
    df['user_pageviews_to_region'] = df['user_pageviews_sum'] / df['mean_pageviews_per_region']
    df['user_hits_to_region'] = df['user_hits_sum'] / df['mean_hits_per_region']
    
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = process_df(train)')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'test = process_df(test)')


# In[ ]:


not_num_cols = ['visitId', 'totals.transactionRevenue', 'month', 'day', 'weekday', 'weekofyear']
num_cols = [col for col in train.columns if train[col].dtype in ['float64', 'int64'] and col not in not_num_cols]

not_cat_cols = ['fullVisitorId', 'sessionId', 'trafficSource.referralPath']
cat_cols = [col for col in train.columns if train[col].dtype == 'object' and col not in not_cat_cols] + ['month', 'day', 'weekday', 'weekofyear']

no_use = ['visitStartTime', "date", "fullVisitorId", "sessionId", "visitId", 'totals.transactionRevenue', 'trafficSource.referralPath']


# ### More features
# 
# For now only several features are calculated.

# In[ ]:


def generate_more_features(df):
    for col in num_cols:
        df[col + '_root'] = df[col] ** 0.5
        
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = generate_more_features(train)\ntest = generate_more_features(test)')


# In[ ]:


def generate_more_features(df, features_slice=[]):
    """
    Generate more features by multiplying all numerical columns by each other.
    But can't do it for all columns due to memory limitations
    """
    for col1 in num_cols[features_slice[0] : features_slice[1]]:
        for col2 in num_cols:
            if col1 != col2:
                # print(col1, col2)
                df[col1 + '_' + col2] = df[col1] * df[col2]
        
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = generate_more_features(train, [0, 3])\ntest = generate_more_features(test, [0, 3])')


# ### Feature processing

# In[ ]:


for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))


# In[ ]:


train = train.sort_values('date')
X = train.drop(no_use, axis=1)
y = train['totals.transactionRevenue']
X_test = test.drop([col for col in no_use if col in test.columns], axis=1)


#     In fact it seems that it will take some time to find a good validation - TimeSeriesSplit gives a high variance in scores, so I'll try kfold for now.

# In[ ]:


params = {"objective" : "regression",
          "metric" : "rmse", 
          #"max_depth": 6,
          "min_child_samples": 20, 
          "reg_alpha": 0.033948965191129526, 
          "reg_lambda": 0.06490202783578762,
          "num_leaves" : 34,
          "learning_rate" : 0.019732018807662323,
          "subsample" : 0.876,
          "colsample_bytree" : 0.85,
          "subsample_freq ": 5,
          #'min_split_gain': 0.024728814179385473,
          #'min_child_weight': 39.40511524645848
         }

n_fold = 5
folds = KFold(n_splits=n_fold, random_state=42)
# Cleaning and defining parameters for LGBM
model = lgb.LGBMRegressor(**params, n_estimators = 20000, nthread = 4, n_jobs = -1)


# In[ ]:


oof = np.zeros(len(train))
oof_1 = np.zeros(len(train))
prediction = np.zeros(len(test))
scores = []
scores_1 = []
feature_importance = pd.DataFrame()
for fold_n, (train_index, valid_index) in enumerate(folds.split(X)):
    print('Fold', fold_n, 'started at', time.ctime())
    X_train, X_valid = X.iloc[train_index], X.iloc[valid_index]
    y_train, y_valid = y.iloc[train_index], y.iloc[valid_index]

    model.fit(X_train, y_train, 
            eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric='rmse',
            verbose=500, early_stopping_rounds=100)
    
    y_pred_valid = model.predict(X_valid)
    oof[valid_index] = y_pred_valid.reshape(-1,)
    scores.append(mean_squared_error(y_valid, y_pred_valid) ** 0.5)
    
    y_val = train.iloc[valid_index][['fullVisitorId', 'totals.transactionRevenue']]
    y_val["totals.transactionRevenue"] = y_val["totals.transactionRevenue"].apply(lambda x : 0.0 if x < 0 else x)
    y_val["totals.transactionRevenue"] = y_val["totals.transactionRevenue"].fillna(0.0)
    y_val['totals.transactionRevenue'] = np.expm1(y_val['totals.transactionRevenue'])
    y_val_sum_true = y_val.groupby('fullVisitorId').sum().reset_index()['totals.transactionRevenue']
    y_val_sum_true = np.log1p(y_val_sum_true)
    
    oof_1[valid_index] = np.expm1(oof[valid_index])
    
    val_df = train.iloc[valid_index][['fullVisitorId']]
    val_df['totals.transactionRevenue'] = oof[valid_index]
    val_df["totals.transactionRevenue"] = val_df["totals.transactionRevenue"].apply(lambda x : 0.0 if x < 0 else x)
    val_df["totals.transactionRevenue"] = val_df["totals.transactionRevenue"].fillna(0.0)
    y_val_sum_pred = val_df.groupby('fullVisitorId').sum().reset_index()['totals.transactionRevenue']
    y_val_sum_pred = np.log1p(y_val_sum_pred)
    scores_1.append(mean_squared_error(y_val_sum_true, y_val_sum_pred) ** 0.5)
    
       
    y_pred = model.predict(X_test, num_iteration=model.best_iteration_)
    prediction += y_pred
    
    # feature importance
    fold_importance = pd.DataFrame()
    fold_importance["feature"] = X.columns
    fold_importance["importance"] = model.feature_importances_
    fold_importance["fold"] = fold_n + 1
    feature_importance = pd.concat([feature_importance, fold_importance], axis=0)
    
    print('')
    print(f'Fold {fold_n}. RMSE: {scores[-1]:.4f}.')
    print('')

prediction /= n_fold
feature_importance["importance"] /= n_fold


# In[ ]:


print('CV mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores), np.std(scores)))
print('CV new mean score: {0:.4f}, std: {1:.4f}.'.format(np.mean(scores_1), np.std(scores_1)))


# In[ ]:


#lgb.plot_importance(model, max_num_features=30);
#feature_importance_lgb.sort_values('importance', ascending=False).set_index('features').plot('')
cols = feature_importance[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:50].index

best_features = feature_importance.loc[feature_importance.feature.isin(cols)]

plt.figure(figsize=(16, 12));
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False));
plt.title('LGB Features (avg over folds)');


# In[ ]:


submission = test[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = prediction
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv(f'lgb_cv{np.mean(scores):.4f}_std_{np.std(scores):.4f}_prediction_old.csv', index=False)
oof_df = pd.DataFrame({"fullVisitorId": train["fullVisitorId"], "PredictedLogRevenue": oof})
oof_df.to_csv(f'lgb_cv{np.mean(scores):.4f}_std_{np.std(scores):.4f}_oof_old.csv', index=False)


# In[ ]:


submission = test[['fullVisitorId']].copy()
submission.loc[:, 'PredictedLogRevenue'] = np.expm1(prediction)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission["PredictedLogRevenue"] = submission["PredictedLogRevenue"].fillna(0.0)
grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test["PredictedLogRevenue"] = np.log1p(grouped_test["PredictedLogRevenue"])
grouped_test.to_csv(f'lgb_cv{np.mean(scores_1):.4f}_std_{np.std(scores_1):.4f}_prediction_new.csv', index=False)
oof_df = pd.DataFrame({"fullVisitorId": train["fullVisitorId"], "PredictedLogRevenue": oof_1})
oof_df.to_csv(f'lgb_cv{np.mean(scores_1):.4f}_std_{np.std(scores_1):.4f}_oof_new.csv', index=False)


# In[ ]:





# In[ ]:





# In[ ]:




