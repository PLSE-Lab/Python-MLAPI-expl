#!/usr/bin/env python
# coding: utf-8

# # 1. Read dataset

# In[ ]:


import pandas as pd
import numpy as np
import time
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('seaborn')
sns.set(font_scale=2)

import warnings 
warnings.filterwarnings('ignore')


# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize

def load_df(csv_path='../input/train.csv', nrows=None):
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

print(os.listdir("../input"))


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = load_df()\ntest_df = load_df("../input/test.csv")')


# In[ ]:


target = train_df['totals.transactionRevenue'].fillna(0).astype(float)
target = target.apply(lambda x: np.log(x) if x > 0 else x)
del train_df['totals.transactionRevenue']


# In[ ]:


columns = [col for col in train_df.columns if train_df[col].nunique() > 1]
#____________________________
train_df = train_df[columns]
test_df = test_df[columns]


# # 2. Feature engineering

# ## 2.1 Check null data

# In[ ]:


train_df.head()

percent = (100 * train_df.isnull().sum() / train_df.shape[0]).sort_values(ascending=False)

percent[:10]


# In[ ]:


percent = (100 * test_df.isnull().sum() / test_df.shape[0]).sort_values(ascending=False)
percent[:10]


# ### Remove features with NaN percents larger than 70%

# - For now, Let's remove the columns with NaN percents larger than 70%

# In[ ]:


drop_cols = ['trafficSource.referralPath', 'trafficSource.adContent', 'trafficSource.adwordsClickInfo.slot', 'trafficSource.adwordsClickInfo.page',
            'trafficSource.adwordsClickInfo.adNetworkType']


# In[ ]:


train_df.drop(drop_cols, axis=1, inplace=True)
test_df.drop(drop_cols, axis=1, inplace=True)


# ### trafficSource.keyword 

# In[ ]:


train_df['trafficSource.keyword'].fillna('nan', inplace=True)
test_df['trafficSource.keyword'].fillna('nan', inplace=True)


# In[ ]:


# for ele in train_df['trafficSource.keyword'].vablue_counts().index:
#     print(ele)

# Save your page


# - After looking the feature, simply I think that the category feature can be divided into youtube, google, other categories.

# In[ ]:


def add_new_category(x):
    x = str(x).lower()
    if x == 'nan':
        return 'nan'
    
    x = ''.join(x.split())
    
    if 'youtube' in x or 'you' in x or 'yo' in x or 'tub' in x:
        return 'youtube'
    elif 'google' in x or 'goo' in x or 'gle' in x:
        return 'google'
    else:
        return 'other'


# In[ ]:


train_df['trafficSource.keyword'] = train_df['trafficSource.keyword'].apply(add_new_category)
test_df['trafficSource.keyword'] = test_df['trafficSource.keyword'].apply(add_new_category)


# In[ ]:


train_df['trafficSource.keyword'].value_counts().sort_values(ascending=False).plot.bar()
plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats = ['trafficSource.keyword']


# ### totals.pageviews

# - NaN values in this feature could be relplaced with 0 value becuase nan view would mean no view.

# In[ ]:


train_df['totals.pageviews'].fillna(0, inplace=True)
test_df['totals.pageviews'].fillna(0, inplace=True)


# In[ ]:


train_df['totals.pageviews'] = train_df['totals.pageviews'].astype(int)
test_df['totals.pageviews'] = test_df['totals.pageviews'].astype(int)


# In[ ]:


train_df['totals.pageviews'].plot.hist(bins=10)
plt.yscale('log')
plt.show()


# ## 2.2 Object features

# In[ ]:


features_object = [col for col in train_df.columns if train_df[col].dtype == 'object']


# In[ ]:


features_object


# ### channelGrouping

# In[ ]:


train_df['channelGrouping'].value_counts().plot.bar()
plt.show()


# In[ ]:


categorical_feats.append('channelGrouping')


# ### device.browser

# In[ ]:


plt.figure(figsize=(20, 10))
train_df['device.browser'].value_counts().plot.bar()
plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats.append('device.browser')


# ### device.deviceCategory

# In[ ]:


# plt.figure(figsize=(10, 10))
train_df['device.deviceCategory'].value_counts().plot.bar()
# plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats.append('device.deviceCategory')


# ### device.operatingSystem

# In[ ]:


# plt.figure(figsize=(10, 10))
train_df['device.operatingSystem'].value_counts().plot.bar()
plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats.append('device.operatingSystem')


# ### geoNetwork.city

# In[ ]:


train_df['geoNetwork.city'].value_counts()


# - There are so many cities.

# In[ ]:


categorical_feats.append('geoNetwork.city')


# ### geoNetwork.continent

# In[ ]:


train_df['geoNetwork.continent'].value_counts()


# In[ ]:


categorical_feats.append('geoNetwork.continent')


# ### geoNetwork.country

# In[ ]:


train_df['geoNetwork.country'].value_counts()[:10].plot.bar()
plt.show()


# - In this dataset, many user ares in US

# In[ ]:


categorical_feats.append('geoNetwork.country')


# ### geoNetwork.metro

# In[ ]:


train_df['geoNetwork.metro'].value_counts()[:10].plot.bar()


# In[ ]:


categorical_feats.append('geoNetwork.metro')


# ### geoNetwork.networkDomain

# In[ ]:


train_df['geoNetwork.networkDomain'].value_counts()


# - There are so many domains. How can we deal with it?
# - One-hot is not good choice. It will generate 28064 featues...
# - Just remove this feature? or use this feature in efficient way?

# In[ ]:


categorical_feats.append('geoNetwork.networkDomain')


# ### geoNetwork.region

# In[ ]:


train_df['geoNetwork.region'].value_counts()


# -  US(California, New York), UK(England) and Thailand(Bangkok), Vietnam(Ho Chi Minh), Turkey(Istanbul) are top 5 region

# In[ ]:


categorical_feats.append('geoNetwork.region')


# ### geoNetwork.subContinent

# In[ ]:


train_df['geoNetwork.subContinent'].value_counts().plot.bar()
plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats.append('geoNetwork.subContinent')


# ### totals.hits

# In[ ]:


train_df['totals.hits'].value_counts()


# - This feature could be considered as continuous feature

# In[ ]:


train_df['totals.hits'] = train_df['totals.hits'].astype(int)
test_df['totals.hits'] = test_df['totals.hits'].astype(int)


# ### trafficSource.adwordsClickInfo.gclId

# In[ ]:


train_df['trafficSource.adwordsClickInfo.gclId'].value_counts()


# - I don't have any idea for this complex feature. For now, I want to remove this feature.

# In[ ]:


train_df.drop('trafficSource.adwordsClickInfo.gclId', axis=1, inplace=True)
test_df.drop('trafficSource.adwordsClickInfo.gclId', axis=1, inplace=True)


# ### trafficSource.campaign

# In[ ]:


train_df['trafficSource.campaign'].value_counts().plot.bar()
plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats.append('trafficSource.campaign')


# ### trafficSource.medium

# In[ ]:


train_df['trafficSource.medium'].value_counts().plot.bar()
plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats.append('trafficSource.medium')


# ### trafficSource.source

# In[ ]:


# for value in train_df['trafficSource.source'].value_counts().index:
#     print(value)
# save your page


# - google, baidu, facebook, reddit, yahoo, bing, yandex, 

# In[ ]:


def add_new_category(x):
    x = str(x).lower()
    if 'google' in x:
        return 'google'
    elif 'baidu' in x:
        return 'baidu'
    elif 'facebook' in x:
        return 'facebook'
    elif 'reddit' in x:
        return 'reddit'
    elif 'yahoo' in x:
        return 'yahoo'
    elif 'bing' in x:
        return 'bing'
    elif 'yandex' in x:
        return 'yandex'
    else:
        return 'other'


# In[ ]:


train_df['trafficSource.source'] = train_df['trafficSource.source'].apply(add_new_category)
test_df['trafficSource.source'] = test_df['trafficSource.source'].apply(add_new_category)


# In[ ]:


train_df['trafficSource.source'].value_counts().sort_values(ascending=False).plot.bar()
plt.yscale('log')
plt.show()


# In[ ]:


categorical_feats.append('trafficSource.source')


# ### 	device.isMobile

# - THis feature is boolean type. So let's do astype(int)

# In[ ]:


train_df['device.isMobile'] = train_df['device.isMobile'].astype(int)
test_df['device.isMobile'] = test_df['device.isMobile'].astype(int)


# ## 2.3 Time feature

# In[ ]:


len_train = train_df.shape[0]

df_all = pd.concat([train_df, test_df])


# In[ ]:


def change_date_to_datetime(x):
    str_time = str(x)
    date = '{}-{}-{}'.format(str_time[:4], str_time[4:6], str_time[6:])
    return date

def add_time_feature(data):
    data['date'] = pd.to_datetime(data['date'])
    data['Year'] = data.date.dt.year
    data['Month'] = data.date.dt.month
    data['Day'] = data.date.dt.day
    data['WeekOfYear'] = data.date.dt.weekofyear
    return data

df_all['date'] = df_all['date'].apply(change_date_to_datetime)
df_all = add_time_feature(df_all)


# In[ ]:


categorical_feats += ['Year', 'Month', 'Day', 'WeekOfYear']


# In[ ]:


df_all.drop('date', axis=1, inplace=True)


# # 3. Model development

# ## 3.1 Label encoding

# In[ ]:


from sklearn.preprocessing import LabelEncoder


# In[ ]:


for col in categorical_feats:
    lbl = LabelEncoder()
    df_all[col] = lbl.fit_transform(df_all[col])


# In[ ]:


train_df = df_all[:len_train]
test_df = df_all[len_train:]


# ## 3.2 Drop features

# In[ ]:


train_fullVisitorId = train_df['fullVisitorId']
train_sessionId = train_df['sessionId']
train_visitId = train_df['visitId']

test_fullVisitorId = test_df['fullVisitorId']
test_sessionId = test_df['sessionId']
test_visitId = test_df['visitId']

train_df.drop(['fullVisitorId', 'sessionId', 'visitId'], axis=1, inplace=True)
test_df.drop(['fullVisitorId', 'sessionId', 'visitId'], axis=1, inplace=True)


# In[ ]:


train_df.head()


# In[ ]:


param = {'num_leaves':128,
         'min_data_in_leaf': 800, 
         'objective':'regression',
         'max_depth': 7,
         'learning_rate':0.006,
         "min_child_samples":40,
         "boosting":"gbdt",
         "feature_fraction":0.8,
         "bagging_freq":1,
         "bagging_fraction":0.8 ,
         "bagging_seed": 3,
         "metric": 'rmse',
         "lambda_l1": 1,
         'lambda_l2': 1,
         "verbosity": -1,
        "subsample": 0.8,
        "colsample_bytree": 0.5,
        "colsample_bylevel":0.6}


# ## 3.3 Training model

# In[ ]:


folds = KFold(n_splits=5, shuffle=True, random_state=15)
oof = np.zeros(len(train_df))
predictions = np.zeros(len(test_df))
start = time.time()
features = list(train_df.columns)
feature_importance_df = pd.DataFrame()

for fold_, (trn_idx, val_idx) in enumerate(folds.split(train_df.values, target.values)):
    trn_data = lgb.Dataset(train_df.iloc[trn_idx], label=target.iloc[trn_idx], categorical_feature=categorical_feats)
    val_data = lgb.Dataset(train_df.iloc[val_idx], label=target.iloc[val_idx], categorical_feature=categorical_feats)
    
    num_round = 10000
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=400, early_stopping_rounds = 500, categorical_feature=categorical_feats)
    oof[val_idx] = clf.predict(train_df.iloc[val_idx].values, num_iteration=clf.best_iteration)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = features
    fold_importance_df["importance"] = clf.feature_importance()
    fold_importance_df["fold"] = fold_ + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    predictions += clf.predict(test_df.values, num_iteration=clf.best_iteration) / folds.n_splits


# In[ ]:


print("CV score: {:<8.5f}".format(mean_squared_error(oof, target)**0.5))


# In[ ]:


cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:1000].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

plt.figure(figsize=(14,10))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
plt.tight_layout()
plt.savefig('lgbm_importances.png')


# ## 3.4 Submission

# In[ ]:


submission = pd.DataFrame()

submission['fullVisitorId'] = test_fullVisitorId

submission['PredictedLogRevenue'] = predictions

grouped_test = submission[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
grouped_test.to_csv('submit.csv',index=False)


# In[ ]:




