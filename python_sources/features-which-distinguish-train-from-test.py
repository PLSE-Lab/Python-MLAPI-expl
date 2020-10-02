#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# # Background
# - In this competition, many kaggler suffer from the discrepancy between CV and LB.
# - This could be caused by some features which have different distribution in both train and test.
# - I asked some methods to find those features. https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/67850
# - Very kind kaggler, "kain" recommended some strategies.
# - After searching and studying them, I want to share the adversarial validation.
# - I referred to various kernels. Thanks for sharing! If you got a help, please upvote them!
# - https://www.kaggle.com/konradb/adversarial-validation-and-other-scary-terms
# - https://www.kaggle.com/rspadim/adversarial-validation-porto-seguro
# - https://www.kaggle.com/ogrellier/adversarial-validation-and-lb-shakeup
# - I forked this helpful kernel, https://www.kaggle.com/prashantkikani/ensembling-fe-is-the-answer. Thanks!

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
from sklearn.model_selection import KFold, GridSearchCV

# I don't like SettingWithCopyWarnings ...
warnings.simplefilter('error', SettingWithCopyWarning)
gc.enable()
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read data

# In[ ]:


train = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_train.gz', 
                    dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
test = pd.read_csv('../input/create-extracted-json-fields-dataset/extracted_fields_test.gz', 
                   dtype={'date': str, 'fullVisitorId': str, 'sessionId':str}, nrows=None)
train.shape, test.shape


# # Feature engineering

# In[ ]:


for df in [train, test]:
    df['date'] = pd.to_datetime(df['visitStartTime'], unit='s')
    df['sess_date_dow'] = df['date'].dt.dayofweek
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


# # Prepare features

# In[ ]:


excluded_features = [
    'date', 'fullVisitorId', 'sessionId', 'totals.transactionRevenue', 
    'visitId', 'visitStartTime'
]

categorical_features = [
    _f for _f in train.columns
    if (_f not in excluded_features) & (train[_f].dtype == 'object')
]


# # Factorize features

# In[ ]:


for f in categorical_features:
    train[f], indexer = pd.factorize(train[f])
    test[f] = indexer.get_indexer(test[f])


# # Prepare target for adversarial validation

# In[ ]:


print(train.shape, test.shape)

train['istrain'] = 1
test['istrain'] = 0

df = pd.concat([train, test])

target = df['istrain']
df.drop('istrain', axis=1, inplace=True)

use_cols = [col for col in df.columns if col not in excluded_features]

df = df[use_cols]


# # Check auc

# In[ ]:


get_ipython().run_cell_magic('time', '', '\nparams = {\n    "objective" : "binary",\n    "metric" : "auc",\n    "num_leaves" : 64,\n    "learning_rate" : 0.01,\n    "bagging_fraction" : 0.7,\n    "feature_fraction" : 0.7,\n    "bagging_frequency" : 1,\n    "bagging_seed" : 1989,\n    "verbosity" : -1,\n    \'lambda_l1\':1,\n    \'lambda_l2\':1,\n    \'max_depth\': -1,\n    \'min_data_in_leaf\': 100,\n    "seed": 1989,\n}\n\nFOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)\nfeatures = list(df.columns)\nfeature_importance_df = pd.DataFrame()\n\nfor fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(df)):\n    print(\'#\'*30, \'{} of 5 folds\'.format(fold_ +1), \'#\'*30)\n    trn_data = lgb.Dataset(df.iloc[trn_idx].values, label = target.iloc[trn_idx].values)\n    val_data = lgb.Dataset(df.iloc[val_idx].values, label = target.iloc[val_idx].values)\n    \n    num_round = 2000\n    clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], \n                    verbose_eval=200, early_stopping_rounds = 50)\n    \n    \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] = features\n    fold_importance_df["split"] = clf.feature_importance(importance_type=\'split\')\n    fold_importance_df["gain"] = clf.feature_importance(importance_type=\'gain\')\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)    \n    del trn_data, val_data, clf')


# In[ ]:


feature_importance_df = feature_importance_df.groupby('feature').mean().sort_values('gain', ascending=False).reset_index()


# # Feature importance when distinguish train from test

# ## Sorted by gain

# In[ ]:


plt.figure(figsize=(10, 30))
sns.barplot(x='gain', y='feature', data=feature_importance_df)
plt.show()


# ## Sorted by split

# In[ ]:


plt.figure(figsize=(10, 30))
sns.barplot(x='split', y='feature', data=feature_importance_df.sort_values('split', ascending=False))
plt.show()


# In[ ]:





# # User-level

# In[ ]:


del df


# In[ ]:


get_ipython().run_cell_magic('time', '', "new_train = train[use_cols + ['fullVisitorId']].groupby('fullVisitorId').mean()\nnew_test = test[use_cols + ['fullVisitorId']].groupby('fullVisitorId').mean()")


# In[ ]:


print(new_train.shape, new_test.shape)

new_train['istrain'] = 1
new_test['istrain'] = 0

df = pd.concat([new_train, new_test])

target = df['istrain']
df.drop('istrain', axis=1, inplace=True)


# In[ ]:


print(df.shape)


# In[ ]:


get_ipython().run_cell_magic('time', '', '\nparams = {\n    "objective" : "binary",\n    "metric" : "auc",\n    "num_leaves" : 64,\n    "learning_rate" : 0.01,\n    "bagging_fraction" : 0.7,\n    "feature_fraction" : 0.7,\n    "bagging_frequency" : 1,\n    "bagging_seed" : 1989,\n    "verbosity" : -1,\n    \'lambda_l1\':1,\n    \'lambda_l2\':1,\n    \'max_depth\': -1,\n    \'min_data_in_leaf\': 100,\n    "seed": 1989,\n}\n\nFOLDs = KFold(n_splits=5, shuffle=True, random_state=1989)\nfeatures = list(df.columns)\nfeature_importance_df = pd.DataFrame()\n\nfor fold_, (trn_idx, val_idx) in enumerate(FOLDs.split(df)):\n    print(\'#\'*30, \'{} of 5 folds\'.format(fold_ +1), \'#\'*30)\n    trn_data = lgb.Dataset(df.iloc[trn_idx].values, label = target.iloc[trn_idx].values)\n    val_data = lgb.Dataset(df.iloc[val_idx].values, label = target.iloc[val_idx].values)\n    \n    num_round = 2000\n    clf = lgb.train(params, trn_data, num_round, valid_sets = [trn_data, val_data], \n                    verbose_eval=200, early_stopping_rounds = 50)\n    \n    \n    fold_importance_df = pd.DataFrame()\n    fold_importance_df["feature"] = features\n    fold_importance_df["split"] = clf.feature_importance(importance_type=\'split\')\n    fold_importance_df["gain"] = clf.feature_importance(importance_type=\'gain\')\n    fold_importance_df["fold"] = fold_ + 1\n    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)    \n    del trn_data, val_data, clf')


# In[ ]:


feature_importance_df = feature_importance_df.groupby('feature').mean().sort_values('gain', ascending=False).reset_index()


# # Feature importance

# ## Sorted by gain

# In[ ]:


plt.figure(figsize=(10, 30))
sns.barplot(x='gain', y='feature', data=feature_importance_df)
plt.show()


# ## Sorted by split

# In[ ]:


plt.figure(figsize=(10, 30))
sns.barplot(x='split', y='feature', data=feature_importance_df.sort_values('split', ascending=False))
plt.show()


# # Conclusion
# - Unfortunately, train and test are different.
# - So, we need to consider the difference when make features!

# - If there is bug or mistake, please talk to me! And if you have any feedback, please talk to me. 
# - Hope this kernel helps you.

# In[ ]:




