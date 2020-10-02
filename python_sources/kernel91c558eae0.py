#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importing Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import gc
import sys

from pandas.io.json import json_normalize
from datetime import datetime
from sklearn import preprocessing

import os
print(os.listdir("../input"))


# In[ ]:


def load_df(csv_path, nr=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    #converters = cv
    #dtype = dt
    #nrows = nr
    #column_as_df = cad
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     dtype={'fullVisitorId': 'str'},
                     nrows=nr)
    
    for column in JSON_COLUMNS:
        cad = json_normalize(df[column])
        cad.columns = [f"{column}.{subcolumn}" for subcolumn in cad.columns]
        df = df.drop(column, axis=1).merge(cad, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = load_df('../input/train.csv')\ntest = load_df('../input/test.csv')\n\nprint('train date:', min(train['date']), 'to', max(train['date']))\nprint('test date:', min(test['date']), 'to', max(test['date']))")


# In[ ]:


# only train feature
for c in train.columns.values:
    if c not in test.columns.values: print(c)


# In[ ]:


# totals, the sub-column transactionRevenue contains the revenue information we are trying to predict
#train_rev = train_revenue
train_revenue = train[~train['totals.transactionRevenue'].isnull()].copy()
print(len(train_revenue))
train_revenue.head()


# In[ ]:


train['totals.transactionRevenue'].fillna(0, inplace=True)
train['totals.transactionRevenue'] = np.log1p(train['totals.transactionRevenue'].astype(float))
print(train['totals.transactionRevenue'].describe())


# In[ ]:


ad = train.append(test, sort=False).reset_index(drop=True)


# In[ ]:


print(ad.info())


# In[ ]:


null_cnt = train.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])


# In[ ]:


ad['totals.pageviews'].fillna(1, inplace=True)
ad['totals.newVisits'].fillna(0, inplace=True)
ad['totals.bounces'].fillna(0, inplace=True)
ad['totals.pageviews'] = ad['totals.pageviews'].astype(int)
ad['totals.newVisits'] = ad['totals.newVisits'].astype(int)
ad['totals.bounces'] = ad['totals.bounces'].astype(int)

ad['trafficSource.isTrueDirect'].fillna(False, inplace=True)


# In[ ]:


cc = [col for col in ad.columns if ad[col].nunique() == 1]

print('drop columns:', cc)
ad.drop(cc, axis=1, inplace=True)


# In[ ]:


format_str = '%Y%m%d'
ad['formated_date'] = ad['date'].apply(lambda x: datetime.strptime(str(x), format_str))
ad['_month'] = ad['formated_date'].apply(lambda x:x.month)
ad['_quarterMonth'] = ad['formated_date'].apply(lambda x:x.day//8)
ad['_day'] = ad['formated_date'].apply(lambda x:x.day)
ad['_weekday'] = ad['formated_date'].apply(lambda x:x.weekday())

#ad(['date','formated_date'], axis=1, inplace=True)


# In[ ]:


print(ad['channelGrouping'].value_counts())


# In[ ]:


print('train all:', len(train))
print('train unique fullVisitorId:', train['fullVisitorId'].nunique())
print('train unique visitId:', train['visitId'].nunique())
print('-' * 30)
print('test all:', len(test))
print('test unique fullVisitorId:', test['fullVisitorId'].nunique())
print('test unique visitId:', test['visitId'].nunique())
#print('common fullVisitorId:', len(pd.merge(train, test, how='inner', on='fullVisitorId'))) # 183434


# In[ ]:


print(ad['visitNumber'].value_counts()[:5])
print('-' * 30)
print(ad['totals.newVisits'].value_counts())
print('-' * 30)
print(ad['totals.bounces'].value_counts())


# In[ ]:


ad['_visitStartHour'] = ad['visitStartTime'].apply(
    lambda x: str(datetime.fromtimestamp(x).hour))


# In[ ]:


print('train all sessionId:', len(train['sessionId']))
print('train unique sessionId:', train['sessionId'].nunique())


# In[ ]:


print('unique browser count:', train['device.browser'].nunique())
print('-' * 30)
print(ad['device.browser'].value_counts()[:10])


# In[ ]:


pd.crosstab(ad['device.deviceCategory'], ad['device.isMobile'], margins=False)


# In[ ]:


print('unique operatingSystem count:', train['device.operatingSystem'].nunique())
print('-' * 30)
print(ad['device.operatingSystem'].value_counts()[:10])


# In[ ]:


print(ad['geoNetwork.city'].value_counts()[:10])
print('-' * 30)
print(ad['geoNetwork.region'].value_counts()[:10])
print('-' * 30)
print(ad['geoNetwork.subContinent'].value_counts()[:10])
print('-' * 30)
print(ad['geoNetwork.continent'].value_counts())


# In[ ]:


print(ad['geoNetwork.metro'].value_counts()[:10])


# In[ ]:


print(ad['geoNetwork.networkDomain'].value_counts()[:10])


# In[ ]:


print(ad['totals.hits'].value_counts()[:10])

ad['totals.hits'] = ad['totals.hits'].astype(int)
ad['_meanHitsPerDay'] = ad.groupby(['_day'])['totals.hits'].transform('mean')
ad['_meanHitsPerWeekday'] = ad.groupby(['_weekday'])['totals.hits'].transform('mean')
ad['_meanHitsPerMonth'] = ad.groupby(['_month'])['totals.hits'].transform('mean')
ad['_sumHitsPerDay'] = ad.groupby(['_day'])['totals.hits'].transform('sum')
ad['_sumHitsPerWeekday'] = ad.groupby(['_weekday'])['totals.hits'].transform('sum')
ad['_sumHitsPerMonth'] = ad.groupby(['_month'])['totals.hits'].transform('sum')


# In[ ]:


print(ad['totals.pageviews'].value_counts()[:10])
ad['totals.pageviews'] = ad['totals.pageviews'].astype(int)


# In[ ]:


print(ad['trafficSource.adContent'].value_counts()[:10])
print('-' * 30)
print(train_revenue['trafficSource.adContent'].value_counts())

ad['_adContentGMC'] = (ad['trafficSource.adContent'] == 'Google Merchandise Collection').astype(np.uint8)


# In[ ]:


print(ad['trafficSource.campaign'].value_counts()[:10])
ad['_withCampaign'] = (ad['trafficSource.campaign'] != '(not set)').astype(np.uint8)


# In[ ]:


print(ad['trafficSource.isTrueDirect'].value_counts())


# In[ ]:


print(ad['trafficSource.keyword'].value_counts()[:10])


# In[ ]:


print(ad['trafficSource.medium'].value_counts())
print('-' * 30)
print(train_revenue['trafficSource.medium'].value_counts())


# In[ ]:


print(ad['trafficSource.referralPath'].value_counts()[:10])


# In[ ]:


print(ad['trafficSource.source'].value_counts()[:10])
ad['_sourceGpmall'] = (ad['trafficSource.source'] == 'mall.googleplex.com').astype(np.uint8)


# In[ ]:


train_revenue = train_revenue.sort_values(['visitStartTime']).reset_index()
train_revenue['_buyCount'] = train_revenue.groupby('fullVisitorId').cumcount() + 1
ad = pd.merge(ad, train_revenue[['_buyCount','fullVisitorId','visitId']], 
                    on=['fullVisitorId','visitId'], how='left')
for fvId in train_revenue['fullVisitorId'].unique():
    visitor_data = ad[ad['fullVisitorId'] == fvId].sort_values(['visitStartTime'])['_buyCount'].reset_index()
    ad.loc[ad['fullVisitorId'] == fvId, '_buyCount'] = visitor_data['_buyCount'].fillna(method='ffill').values
ad['_buyCount'].fillna(0, inplace=True)
ad['_buyRate'] = ad['_buyCount'] / ad['visitNumber']


# In[ ]:


null_cnt = ad.isnull().sum().sort_values()
print(null_cnt[null_cnt > 0])


# In[ ]:


ad.info()


# In[ ]:


c = ['fullVisitorId',
     'visitNumber',
     'device.deviceCategory',
     'geoNetwork.subContinent',
     'totals.transactionRevenue',
     'totals.newVisits',
     'totals.hits',
     'totals.pageviews',
     '_month',
     '_quarterMonth',
     '_weekday',
     '_visitStartHour',
     '_adContentGMC',
     '_withCampaign',
     '_sourceGpmall',
     '_buyRate']
ad = ad[c]

for i, t in ad.loc[:, ad.columns != 'fullVisitorId'].dtypes.iteritems():
    if t == object:
        ad = pd.concat([ad, pd.get_dummies(ad[i].astype(str), prefix=i)], axis=1)
        ad.drop(i, axis=1, inplace=True)


# In[ ]:


ad.info()


# In[ ]:


train = ad[ad['totals.transactionRevenue'].notnull()]
test = ad[ad['totals.transactionRevenue'].isnull()].drop(['totals.transactionRevenue'], axis=1)


# In[ ]:


train_id = train['fullVisitorId']
test_id = test['fullVisitorId']

Y_train_reg = train.pop('totals.transactionRevenue')
Y_train_cls = (Y_train_reg.fillna(0) > 0).astype(np.uint8)

X_train = train.drop(['fullVisitorId'], axis=1)
X_test  = test.drop(['fullVisitorId'], axis=1)

print(X_train.shape, X_test.shape)


# In[ ]:


import sys
import gc

del ad, train, test, train_revenue
gc.collect()

print(pd.DataFrame([[val for val in dir()], [sys.getsizeof(eval(val)) for val in dir()]],
                   index=['name','size']).T.sort_values('size', ascending=False).reset_index(drop=True)[:10])


# In[ ]:


from sklearn import ensemble, metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.model_selection import GridSearchCV, KFold, GroupKFold
from sklearn.metrics import mean_squared_error, classification_report, roc_auc_score, f1_score, log_loss

import xgboost as xgb
import lightgbm as lgb


# In[ ]:


if '_revenueProba' in X_train.columns : del X_train['_revenueProba']
if '_revenueProba' in X_test.columns : del X_test['_revenueProba']


# In[ ]:


get_ipython().run_cell_magic('time', '', 'reg = ensemble.GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=3, verbose=1, random_state=42)\nreg.fit(X_train, Y_train_cls)\npred_reg = reg.predict(X_test)\n\nprint(len(pred_reg), len(pred_reg[pred_reg > 0.1]))')


# In[ ]:


reg = ensemble.GradientBoostingRegressor(n_estimators=1000, learning_rate=0.5, max_depth=3, verbose=1, random_state=42)
reg.fit(X_train[Y_train_reg > 0], Y_train_reg[Y_train_reg > 0])

pred = np.zeros(len(pred_reg))
for i in np.arange(len(pred_reg)):
        pred[i] = reg.predict([X_test.iloc[i]])[0] * pred_reg[i]


# In[ ]:


#submission = sub


# In[ ]:


sub = pd.DataFrame({'fullVisitorId':test_id, 'PredictedLogRevenue':pred})
sub["PredictedLogRevenue"] = sub["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
sub["PredictedLogRevenue"] = sub["PredictedLogRevenue"].fillna(0.0)
sub_sum = sub[['fullVisitorId', 'PredictedLogRevenue']].groupby('fullVisitorId').sum().reset_index()
sub_sum.to_csv("submission.csv", index=False)
sub_sum[sub_sum['PredictedLogRevenue'] > 0.0]


# In[ ]:


sub_sum['PredictedLogRevenue'].describe()

