#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


# In[ ]:


#import the data
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[ ]:


train.head()
test.head()


# In[ ]:


# dataset discovery
train.info
test.info
train.columns
train.dtypes
test.dtypes


# In[ ]:


# create new dataframes
train_df = train
test_df = test


# In[ ]:


# parse the json objects in the train dataframe
import json
from pandas.io.json import json_normalize
#train_df = train_df.join(json_normalize(train_df["totals"].tolist()).add_prefix("totals.")).drop(["totals"], axis=1)
train_df = train_df.join(json_normalize(train_df['trafficSource'].map(json.loads).tolist()).add_prefix('trafficSource.'))    .drop(['trafficSource'], axis=1)
train_df = train_df.join(json_normalize(train_df['device'].map(json.loads).tolist()).add_prefix('device.'))    .drop(['device'], axis=1)
train_df = train_df.join(json_normalize(train_df['geoNetwork'].map(json.loads).tolist()).add_prefix('geoNetwork.'))    .drop(['geoNetwork'], axis=1)
train_df = train_df.join(json_normalize(train_df['totals'].map(json.loads).tolist()).add_prefix('totals.'))    .drop(['totals'], axis=1)
train_df.head()


# In[ ]:


# parse the json objects in the test dataframe

test_df = test_df.join(json_normalize(test_df['trafficSource'].map(json.loads).tolist()).add_prefix('trafficSource.'))    .drop(['trafficSource'], axis=1)
test_df = test_df.join(json_normalize(test_df['device'].map(json.loads).tolist()).add_prefix('device.'))    .drop(['device'], axis=1)
test_df = test_df.join(json_normalize(test_df['geoNetwork'].map(json.loads).tolist()).add_prefix('geoNetwork.'))    .drop(['geoNetwork'], axis=1)
test_df = test_df.join(json_normalize(test_df['totals'].map(json.loads).tolist()).add_prefix('totals.'))    .drop(['totals'], axis=1)
test_df.head()
#test_df.dtypes


# In[ ]:


#train_df['totals.transactionRevenue'].unique()


# In[ ]:


#transform the date colummns - conver integer to date 
train_df['visitStartTime'] = pd.to_datetime(train_df['visitStartTime'],unit='s')
train_df['date'] = pd.to_datetime(train_df['date'].astype(str),format='%Y%m%d')

test_df['visitStartTime'] = pd.to_datetime(test_df['visitStartTime'],unit='s')
test_df['date'] = pd.to_datetime(test_df['date'].astype(str),format='%Y%m%d')


# In[ ]:


# import required libraries
import xgboost as xgb
from sklearn.preprocessing import StandardScaler,Imputer
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import RandomizedSearchCV
from sklearn_pandas import DataFrameMapper, CategoricalImputer


# In[ ]:


train_df.head()


# In[ ]:


# find the index of columns of interest
cols = ["channelGrouping","socialEngagementType","visitNumber","trafficSource.adContent","trafficSource.adwordsClickInfo.adNetworkType","trafficSource.adwordsClickInfo.criteriaParameters","trafficSource.adwordsClickInfo.gclId"
        ,"trafficSource.adwordsClickInfo.isVideoAd","trafficSource.adwordsClickInfo.page","trafficSource.adwordsClickInfo.slot","trafficSource.campaign","trafficSource.isTrueDirect","trafficSource.keyword","trafficSource.medium"
       ,"trafficSource.referralPath","trafficSource.source","device.deviceCategory","geoNetwork.city","geoNetwork.continent","geoNetwork.country","geoNetwork.metro","geoNetwork.networkDomain","geoNetwork.region","geoNetwork.subContinent","totals.bounces","totals.hits","totals.newVisits","totals.pageviews","totals.visits","totals.transactionRevenue"]
#train_df.head()
[train_df.columns.get_loc(c) for c in train_df.columns if c in cols]
[test_df.columns.get_loc(c) for c in test_df.columns if c in cols]


# In[ ]:


# split the train data into a matrix of predictor variables and a vector of target variable
X_train, y_train = train_df.iloc[:,[0,4,6,8,9,10,12,13,14,15,17,18,19,20,21,22,25,38,40,41,46,47,48,49,50,51,52,54]],train_df.iloc[:,53]
X_test = test_df.iloc[:,[0,4,6,8,9,10,12,13,14,15,16,17,18,19,20,21,24,37,39,40,45,46,47,48,49,50,51,52]]
X_train.head()
y_train.head()
X_test.head()
X_train.info()
X_test.info()


# ## Clustering for dataset exploration

# 
