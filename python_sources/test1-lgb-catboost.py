#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from math import sqrt
import lightgbm as lgb

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


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


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = load_df()\ntest_df = load_df("../input/test.csv")')


# In[ ]:


train_df.info()


# In[ ]:


train_df.head(5)


# In[ ]:


#target variable
#Since we are predicting the natural log of sum of all transactions of the user
#let us sum up the transaction revenue at user level and take a log and then do a scatter plot.
train_df["totals.transactionRevenue"] = train_df["totals.transactionRevenue"].astype('float')
gdf = train_df.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(8,6))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12) 
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()


# In[ ]:


nzt=pd.notnull(train_df['totals.transactionRevenue']).sum() #number of instances with non zero revenue
nzt


# In[ ]:


nzu=(gdf["totals.transactionRevenue"]>0).sum() #number of unique customers with non zero revenue
nzu


# In[ ]:


train_df.fullVisitorId.nunique() #unique visiters in the train set


# In[ ]:


test_df.fullVisitorId.nunique() #unique visiters in the train set


# In[ ]:


len(set(train_df.fullVisitorId.unique()).intersection(set(test_df.fullVisitorId.unique()))) #unique visitors between train and test


# In[ ]:


const_cols = [c for c in train_df.columns if train_df[c].nunique(dropna=False)==1 ] 
const_cols #columns with constant values which can be dropped


# In[ ]:


(set(train_df.columns).difference(set(test_df.columns))) #variable which are not common in both test and train


# In[ ]:


cols_to_drop = const_cols + ['sessionId'] #drop constant columns

train_df = train_df.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
test_df = test_df.drop(cols_to_drop, axis=1)


# In[ ]:


train_df["totals.transactionRevenue"].fillna(0, inplace=True) #impute 0 in Na's place
train_y = train_df["totals.transactionRevenue"].values
train_id = train_df["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values


# In[ ]:


#Label encoder
cat_cols = ["channelGrouping", "device.browser", 
            "device.deviceCategory", "device.operatingSystem", 
            "geoNetwork.city", "geoNetwork.continent", 
            "geoNetwork.country", "geoNetwork.metro",
            "geoNetwork.networkDomain", "geoNetwork.region", 
            "geoNetwork.subContinent", "trafficSource.adContent", 
            "trafficSource.adwordsClickInfo.adNetworkType", 
            "trafficSource.adwordsClickInfo.gclId", 
            "trafficSource.adwordsClickInfo.page", 
            "trafficSource.adwordsClickInfo.slot", "trafficSource.campaign",
            "trafficSource.keyword", "trafficSource.medium", 
            "trafficSource.referralPath", "trafficSource.source",
            'trafficSource.adwordsClickInfo.isVideoAd', 'trafficSource.isTrueDirect']
for col in cat_cols:
    print(col)
    lbl = LabelEncoder()
    lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    train_df[col] = lbl.transform(list(train_df[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


# In[ ]:


#Convert numerical columns into float
num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
for col in num_cols:
    train_df[col] = train_df[col].astype(float)
    test_df[col] = test_df[col].astype(float)


# In[ ]:


train_df['date'].


# In[ ]:


train_df.info()


# In[ ]:


import datetime
# Split the train dataset into development and valid based on time 
dev_df = train_df.iloc[0:700000,:]
val_df = train_df.iloc[700001:,:]
dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
val_y = np.log1p(val_df["totals.transactionRevenue"].values)

dev_X = dev_df[cat_cols + num_cols] 
val_X = val_df[cat_cols + num_cols] 
test_X = test_df[cat_cols + num_cols]


# def run_lgb(train_X, train_y, val_X, val_y, test_X):
#     params = {
#         "objective" : "regression",
#         "metric" : "rmse", 
#         "num_leaves" : 30,
#         "min_child_samples" : 100,
#         "learning_rate" : 0.1,
#         "bagging_fraction" : 0.7,
#         "feature_fraction" : 0.5,
#         "bagging_frequency" : 5,
#         "bagging_seed" : 2018,
#         "verbosity" : -1
#     }
#     
#     lgtrain = lgb.Dataset(train_X, label=train_y)
#     lgval = lgb.Dataset(val_X, label=val_y)
#     model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
#     
#     pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
#     pred_eval=model.predict(val_X,num_iteration=model.best_iteration)
#     pred_eval[pred_eval<0] = 0
#     rms = sqrt(mean_squared_error(val_y,pred_eval))
#     print(rms)
#     return pred_test_y, model
# 

# pred_test, model,rms = run_lgb(dev_X, dev_y, val_X, val_y, test_X)

# In[ ]:


def run_catbst(train_X, train_y, val_X, val_y, test_X):
    eval_set=[(val_X,val_y)]
    clf=cb.CatBoostRegressor(depth=10, l2_leaf_reg= 1,
                            learning_rate= 0.15 ,eval_metric="RMSE",iterations=1000, early_stopping_rounds=100)
    model=clf.fit(train_X,train_y,eval_set=eval_set)
    pred_test_y= model.predict(test_X)
    pred_eval=model.predict(val_X)
    pred_eval[pred_eval<0] = 0
    rms = sqrt(mean_squared_error(val_y,pred_eval))
    return pred_test_y, model ,rms


# In[ ]:


import catboost as cb
pred_test, model,rms = run_catbst(dev_X, dev_y, val_X, val_y, test_X)


# In[ ]:


rms


# In[ ]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("test1_cat.csv", index=False)

