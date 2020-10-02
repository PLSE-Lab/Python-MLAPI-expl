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


# In[ ]:


import os
import json
from pandas.io.json import json_normalize
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import matplotlib.pyplot as plt


# # Data Imput

# In[ ]:


# Loading data and flattening JSON columns
def load_df(csv_path='../input/train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, 
                     # Set the date, fullVisitorId, sessionId as string of constant
                     dtype={'date': str, 'fullVisitorId': str, 'sessionId': str}, 
                     nrows=nrows)
    
    for column in JSON_COLUMNS:
        column_as_df = json_normalize(list(df[column]))
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Loaded {os.path.basename(csv_path)}. Shape: {df.shape}")
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_df = load_df()\ntest_df = load_df('../input/test.csv')")


# # EDA
# ## Missing Value Detection

# In[ ]:


def na_detect(df):
    total = df.isnull().sum().sort_values(ascending = False)
    percent = (df.isnull().sum() / df.isnull().count() * 100 ).sort_values(ascending = False)
    df_opt = pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])
    
    plt.figure(figsize=(20,20))
    fig, ax = plt.subplots()
    col_na = total[total>0]
    bar_na = ax.barh(col_na.index, col_na.values, 0.8)
    for i, v in enumerate(col_na.values):
        ax.text(v + 5, i - .15 , str(v), color='red')#, fontweight='bold')
    plt.title('Variables with Missing Value')
    plt.xlabel('Quantity of Missing Value')
    plt.ylabel('Columns')
    plt.show()
    
    print (df_opt[~(df_opt['Total'] == 0)])
    
    return


# In[ ]:


na_detect(train_df)


# ## Difference Between Train and Test Dataset

# In[ ]:


set(train_df.columns).difference(set(test_df.columns))


# Besides the response variable 'totals.transactionRevenue', the training set also has a column 'trafficSource.campaignCode'  which doesn't exist in test set.

# ## TimeStamp/Date Conversion  

# In[ ]:


def date_convert(df):
    df['visitdate'] = pd.to_datetime(df['visitStartTime'], unit='s')
    #df['visitdate'] = pd.datetime.utcfromtimestamp(test_df['visitStartTime'])
    df['wday'] = df['visitdate'].dt.weekday
    df['hour'] = df['visitdate'].dt.hour
    df['day'] = df['visitdate'].dt.day
    df['month'] = df['visitdate'].dt.month
    return    


# In[ ]:


for df in [train_df, test_df]:
    date_convert(df)
print('TrainSet:', train_df.shape)
print('TestSet:', test_df.shape)


# ## Constant Variable Detection and Removal

# In[ ]:


def constant_process(df):
    num_constant = 0
    constant_cols = []
    for col in df.columns:
        if df[col].nunique()==1:
            constant_cols.append(col)
            num_constant = num_constant+1
            
    print('Number of Constant Variables:', num_constant)
    print(constant_cols)
    df = df.drop(constant_cols, axis=1)
    print('Shape: ', df.shape)
    return df


# In[ ]:


ctrain_df = constant_process(train_df)
ctest_df = constant_process(test_df)


# In[ ]:


print('Unique Variables in Train:', ctrain_df['sessionId'].nunique())
print('Unique Variables in Test:', ctest_df['sessionId'].nunique())


# We found the 'sessionId' exists duplicate which is wired because it is supposed to be unique as identifier.

# In[ ]:


dup_session = ctrain_df[ctrain_df.duplicated(subset='sessionId', keep=False)].sort_values('sessionId',ascending = False)
dup_session.head(2)


# The rest of columns with missing values are trafficSource.keyword and trafficSource.referralPath.

# # Numerical Variables Processing
# Considering the 'totals' of nunerical variables, we convert them into numerical type of float and replace the NAs in 'totals.transactionRevenue' with 0.

# In[ ]:


ctrain_df["totals.transactionRevenue"].fillna(0, inplace=True)
ctrain_df['totals.transactionRevenue'] = ctrain_df['totals.transactionRevenue'].astype(int)
ctrain_df['totals.hits'] = ctrain_df['totals.hits'].astype(int)
ctrain_df['totals.pageviews'] = ctrain_df['totals.hits'].astype(int)
ctest_df['totals.hits'] = ctest_df['totals.hits'].astype(int)
ctest_df['totals.pageviews'] = ctest_df['totals.hits'].astype(int)


# # Numerical Feature Distribution

# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(1,2,2)
ax = sns.distplot(np.log1p(ctrain_df[ctrain_df['totals.transactionRevenue'] > 0]["totals.transactionRevenue"]), kde=True)
ax.set_xlabel('Transaction Revenue Log', fontsize=15)
ax.set_ylabel('Distribuition', fontsize=15)
ax.set_title("Distribuition of Revenue Log", fontsize=20)
plt.subplot(1,2,1)
sns.distplot(ctrain_df["totals.transactionRevenue"], kde=True)
plt.xlabel('Transaction Revenue', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Revenue", fontsize=20)


# The Revenue is typically long-tail distributed but the effective revenue which is greater than 0 is approximately norally distributed. We also check the missing values in the valid Revenue rows as follow.

# In[ ]:


valid_df = ctrain_df[ctrain_df['totals.transactionRevenue'] > 0]
na_detect(valid_df)


# We remove the columns with over 95% missing values.

# In[ ]:


ctrain_df = ctrain_df.drop(['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot'], axis=1)
print('Train Shape: ' ,ctrain_df.shape)
ctest_df = ctest_df.drop(['trafficSource.adContent', 'trafficSource.adwordsClickInfo.adNetworkType', 'trafficSource.adwordsClickInfo.gclId', 'trafficSource.adwordsClickInfo.page', 'trafficSource.adwordsClickInfo.slot'], axis=1)
print('Test Shape:' ,ctest_df.shape)


# In[ ]:


plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
sns.distplot(ctrain_df["totals.hits"], kde=True)
plt.xlabel('Hits', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Hits", fontsize=20)
plt.subplot(1,2,2)
sns.distplot(ctrain_df["totals.pageviews"], kde=True)
plt.xlabel('Page Views', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Page Views", fontsize=20)


# According  to the long-tail distribution of 'totals.pageviews', we replace the missing values with medians. 

# In[ ]:


ctrain_df["totals.pageviews"].fillna(value=ctrain_df['totals.pageviews'].median(), inplace=True)
ctest_df["totals.pageviews"].fillna(value=ctest_df['totals.pageviews'].median(), inplace=True)


# In[ ]:


plt.figure(figsize=(10,5))
sns.distplot(ctrain_df["visitNumber"], kde=True)
plt.xlabel('Visit Number', fontsize=15)
plt.ylabel('Distribuition', fontsize=15)
plt.title("Distribuition of Visit Number", fontsize=20)


# + In addition, all numerical features including visitNumber, pageviews, and hits are measured in same scale level. Therefore, we don't need to normalize them.
# + All numerical variables we concern are long-tail distributed and continuous, therefore we can't use Correlation Coefficient to measure the correlations. Therefore, we decide to build a baseline tree-based model to measure the feature importance.
# 

# # Discrete Variables Processing - One Hot Encoding

# In[ ]:


non_relevant = ["date", "fullVisitorId", "sessionId", "visitId", "visitStartTime", "visitdate", "totals.transactionRevenue"]


# We use Label Encoding to save memory because of the tree-based models.

# In[ ]:


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

categorical_cols = [c for c in ctrain_df.columns if not c.startswith("total")]
categorical_cols = [c for c in categorical_cols if c not in non_relevant]
for c in categorical_cols:

    le = LabelEncoder()
    train_vals = list(ctrain_df[c].values.astype(str))
    test_vals = list(ctest_df[c].values.astype(str))
    
    le.fit(train_vals + test_vals)
    
    ctrain_df[c] = le.transform(train_vals)
    ctest_df[c] = le.transform(test_vals)


# # Response Variable

# In[ ]:


train_y = ctrain_df['totals.transactionRevenue']
del ctrain_df['totals.transactionRevenue']


# # Cross Validation 

# In[ ]:


def get_folds(df=None, n_splits=5):
    unique_sessions = np.array(sorted(df['sessionId'].unique()))
    folds = GroupKFold(n_splits=n_splits)
    fold_ids = []
    ids = np.arange(df.shape[0])
    for dev_s, val_s in folds.split(X=unique_sessions, y=unique_sessions, groups=unique_sessions):
        fold_ids.append(
            [
                ids[df['sessionId'].isin(unique_sessions[dev_s])],
                ids[df['sessionId'].isin(unique_sessions[val_s])]
            ]
        )

    return fold_ids


# In[ ]:


import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import GroupKFold
from sklearn import metrics


# # LightBM

# In[ ]:


get_ipython().run_cell_magic('time', '', "features = [f for f in ctrain_df.columns if f not in non_relevant]\nprint(features)\n\nfolds = get_folds(df=ctrain_df, n_splits=5)\n\nimportances = pd.DataFrame()\ndev_reg_preds = np.zeros(ctrain_df.shape[0])\nval_reg_preds = np.zeros(ctest_df.shape[0])\n\nfor f, (dev, val) in enumerate(folds):\n    dev_x, dev_y = ctrain_df[features].iloc[dev], train_y.iloc[dev]\n    val_x, val_y = ctrain_df[features].iloc[val], train_y.iloc[val]\n    \n    reg = lgb.LGBMRegressor(\n        num_leaves=31,\n        learning_rate=0.03,\n        n_estimators=1000,\n        subsample=.9,\n        colsample_bytree=.9,\n        random_state=1\n    )\n    \n    reg.fit(\n        dev_x, np.log1p(dev_y),\n        eval_set=[(val_x, np.log1p(val_y))],\n        early_stopping_rounds=50,\n        verbose=100,\n        eval_metric='rmse'\n    )\n    \n    importance_df = pd.DataFrame()\n    importance_df['feature'] = features\n    importance_df['gain'] = reg.booster_.feature_importance(importance_type='gain')\n    importance_df['fold'] = f + 1\n    importances = pd.concat([importances, importance_df], axis=0, sort=False)\n    dev_reg_preds[val] = reg.predict(val_x, num_iteration=reg.best_iteration_)\n    dev_reg_preds[dev_reg_preds < 0] = 0\n    preds = reg.predict(ctest_df[features], num_iteration=reg.best_iteration_)\n    preds[preds < 0] = 0\n    val_reg_preds += np.expm1(preds)/len(folds)\nprint('RMSE=' ,metrics.mean_squared_error(np.log1p(train_y), dev_reg_preds) ** .5)")


# In[ ]:


val_reg_preds.shape


# # Feature Importance

# In[ ]:


import warnings
warnings.simplefilter('ignore', FutureWarning)

importances['gain_log'] = np.log1p(importances['gain'])
mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])

plt.figure(figsize=(8, 12))
sns.barplot(x='gain_log', y='feature', data=importances.sort_values('mean_gain', ascending=False))


# # Prediction and Submission

# In[ ]:


ctest_df["PredictedLogRevenue"] = val_reg_preds
submission = ctest_df.groupby("fullVisitorId").agg({"PredictedLogRevenue" : "sum"}).reset_index()
submission["PredictedLogRevenue"] = np.log1p(submission["PredictedLogRevenue"])
submission["PredictedLogRevenue"] =  submission["PredictedLogRevenue"].apply(lambda x : 0.0 if x < 0 else x)
submission.to_csv("baseline.csv", index=False)
submission.head()

