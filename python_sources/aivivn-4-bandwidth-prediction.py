#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **LOAD DATA**

# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test_id.csv')

print(train.shape)
print(test.shape)


# In[3]:


train['UPDATE_TIME'] = train['UPDATE_TIME'].astype('datetime64[ns]')
test['UPDATE_TIME'] = test['UPDATE_TIME'].astype('datetime64[ns]')

train.drop(['ZONE_CODE'], axis=1, inplace=True)
test.drop(['ZONE_CODE'], axis=1, inplace=True)

train.head()


# **PREPROCESS**

# In[12]:


train_df= train.loc[train.UPDATE_TIME >= '2019-03-02']

groupby_min = train_df.groupby(['SERVER_NAME','HOUR_ID']).min().reset_index()
groupby_max = train_df.groupby(['SERVER_NAME','HOUR_ID']).max().reset_index()
groupby_mean = train_df.groupby(['SERVER_NAME','HOUR_ID']).mean().reset_index()

groupby_mean['mape_bandwidth'] = (groupby_max['BANDWIDTH_TOTAL'] - groupby_min['BANDWIDTH_TOTAL'])/groupby_min['BANDWIDTH_TOTAL'] * 100
groupby_mean['mape_user'] = (groupby_max['MAX_USER'] - groupby_min['MAX_USER'])/groupby_min['MAX_USER'] * 100
groupby_mean['min_of_BANDWIDTH'] = groupby_min['BANDWIDTH_TOTAL']
groupby_mean['min_of_MAXUSER'] = groupby_min['MAX_USER']
groupby_mean.head()


# In[13]:


valid = train_df.drop(['BANDWIDTH_TOTAL', 'MAX_USER'], axis=1)
valid = valid.join(groupby_mean.set_index(['SERVER_NAME','HOUR_ID']),
                     on=['SERVER_NAME','HOUR_ID'])
test_df = test.join(groupby_mean.set_index(['SERVER_NAME','HOUR_ID']),
                     on=['SERVER_NAME','HOUR_ID'])

valid.head()


# **APPLY THE TRICK**

# In[14]:


THRESHOLD = 100

valid.loc[(valid["mape_bandwidth"] > THRESHOLD) & (valid["min_of_BANDWIDTH"] < 200),
          "BANDWIDTH_TOTAL"] = np.nan
test_df.loc[(test_df["mape_bandwidth"] > THRESHOLD) & (test_df["min_of_BANDWIDTH"] < 200),
            "BANDWIDTH_TOTAL"] = np.nan

print(test_df['BANDWIDTH_TOTAL'].describe())


# In[15]:


valid.loc[(valid["mape_user"] > THRESHOLD) & (valid["min_of_MAXUSER"] < 200),
          "MAX_USER"] = np.nan
test_df.loc[(test_df["mape_user"] > THRESHOLD) & (valid["min_of_MAXUSER"] < 200),
            "MAX_USER"] = np.nan

print(test_df['MAX_USER'].describe())


# In[16]:


valid['MAX_USER'].fillna(0, inplace=True)
valid['BANDWIDTH_TOTAL'].fillna(0, inplace=True)
test_df['MAX_USER'].fillna(0, inplace=True)
test_df['BANDWIDTH_TOTAL'].fillna(0, inplace=True)
print(test_df.shape)


# In[17]:


def MAPE(y_true, y_pred):
    error = np.abs(y_true - y_pred)/ y_true
    error.replace([np.inf, -np.inf], np.nan, inplace=True)
    error.dropna(inplace=True)
    return np.mean(error)*100

bandwidth_mape = MAPE(train_df['BANDWIDTH_TOTAL'], valid['BANDWIDTH_TOTAL'])
user_mape = MAPE(train_df['MAX_USER'], valid['MAX_USER'])

print('MAPE bandwidth : ', bandwidth_mape)
print('MAPE user : ', user_mape)
print('MAPE total: ', bandwidth_mape*0.8 + user_mape*0.2)


# In[18]:


test_df['MAX_USER'] = test_df.MAX_USER.astype(int).astype(str)
test_df['BANDWIDTH_TOTAL'] = test_df.BANDWIDTH_TOTAL.round(2).astype(str)
test_df['label'] = test_df['BANDWIDTH_TOTAL'].str.cat(test_df['MAX_USER'],sep=" ")

test_df[['id','label']].to_csv('sub_aivn.csv', index=False)
test_df.head()

