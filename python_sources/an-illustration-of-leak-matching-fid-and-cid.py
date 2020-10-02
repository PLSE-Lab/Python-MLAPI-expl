#!/usr/bin/env python
# coding: utf-8

# ### This kernel is a simple illustration of the previous data leak, forking @Ankit's kernel "Story Of A Leak".

# We can download some leaked data from demo account of Google Merchandise Account (see [here](https://support.google.com/analytics/answer/6367342?hl=en)). In particular, we can find from User Explorer the purchase history of any user (identified by *Client ID*, or *cid*), but here will simply use @Ankit's data  (with a note that it has a problem to be addressed later).
# 
# Let's load the external 'leaked' data. We will use only the train set for illusration.

# In[ ]:


import os
import json
import pandas as pd
from pandas.io.json import json_normalize

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

# Keep only a subset of columns for illustration
df_train = train[['fullVisitorId','date','sessionId','visitStartTime','totals.transactionRevenue']]

#Load external data
train_store_1 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
train_store_2 = pd.read_csv('../input/exported-google-analytics-data/Train_external_data_2.csv', low_memory=False, skiprows=6, dtype={"Client Id":'str'})
df_extra = pd.concat([train_store_1, train_store_2], sort=False)


# `df_train` is our training set, and `df_extra` is the external data corresponding to training set's time period.
# 
# Let's take a look at the data first.

# In[ ]:


df_train.head()


# In[ ]:


df_extra.head()


# The part after `_` in `sessionId` in `df_train` is called `visitId`, which is the timestamp of the 1st session of a user. This `visitId` also shows up in  `Client ID`, after a dot `.`.
# 
# A few notes:
# 1. The part before the `_` in `sessionId` is `fullVisitorId`, or `fid`. We will use this terminology onward.
# 2. In general, `Client ID` can take any format, but by default it is a numerical string (randomly generated to distinguish a cookie) and`visitId` joined by dot in between.
# 3. Since `visitId` is a timestamp down to second precision, it is rare to see two sessions having same `visitId`. When this happens, we can further cross-check other information of the session, e.g. PV, location etc.
# 
# Therefore, given a `sessionId`, we can split it to get the `fullVisitorId` and `visitId`, where `visitId` can be referenced to the `Client ID` in external data, giving us the revenue of a `fullVisitorId`.
# 
# Reference [here](https://support.google.com/analytics/answer/3437719?hl=en).
# 
# 
# Lets extract `visitId` from both tables.

# In[ ]:


def get_visitId_by_cid(cid):
    return cid.split('.')[1]
def get_visitId_by_sid(sid):
    return sid.split('_')[1]


# In[ ]:


df_train['visitId'] = df_train['sessionId'].apply(get_visitId_by_sid)
df_extra['visitId'] = df_extra['Client Id'].apply(get_visitId_by_cid)


# In[ ]:


df_train.head()


# In[ ]:


df_extra.head()


# Joining these tables together by `visitId`...

# In[ ]:


df_joint = df_train.merge(df_extra, how='inner', left_on='visitId', right_on='visitId')


# In[ ]:


df_joint.head(10)


# You can see that lots of sessions do not have any revenue in train set, but external data says otherwise. Part of the reason is the train data is on session level but external data is on user level, so I should have group-sum the revenue by `fullVisitorId` before joining. Another possibility is the organiser filtered out some  purchases, or multiple users share the same `visitId`, or there exists other problems in our method. But you get the idea. 
# 
# Another puzzle is that the revenue from two sides do not match. Let's take a closer look:

# In[ ]:


df_joint['totals.transactionRevenue'] = pd.to_numeric(df_joint['totals.transactionRevenue']) / 10**6  # convert to float USD. Should have done earlier.


# In[ ]:


# Truncate the table for better view
df_trun = df_joint[df_joint['totals.transactionRevenue'] > 0.0][['date', 'Client Id', 'fullVisitorId', 'visitId', 'totals.transactionRevenue', 'Revenue', 'Transactions']]  # take a look at all rows with non-zero revenue 


# In[ ]:


df_trun.head()


# Notice that `Revenue` looks close to the double of `totals.transactionRevenue`,  and a lot of `Transactions` are taking value of 2 (even numbers). In fact, this is a well-know issue of GA, where sampling is used if you query a lot of data at same time. This problem affects @Ankit's data too. You can verify this by searching for a Client ID in the User Exploer view, for example `446817349.1472832496` on Sep 2nd 2016, and it will show the correct, unsampled value ($ 117.34). 
# 
# Also note that we did not touch the topic of shipping cost and tax, and they may (not) affect the revenues in the two tables. I will leave you to explore.
# 
# That's it. Hope you like it.
