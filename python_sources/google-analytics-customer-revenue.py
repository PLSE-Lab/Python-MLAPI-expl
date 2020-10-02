#!/usr/bin/env python
# coding: utf-8

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


# Function to load json files.. taken from https://www.kaggle.com/julian3833/1-quick-start-read-csv-and-flatten-json-fields/notebook

# In[ ]:


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Not sure how we got the input files here... will revert later

# In[ ]:


import pandas as pd
import numpy as np
import json
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import datetime as datetime
from sklearn.preprocessing import Imputer


# Import a lot of stuff. I don't know how to use most of these.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_data = load_df()\ntest_data = load_df("../input/test.csv")\ntrain_data.head()\ntest_data.head()')


# Shows us some data- some of these in the curly brackets look useless. And by that I mean, I don't know how to use them.

# In[ ]:


train_data.describe()


# Any way to show these in appropriate units?

# In[ ]:


list(train_data.columns.values)


# **Features descriptions**:
# 
# Returning back to Data description for understanding features.
# 
# *     channelGrouping - The channel via which the user came to the Store.
# *     date - The date on which the user visited the Store.
# *     geoNetwork - This section contains information about the geography of the user.
# *     sessionId - A unique identifier for this visit to the store.
# *     socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
# *     totals - This section contains aggregate values across the session.
# *     trafficSource - This section contains information about the Traffic Source from which the session originated.
# *     visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
# *     visitNumber - The session number for this user. If this is the first session, then this is set to 1.
# *     visitStartTime - The timestamp (expressed as POSIX time).
# 

# **1. channelGrouping** - the channel via which the user came to the store. By far the largest was organic search. I don't know what social or direct means anyways, so who knows how useful this will be.

# In[ ]:


(train_data['channelGrouping'].value_counts()).plot(kind='bar')


# **2. date and visitStartTime**
# 
# 

# In[ ]:


train_data['date'] = pd.to_datetime(train_data['date'], format="%Y%m%d")
(train_data['date'].value_counts()).plot()


# Spikes during holiday seasons to no surprise... also random spikes during summer? Hopefully will add in more granular labels.

# **3. socialEngagementType**
# This is useless.. no one is socially engaged.

# In[ ]:


(train_data['socialEngagementType'].value_counts()).plot(kind ='bar')
plt.xticks(rotation='horizontal')


# **4. visitNumber**
# 
# Almost no one visits more than once!

# In[ ]:


(train_data['visitNumber'].value_counts()).plot()


# **5. Devices**

# In[ ]:


train_data.groupby('device.browser')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('device.deviceCategory')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('device.operatingSystem')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')


# **6. Traffic Source**

# In[ ]:


train_data.groupby('trafficSource.source')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('trafficSource.medium')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')


# **7. Continent**

# In[ ]:


train_data.groupby('geoNetwork.continent')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('geoNetwork.subContinent')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')
train_data.groupby('geoNetwork.networkDomain')['totals.transactionRevenue'].agg(['mean']).sort_values(by="mean", ascending=True).head(10).plot(kind = 'barh')


# **8. Unique customers**

# In[ ]:


train_data["totals.transactionRevenue"] = train_data["totals.transactionRevenue"].astype('float')
gdf = train_data.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()
Nonzero_instance = pd.notnull(train_data["totals.transactionRevenue"]).sum()
Unique_customer = (gdf["totals.transactionRevenue"] > 0).sum()

print("There were", Nonzero_instance, "instances of activities that involved non-zero revenues.")
print("The ratio of activities that involved non-zero revenues to total activities was", '{:.2%}'.format(Nonzero_instance/train_data.shape[0]))
print()
print("There were", Unique_customer, "instances of activities that involved non-zero revenues from unique customers.")
print("The ratio of unique activities that involved non-zero revenues to unique total activities was", '{:.2%}'.format(Unique_customer/gdf.shape[0]))


# This means that the vast majority of predicted values should be 0. The 1% will feature heavily in the analysis of this dataset. Let's take a look at the training set and the test set, just from the perspective of unique visitors.

# In[ ]:


print("In the train set, there were", train_data.fullVisitorId.nunique(), "unique visitors. There were", train_data.shape[0], "total non-unique visitors.")
print("In the test set, there were", test_data.fullVisitorId.nunique(), "unique visitors. There were", test_data.shape[0], "total non-unique visitors.")
print("There were", len(set(train_data.fullVisitorId.unique()).intersection(set(test_data.fullVisitorId.unique()))), "common visitors in the two data sets.")


# Why don't we just set these common visitors to what they were in the training set? It may actually be a good idea to see if non-unique visitors change their spending habits over time. Two hypotheses:
# 1) If they visit more than once, they are much more likely to spend money on the second time around.
# 2) If they spent the first time, they will also spend the second time. The likelihood of spending the first time and not spending the second time is near 0.

# **II. Preprocessing**
# 

# In[ ]:


total = train_data.isnull().sum().sort_values(ascending=False)
percent = total/train_data.shape[0]
Null_df = pd.concat([total, percent], axis = 1, keys = ['total', 'percent'])
Null_df[:20]
train_data


# In[ ]:


train_data['totals.newVisits'] = train_data['totals.newVisits'].fillna(0)
train_data['totals.bounces'] = train_data['totals.bounces'].fillna(0)
train_data['totals.newVisits'] = train_data['totals.newVisits'].astype(int)
train_data['totals.bounces'] = train_data['totals.bounces'].astype(int)
train_data['totals.pageviews'].fillna(1, inplace=True)

train_data


# In[ ]:


train_data['totals.hits'] = train_data['totals.hits'].astype(float)
train_data['totals.transactionRevenue'] = train_data['totals.transactionRevenue'].astype(float)
train_data['totals.transactionRevenue'].fillna(0.0,inplace=True)
train_data['totals.transactionRevenue_log'] = (np.log(train_data[train_data['totals.transactionRevenue']>0]["totals.transactionRevenue"]))
train_data['totals.transactionRevenue_log'].fillna(0, inplace=True)


# **III. Regression**
# 
# Below regression borrows heavily from: https://www.kaggle.com/vik1511/a-simple-data-visualization-lgbm

# In[ ]:


train_x = train_data.drop(['date','fullVisitorId','sessionId','visitId','totals.transactionRevenue_log'],axis=1)
train_y = train_data['totals.transactionRevenue_log']


# In[ ]:


train_x['totals.pageviews'] = train_x['totals.pageviews'].astype(float)
categorical_col = train_x.select_dtypes(include = [np.object]).columns
numerical_col = train_x.select_dtypes(include = [np.number]).columns
categorical_col, numerical_col


# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in categorical_col:
    train_col = list(train_x[col].values.astype(str))
    le.fit(train_col)
    train_x[col] = le.transform(train_col)


# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
for col in numerical_col :
    train_x[col] = (train_x[col] - np.mean(train_x[col])/(np.max(train_x[col]) - np.min(train_x[col])))


# In[ ]:


from sklearn.model_selection import train_test_split
trainX, crossX, trainY, crossY = train_test_split(train_x.values, train_y, test_size = 0.25, random_state = 20)
trainY = trainY.fillna(0)
trainY


# In[ ]:


from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
trainX.isnull()

