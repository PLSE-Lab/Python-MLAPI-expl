#!/usr/bin/env python
# coding: utf-8

# *  we're predicting the natural log of the total revenue per unique user, which is, based on totals.transactionRevenue.  (Where a Nan is actually a 0).
# * We should log the sum total of *totals.transactionRevenue*
#      * https://www.kaggle.com/c/google-analytics-customer-revenue-prediction/discussion/65691#387112
# 
#  * https://www.kaggle.com/mlisovyi/flatten-json-fields-smart-dump-data
#  * https://www.kaggle.com/jpmiller/showing-nan-in-its-various-forms

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import pandas as pd
import json
# import missingno as msno
# import hvplot.pandas

PATH = '../input/'


# ### Data Prep
# 
# * Additional ideas for missing values and unary columns top drop:
#  https://www.kaggle.com/mlisovyi/flatten-json-fields-smart-dump-data
#  * https://www.kaggle.com/c/google-analytics-customer-revenue-prediction/discussion/65691#387112

# In[ ]:


json_cols = ['device', 'geoNetwork', 'totals', 'trafficSource']

nan_list = ["not available in demo dataset",
            "unknown.unknown",
            "(not provided)",
            "(not set)"
#             ,"Not Socially Engaged" # this last one is borderline 
           ]
nan_dict = {nl:np.nan for nl in nan_list}

# columns to drop : https://www.kaggle.com/c/google-analytics-customer-revenue-prediction/discussion/65691#387112
list_single_value = ['trafficSource.campaignCode', 'socialEngagementType', 'totals.visits']

def df_prep(file):
    df = pd.read_csv(file, dtype={'fullVisitorId': str, 'date': str}, 
            parse_dates=['date'],infer_datetime_format=True, nrows=None)
    
    for jc in json_cols:  # parse json  # Would probably be better with json_normalize from pandas
        flat_df = pd.DataFrame(df.pop(jc).apply(pd.io.json.loads).values.tolist())
        flat_df.columns = ['{}.{}'.format(jc, c) for c in flat_df.columns]
        df = df.join(flat_df)
    ad_df = df.pop('trafficSource.adwordsClickInfo').apply(pd.Series) # handle dict column
    ad_df.columns = ['adwords.{}'.format(c) for c in ad_df.columns]
    df = df.join(ad_df)
    df.replace(nan_dict, inplace=True) # handle disguised NaNs
    
    # Remove all-missing columns
    df.dropna(how="all",axis=1,inplace=True)
    
    df.drop([c for c in list_single_value if c in df.columns], axis=1, inplace=True)
    
# ### From : https://www.kaggle.com/mlisovyi/flatten-json-fields-smart-dump-data
    df['trafficSource.isTrueDirect'] = (df['trafficSource.isTrueDirect'].fillna(False)).astype(bool)
    df['totals.bounces'] = df['totals.bounces'].fillna(0).astype(np.uint8)
    df['totals.newVisits'] = df['totals.newVisits'].fillna(0).astype(np.uint8) # has NaNs ?
    df['totals.pageviews'] = df['totals.pageviews'].fillna(0).astype(np.uint16)
    
    # rename lat Long
    df.rename(columns={'geoNetwork.latitude':'Latitude', 'geoNetwork.longitude':"Longitude"},inplace=True)

    #parse unix epoch timestamp
    df.visitStartTime = pd.to_datetime(df.visitStartTime,unit='s',infer_datetime_format=True)
    
#     df.set_index(['fullVisitorId', 'sessionId'], inplace=True) # disabled for now

    df.drop(["sessionId"],axis=1,inplace=True)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain = df_prep(PATH+\'train_v2.csv\')\nprint("train Shape: ",train.shape)\ntest = df_prep(PATH+\'test_v2.csv\')\nprint("test Shape: ",test.shape)\ndisplay(train.head(7))')


# In[ ]:


train.columns


# In[ ]:


train[['channelGrouping', 'date', 'fullVisitorId', 'visitId',
       'visitNumber', 'visitStartTime', 'device.browser',
       'device.deviceCategory', 'device.isMobile', 'device.operatingSystem',
       'geoNetwork.city', 'geoNetwork.continent', 'geoNetwork.country',
       'geoNetwork.metro', 'geoNetwork.networkDomain', 'geoNetwork.region',
       'geoNetwork.subContinent', 'totals.bounces', 'totals.hits',
       'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue',
        'trafficSource.adContent', 'trafficSource.campaign', 'trafficSource.isTrueDirect',
       'trafficSource.keyword', 'trafficSource.medium',
       'trafficSource.referralPath', 'trafficSource.source', 'adwords.page',
       'adwords.slot', 'adwords.gclId', 'adwords.adNetworkType']].nunique()


# In[ ]:


### Many variables only contain a single variable, remove them:
### change code version ; errors due to unhashable dicts
# columns = [col for col in train.columns if train[col].nunique() > 1] # can also be done with ".any() command"
# print(len(columns))
# train = train[columns]
# test = test[columns]


# In[ ]:


train.visitStartTime.describe()


# ## Target col: 
# * We will want to sum then log at the end, (if we do it once per customer, VS predicting CLV at each point in time..?)
# * totals_transactionRevenue - nan is actually 0 
# * **Major, novel  feature: Transactions per session (mean and boolean)**
#     * Dan
#     
# * WE see most visitors never make any purchase

# In[ ]:


#impute 0 for missing/NaNs of target column
train['totals.transactionRevenue'] = pd.to_numeric(train['totals.transactionRevenue'].fillna(0)) #.astype("float")


# In[ ]:


train['totals.transactionRevenue'].dtype


# In[ ]:


train.loc[train['totals.transactionRevenue']>0]['totals.transactionRevenue'].describe()


# In[ ]:


## https://www.kaggle.com/ashishpatel26/light-gbm-with-bayesian-style-parameter-tuning

gdf = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(9,7))
plt.scatter(range(gdf.shape[0]), np.sort(np.log1p(gdf["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()


# In[ ]:


print("Train set: shape {} with {} unique users"
      .format(train.shape,train['fullVisitorId'].nunique()))
print("Test set: shape {} with {} unique users"
      .format(test.shape,test['fullVisitorId'].nunique()))
print("Users in both train and test set:",
      len(set(train.fullVisitorId.unique()).intersection(set(test.fullVisitorId.unique()))))


# #### Test data
# * we predict once per customer. (in test) , using the last entry
# * multiple rows present = we have more history for them -> concat with train data for history.
# * Note the lack of overlap with train -> we may want to entirely exclude the y/target from history, to avouid leaks! 

# In[ ]:


test.head()


# In[ ]:


print("orig test shape:",test.shape)
test_pred = test.set_index("visitStartTime",drop=False).groupby("fullVisitorId").last().reset_index().drop_duplicates("fullVisitorId")
print("pred ready test shape:",test_pred.shape)
test_pred.head()


# ### finals target data
# * start with data per user at their final time stamp. This isn't what we'd use for a final model but can give us great insights , and iscompatible with feature engineering for the historical data!
# * Get last timestamp for each fullVisitorId,  sum historical totals.transactionRevenue transactions, then log that sum.
#     * There's no history for ~80% of users (i.e most appear only once). 
#     * Also, user history - note train/test disjoint and sparsity!
#     
#     
#     * featurize: https://stackoverflow.com/questions/45022226/find-days-since-last-event-pandas-dataframe

# In[ ]:


# df2 = train.drop([#"date",
#                   "sessionId"
# #                   , "visitId" # ? 
#                  ],axis=1)

df2 = train.copy()

df2["sumLog_transactionRevenue"] = df2[["fullVisitorId","totals.transactionRevenue"]].groupby("fullVisitorId")["totals.transactionRevenue"].transform("sum")
# log transform target (we don't do log1P on purpose)! 
df2['sumLog_transactionRevenue'] = df2['sumLog_transactionRevenue'].apply(lambda x: np.log1p(x)) #.apply(lambda x: np.log(x) if x > 0 else x)
print("# unique visitor IDs : ", df2.fullVisitorId.nunique())
print("subset initial Data shape", df2.shape)
df2 = df2.set_index("visitStartTime",drop=False).groupby("fullVisitorId").last().drop("totals.transactionRevenue",axis=1).reset_index()
print("Data with target + only last entry in train per fullVisitorId:", df2.shape)
df2.tail()


# ## Concat context
# * train + test historical data
# *Could drop last entries in test for space saving.. But might be wanted for country level feature, cooccurrence etc'?
# 

# In[ ]:


df_context = pd.concat([train,test])
df_context.shape


# ## Save data
# * Could use Feather or binary format, but let's stay simple
# 

# In[ ]:


# is enabling INDEXes, then keep index!
df2.to_csv("gstore_train_CLV_v1.csv.gz",index=False,compression="gzip")
# train.to_csv("gstore_train_v1.csv.gz",index=False,compression="gzip")
# test.to_csv("gstore_test_v1.csv.gz",index=False,compression="gzip")

df_context.to_csv("gstore_context_all_v1.csv.gz",index=False,compression="gzip")
test_pred.to_csv("gstore_test_Pred_v1.csv.gz",index=False,compression="gzip")


# ## NaN EDA cont
# *Source:  https://www.kaggle.com/jsaguiar/complete-exploratory-analysis
#         * code requires changing (fullVisitorId in index in my version)
# *  'totals_transactionRevenue' column=  the transaction value for each visit. 
# *Train set has 98.72% of missing values which we can consider as zero revenue (no purchase).
# * The black lines are the closest normal distribution that we can fit to each distribution.

# In[ ]:


non_missing = len(train[~train['totals.transactionRevenue'].isnull()])
num_visitors = train[~train['totals.transactionRevenue'].isnull()]['fullVisitorId'].nunique()
print("totals.transactionRevenue has {} non-missing values or {:.3f}% (train set)"
      .format(non_missing, 100*non_missing/len(train)))
print("Only {} unique users have transactions or {:.3f}% (train set)"
      .format(num_visitors, num_visitors/train['fullVisitorId'].nunique()))
# Logn Distplot
revenue = train['totals.transactionRevenue'].dropna().astype('float64')
plt.figure(figsize=(10,4))
plt.title("Natural log Distribution - Transactions revenue")
ax1 = sns.distplot(np.log(revenue), color="#006633", fit=norm)
# Log10 Distplot
plt.figure(figsize=(10,4))
plt.title("Log10 Distribution - Transactions revenue")
ax1 = sns.distplot(np.log10(revenue), color="#006633", fit=norm)


# ### Transaction Revenue
# 
# Our target column, transactionRevenue, looks especially sparse. Let's look closer...

# In[ ]:


target_df = pd.read_csv('../input/train.csv', usecols=['totals'])
flat_df = pd.io.json.json_normalize(target_df.totals.apply(json.loads))
flat_df['transactionRevenue'] = flat_df.transactionRevenue.astype(np.float32)
flat_df.transactionRevenue.isnull().sum()/flat_df.shape[0]


# In[ ]:


flat_df.fillna(0, inplace=True)
flat_dft.hist('transactionRevenue', bins=24) #.hvplo


# Well, OK...there's quite a bit of 0s here.  Zooming in on the 1%  greater than 0 shows the difference between browsers and buyers.

# In[ ]:


flat_df.replace(0, np.NaN, inplace=True)
flat_df.hist('transactionRevenue', bins=25) #.hvplot

