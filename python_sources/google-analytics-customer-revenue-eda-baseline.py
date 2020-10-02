#!/usr/bin/env python
# coding: utf-8

# # Given that the data has some json fields , lets use a json function 

# In[ ]:


import os
import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(200,200)})
sns.set(font_scale=3)

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


# # Lets load the data and peek through

# In[ ]:


get_ipython().run_cell_magic('time', '', 'train = load_df()\ntest = load_df("../input/test.csv")')


# In[ ]:


train.describe(include='all')


# In[ ]:


train.head()


# In[ ]:


print(len(train.columns))
print(train.columns)


# # Lets look into the total trasactions per vistor 

# In[ ]:


sns.set(rc={'figure.figsize':(50,50)})


# In[ ]:


train["totals.transactionRevenue"] = train["totals.transactionRevenue"].astype('float')
total_rev_by_visid = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

plt.figure(figsize=(8,6))
plt.scatter(range(total_rev_by_visid.shape[0]), np.sort(np.log1p(total_rev_by_visid["totals.transactionRevenue"].values)))
plt.xlabel('index', fontsize=12)
plt.ylabel('TransactionRevenue', fontsize=12)
plt.show()


# ### The above plots proves us that the most transaction value comes from only few of users

# In[ ]:


nzi = pd.notnull(train["totals.transactionRevenue"]).sum()
nzr = (total_rev_by_visid["totals.transactionRevenue"]>0).sum()
print("Number of instances in train set with non-zero revenue : ", nzi, " and ratio is : ", nzi / train.shape[0] * 100 , "%")
print("Number of unique customers with non-zero revenue : ", nzr, "and the ratio is : ", nzr / total_rev_by_visid.shape[0] * 100 , "%")


# In[ ]:


print("Number of unique visitors in train set : ",train.fullVisitorId.nunique(), " out of rows : ",train.shape[0])
print("Number of unique visitors in test set : ",test.fullVisitorId.nunique(), " out of rows : ",test.shape[0])
print("Number of common visitors in train and test set : ",len(set(train.fullVisitorId.unique()).intersection(set(test.fullVisitorId.unique())) ))


# In[ ]:


const_cols = [c for c in train.columns if train[c].nunique(dropna=False)==1 ]
const_cols


# In[ ]:


imp_columns = set(train.columns.tolist()) - set(const_cols)


# In[ ]:


imp_columns


# # Lets look at each of above variables and their relation to total trasaction values

# ## Lets look into the page views vs total transaction

# In[ ]:


np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# In[ ]:


train["totals.pageviews"] = train["totals.pageviews"].astype("float")
total_rev_by_visid = train.groupby("totals.pageviews")["totals.transactionRevenue"].agg(["count","sum","mean","size"])
total_rev_by_visid.reset_index(inplace=True)
total_rev_by_visid["bin"] = total_rev_by_visid["totals.pageviews"] // int(5)


# 
# ### Lets look at how each bin of page views contribute to mean total transaction value

# In[ ]:


sns.set(font_scale=3)
total_rev_by_pageview_bin = total_rev_by_visid.groupby("bin")["mean"].agg(["mean"])
total_rev_by_pageview_bin.reset_index(inplace=True)
total_rev_by_pageview_bin["bin"] = total_rev_by_pageview_bin["bin"].astype(int)
sns.barplot(x=total_rev_by_pageview_bin["bin"],y=total_rev_by_pageview_bin["mean"])


# ### Relation between bins of page views and sum of their transaction value

# In[ ]:


total_rev_by_pageview_bin = total_rev_by_visid.groupby("bin")["sum"].agg(["sum"])
total_rev_by_pageview_bin.reset_index(inplace=True)
total_rev_by_pageview_bin["bin"] = total_rev_by_pageview_bin["bin"].astype(int)
# total_rev_by_pageview_bin.plot(kind="bar")
sns.barplot(x=total_rev_by_pageview_bin["bin"],y=total_rev_by_pageview_bin["sum"])


# In[ ]:


total_rev_by_pageview_bin = total_rev_by_visid.groupby("bin")["count"].agg(["sum"])
total_rev_by_pageview_bin.reset_index(inplace=True)
total_rev_by_pageview_bin["bin"] = total_rev_by_pageview_bin["bin"].astype(int)
# total_rev_by_pageview_bin.plot(kind="bar")
sns.barplot(x=total_rev_by_pageview_bin["bin"],y=total_rev_by_pageview_bin["sum"])


# ### Looking at the above plots the number of pages views has a influance in terms of total trasaction value , Lets move further

# ## Lets look at the "channelGrouping" column 

# In[ ]:


train["channelGrouping"].value_counts()


# In[ ]:


rev_by_channelgrouping = train.groupby("channelGrouping")["totals.transactionRevenue"].agg(["count","size","mean","sum"])
rev_by_channelgrouping.reset_index(inplace=True)


# In[ ]:


sns.barplot(x=rev_by_channelgrouping["channelGrouping"],y=rev_by_channelgrouping["mean"])


# In[ ]:


sns.barplot(x=rev_by_channelgrouping["channelGrouping"],y=rev_by_channelgrouping["sum"])


# In[ ]:


sns.barplot(x=rev_by_channelgrouping["channelGrouping"],y=rev_by_channelgrouping["count"])


# ### Lets look at the same for just visitor with non zero revenue

# In[ ]:


total_rev_by_visid = train.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()


# In[ ]:


total_rev_by_visid_non_zero = total_rev_by_visid[total_rev_by_visid["totals.transactionRevenue"] > 0]


# In[ ]:


non_zero_rev_vistor_id = total_rev_by_visid_non_zero.fullVisitorId.tolist()


# In[ ]:


len(non_zero_rev_vistor_id)


# In[ ]:


rev_by_channelgrouping_non_zero_rev_vistors = train[train["fullVisitorId"].isin(non_zero_rev_vistor_id)].groupby("channelGrouping")["totals.transactionRevenue"].agg(["count","size","mean","sum"])
rev_by_channelgrouping_non_zero_rev_vistors.reset_index(inplace=True)


# In[ ]:


sns.barplot(x=rev_by_channelgrouping_non_zero_rev_vistors["channelGrouping"],y=rev_by_channelgrouping_non_zero_rev_vistors["mean"])


# ^^ The graph is similar compared to all visitors

# In[ ]:


sns.barplot(x=rev_by_channelgrouping_non_zero_rev_vistors["channelGrouping"],y=rev_by_channelgrouping_non_zero_rev_vistors["sum"])


# In[ ]:


sns.barplot(x=rev_by_channelgrouping_non_zero_rev_vistors["channelGrouping"],y=rev_by_channelgrouping_non_zero_rev_vistors["size"])


# ^^ The above charts show that there the total rev is dependent on the channel

# ## Lets Look at device related stats

# In[ ]:


transaction_rev_by_device_browser = train.groupby("device.browser")["totals.transactionRevenue"].agg(["sum","mean","count","size"]).reset_index()


# In[ ]:


transaction_rev_by_device_browser_non_zero_sum = transaction_rev_by_device_browser[transaction_rev_by_device_browser["sum"] > 0].sort_values(by=["sum","mean","count","size"])


# In[ ]:


transaction_rev_by_device_browser_non_zero_sum["buy_ratio"] = transaction_rev_by_device_browser_non_zero_sum["count"].astype("float") * 100 / transaction_rev_by_device_browser_non_zero_sum["size"].astype("float")


# In[ ]:


transaction_rev_by_device_browser_non_zero_sum


# ^^ This looks very interesting , the sum of transaction for chrome is higher and the buy ratio is also higher 

# In[ ]:


imp_columns


# ### Baseline 

# Now let us build a baseline model on this dataset. Before we start building models, let us look at the variable names which are there in train dataset and not in test dataset.

# In[ ]:


print("Variables not in test but in train : ", set(train.columns).difference(set(test.columns)))


# 
# 
# So apart from target variable, there is one more variable "trafficSource.campaignCode" not present in test dataset. So we need to remove this variable while building models. Also we can drop the constant variables which we got earlier.
# 
# Also we can remove the "sessionId" as it is a unique identifier of the visit.
# 

# In[ ]:


cols_to_drop = const_cols + ['sessionId']

train = train.drop(cols_to_drop + ["trafficSource.campaignCode"], axis=1)
test = test.drop(cols_to_drop, axis=1)


# In[ ]:


train['date'] = pd.to_datetime(train['date'],format="%Y%m%d")


# 
# 
# Now let us create development and validation splits based on time to build the model. We can take the last two months as validation sample.
# 

# In[ ]:


# Impute 0 for missing target values
import datetime
from sklearn import preprocessing
train["totals.transactionRevenue"].fillna(0, inplace=True)
train_y = train["totals.transactionRevenue"].values
train_id = train["fullVisitorId"].values
test_id = test["fullVisitorId"].values


# label encode the categorical variables and convert the numerical variables to float
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
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = lbl.transform(list(train[col].values.astype('str')))
    test[col] = lbl.transform(list(test[col].values.astype('str')))


num_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
for col in num_cols:
    train[col] = train[col].astype(float)
    test[col] = test[col].astype(float)

# Split the train dataset into development and valid based on time 
dev_df = train[train['date']<=datetime.date(2017,5,31)]
val_df = train[train['date']>datetime.date(2017,5,31)]
dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
val_y = np.log1p(val_df["totals.transactionRevenue"].values)

dev_X = dev_df[cat_cols + num_cols] 
val_X = val_df[cat_cols + num_cols] 
test_X = test[cat_cols + num_cols] 


# In[ ]:


# custom function to run light gbm model
import lightgbm as lgb
def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {
        "objective" : "regression",
        "metric" : "rmse", 
        "num_leaves" : 30,
        "min_child_samples" : 100,
        "learning_rate" : 0.1,
        "bagging_fraction" : 0.7,
        "feature_fraction" : 0.5,
        "bagging_frequency" : 5,
        "bagging_seed" : 2018,
        "verbosity" : -1
    }
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000, valid_sets=[lgval], early_stopping_rounds=100, verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)


# Now let us compute the evaluation metric on the validation data as mentioned in [this](https://www.kaggle.com/c/ga-customer-revenue-prediction/discussion/66737) new discussion thread. So we need to do a sum for all the transactions of the user and then do a log transformation on top. Let us also make the values less than 0 to 0 as transaction revenue can only be 0 or more.

# In[ ]:


from sklearn import metrics
pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val)
#print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))
val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
print(np.sqrt(metrics.mean_squared_error(np.log1p(val_pred_df["transactionRevenue"].values), np.log1p(val_pred_df["PredictedRevenue"].values))))


# Now let us prepare the submission file similar to validation set.

# In[ ]:


sub_df = pd.DataFrame({"fullVisitorId":test_id})
pred_test[pred_test<0] = 0
sub_df["PredictedLogRevenue"] = np.expm1(pred_test)
sub_df = sub_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
sub_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
sub_df["PredictedLogRevenue"] = np.log1p(sub_df["PredictedLogRevenue"])
sub_df.to_csv("baseline_lgb.csv", index=False)


# In[ ]:





# In[ ]:





# In[ ]:




