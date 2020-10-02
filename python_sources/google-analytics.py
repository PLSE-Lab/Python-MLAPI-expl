#!/usr/bin/env python
# coding: utf-8

# [https://github.com/Shravankp/Emplay_kaggle_challenge/blob/master/customer_revenue_prediction.ipynb](http://)
# 
# Google Analytics Customer Revenue Prediction
# 
# To predict - How much GStore's Revenue from each customers, using EDA and ML models.
# 
# Given in problem description, the 80/20 rule - 80% of revenue comes from only 20% of the customers.So our task is to predict the Gstore's revenue from those customers (using natural log of sum of all transactions per user).
# 
# Features of the given dataset:
# 
# fullVisitorId- A unique identifier for each user of the Google Merchandise Store.
# channelGrouping - The channel via which the user came to the Store.
# date - The date on which the user visited the Store.
# device - The specifications for the device used to access the Store.
# geoNetwork - This section contains information about the geography of the user.
# sessionId - A unique identifier for this visit to the store.
# socialEngagementType - Engagement type, either "Socially Engaged" or "Not Socially Engaged".
# totals - This section contains aggregate values across the session.
# trafficSource - This section contains information about the Traffic Source from which the session originated.
# visitId - An identifier for this session. This is part of the value usually stored as the _utmb cookie. This is only unique to the user. For a completely unique ID, you should use a combination of fullVisitorId and visitId.
# visitNumber - The session number for this user. If this is the first session, then this is set to 1.
# visitStartTime - The timestamp (expressed as POSIX time).

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


import json
import numpy as np
import pandas as pd
from pandas.io.json import json_normalize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, preprocessing, metrics
import lightgbm as lgb


# Note: Some columns contain serialised JSON as strings which should be deserialised and converted to seperate columns.
# 
# 

# In[ ]:


def load_df(csv_path='./train.csv', nrows=None):
    JSON_COLUMNS = ['device', 'geoNetwork', 'totals', 'trafficSource']
    
    df = pd.read_csv(csv_path, 
                     converters={column: json.loads for column in JSON_COLUMNS}, #json.loads takes in a string and converts to dict or list object.
                     dtype={'fullVisitorId': 'str'}, #convert id to string
                     nrows=nrows)
    

    for column in JSON_COLUMNS:
        column_as_df = json_normalize(df[column])  #converts semi-structured json to flat table.
        column_as_df.columns = [f"{column}.{subcolumn}" for subcolumn in column_as_df.columns]
        df = df.drop(column, axis=1).merge(column_as_df, right_index=True, left_index=True)
    print(f"Shape: {df.shape}")
    return df


# In[ ]:


dftrain = load_df("../input/train.csv")


# In[ ]:


dftrain.head()


# In[ ]:


test_df = load_df("../input/test.csv")


# In[ ]:


test_df.head()


# Lets perform EDA:
# 
# Each row in the dataset is one visit to the store.
# 
# Let us look at what all are the numerical variables and categorical variables we have in train dataset.

# In[ ]:


numeric_features = dftrain.select_dtypes(include=[np.number])
print(numeric_features.columns)

categorical_features = dftrain.select_dtypes(include=[np.object])
print(categorical_features.columns)


# **Note**: There are some variables in categorical_features which can be converted to fload and can be used as numeric variables, such as ['totals.bounces', 'totals.hits', 'totals.newVisits', 'totals.pageviews', 'totals.transactionRevenue', 'totals.visits'].
# 
# Now lets check whether the given dataset conforms 80/20 rule :
# 
# * First convert values of totals.transactionRevenue to float
# * Group values according to fullVisitorId ( i.e we are calculating revenue from each customer )
# * Then consider only those customers who have revenue more than zero and find out the ratio.

# In[ ]:


dftrain["totals.transactionRevenue"] = dftrain["totals.transactionRevenue"].astype('float')
grouped_revenue = dftrain.groupby("fullVisitorId")["totals.transactionRevenue"].sum().reset_index()

non_zero_customers = (grouped_revenue["totals.transactionRevenue"]>0).sum()
print("Number of unique customers with non-zero revenue : ", non_zero_customers, "and the ratio is : ", non_zero_customers / grouped_revenue.shape[0])


# 
# Number of unique customers with non-zero revenue :  9996 and the ratio is :  0.013996726255903731
# So the ratio of revenue generating customers to total number of customers is 1.3%. From the above analysis it is confirmed that only 1.3% of the customers bring in revenue to Gstore.
# 
# **Data pre-processing:**
# 
# Some columns have constant values and missing values. So, Lets examine that and drop those columns from feature set which would make our dataset more useful while training models.

# In[ ]:


# convert the 'date' column values to datetime object
import datetime
dftrain['date'] = dftrain['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))
test_df['date'] = test_df['date'].apply(lambda x: datetime.date(int(str(x)[:4]), int(str(x)[4:6]), int(str(x)[6:])))


# In[ ]:


consts = [c for c in dftrain.columns if dftrain[c].nunique(dropna=False)==1 ] #lets include nan values in the count(nunique())
consts


# As we know that train set has 55 columns whereas test set has 53 ,lets examine what are those two extra variables in train set and which of those two can be target variable.

# In[ ]:


set(dftrain.columns).difference(set(test_df.columns))


# As 'trafficSource.campaignCode' is an extra feature not in test set (other than target variable 'totals.transactionRevenue' ).Lets drop this also. Even sessionId can be removed as this is just a unique number for each visit.
# 
# So lets drop consts (columns with constant values declared above), 'sessionId', 'trafficSource.campaignCode'.

# In[ ]:


cols_to_drop = consts + ["sessionId"] + ["trafficSource.campaignCode"]
dftrain = dftrain.drop(cols_to_drop, axis=1)
test_df = test_df.drop(cols_to_drop[:-1], axis=1)


# * Fill in missing values to 0.
# * Now, Identify categorical variables and convert to numbers i.e label encode them.
# * Identify numeric variables and convert them to floats.
# 
# **Note:** Do not include IDs and dates to any of the above operations.

# In[ ]:


#dftrain.head()
#dftrain.info()
#dftrain.describe()

dftrain["totals.transactionRevenue"].fillna(0, inplace=True)
train_y = dftrain["totals.transactionRevenue"].values

#identify categorical variables and label encode them.
categorical_cols = ["channelGrouping", "device.browser", 
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

for col in categorical_cols:
    print(col)
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(dftrain[col].values.astype('str')) + list(test_df[col].values.astype('str')))
    dftrain[col] = lbl.transform(list(dftrain[col].values.astype('str')))
    test_df[col] = lbl.transform(list(test_df[col].values.astype('str')))


numeric_cols = ["totals.hits", "totals.pageviews", "visitNumber", "visitStartTime", 'totals.bounces',  'totals.newVisits']    
for col in numeric_cols:
    dftrain[col] = dftrain[col].astype(float)
    test_df[col] = test_df[col].astype(float)


# As the training set contains data from August 1st 2016 to August 1st 2017. Then take cross-validation set as last three month's data which makes the ratio of train set to cross-val set roughly 7.5 : 2.5 .(This is considering months but not no of examples in trainset , so might not be exactly 7.5 : 2.5).

# In[ ]:


# Split the train dataset into development and valid based on time 
dev_df = dftrain[dftrain['date']<=datetime.date(2017,5,31)]
val_df = dftrain[dftrain['date']>datetime.date(2017,5,31)]

dev_y = np.log1p(dev_df["totals.transactionRevenue"].values)
val_y = np.log1p(val_df["totals.transactionRevenue"].values)

dev_X = dev_df[categorical_cols + numeric_cols] 
val_X = val_df[categorical_cols + numeric_cols] 
test_X = test_df[categorical_cols + numeric_cols]


# 
# Choosing a model to train:
# 
# As we have large dataset (with more columns >30 and rows > 10000)its better to use ensemble learning (gradient boosting models).we'll use LightGBM to train our model and predict revenue from test set.

# In[ ]:


def run_lgb(train_X, train_y, val_X, val_y, test_X):
    params = {"objective" : "regression","metric" : "rmse", 
            "subsample" : 0.9,"colsample_bytree" : 0.9,
            "num_leaves" : 31,"min_child_samples" : 100,
            "learning_rate" : 0.03,"bagging_fraction" : 0.7,
            "feature_fraction" : 0.5,"bagging_frequency" : 5,
            "bagging_seed" : 2018,"verbosity" : -1}
    
    lgtrain = lgb.Dataset(train_X, label=train_y)
    lgval = lgb.Dataset(val_X, label=val_y)
    model = lgb.train(params, lgtrain, 1000,early_stopping_rounds=100, valid_sets=[lgval],  verbose_eval=100)
    
    pred_test_y = model.predict(test_X, num_iteration=model.best_iteration)
    pred_val_y = model.predict(val_X, num_iteration=model.best_iteration)
    return pred_test_y, model, pred_val_y

# Training the model #
pred_test, model, pred_val = run_lgb(dev_X, dev_y, val_X, val_y, test_X)


# 
# Let us compute the evaluation metric on the validation data.
# 
# * Assign zero if predicted value is less than 0.
# * Prepare a validation dataframe. As predicted values are logarithmic values apply np.exp ( we'll convert this to log values again after grouping values acc to 'fullVisitorId' and calculating sum of revenue per user )
# * Apply log on sum for all the transactions per user (or apply sum first on grouped data and then apply log).
# * Calculate rms error.

# In[ ]:


from sklearn import metrics

pred_val[pred_val<0] = 0
val_pred_df = pd.DataFrame({"fullVisitorId":val_df["fullVisitorId"].values})
val_pred_df["transactionRevenue"] = val_df["totals.transactionRevenue"].values
val_pred_df["PredictedRevenue"] = np.expm1(pred_val) #exp(x) -1 can also be used but expm1 gives greater precision when converting log

val_pred_df = val_pred_df.groupby("fullVisitorId")["transactionRevenue", "PredictedRevenue"].sum().reset_index()
val_pred_df["transactionRevenue"] = np.log1p(val_pred_df["transactionRevenue"].values)
val_pred_df["PredictedRevenue"] =  np.log1p(val_pred_df["PredictedRevenue"].values)

#Now apply rms to find out error
print(np.sqrt(metrics.mean_squared_error(val_pred_df["transactionRevenue"].values, val_pred_df["PredictedRevenue"].values)))


# In[ ]:


train_id = dftrain["fullVisitorId"].values
test_id = test_df["fullVisitorId"].values   
submit_df = pd.DataFrame({"fullVisitorId":test_id})

#Repeat same steps as we did for cross-validation
pred_test[pred_test<0] = 0
submit_df["PredictedLogRevenue"] = np.expm1(pred_test)
submit_df = submit_df.groupby("fullVisitorId")["PredictedLogRevenue"].sum().reset_index()
submit_df["PredictedLogRevenue"] = np.log1p(submit_df["PredictedLogRevenue"])

submit_df.columns = ["fullVisitorId", "PredictedLogRevenue"]
print(submit_df.head())
submit_df.to_csv("submission.csv", index=False)


# In[ ]:




