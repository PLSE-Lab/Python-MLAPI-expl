#!/usr/bin/env python
# coding: utf-8

# I am trying identify churn from previous months. I know the organiser's algorithm is unkown but we can try to make an  educated guess and go with it. Below I make an attempt in the hope that it will be challenged and improved. 
# 
# There are a few benefits in having churn from previous months: 
# - Understand how the data set was generated
# - Create features for the provided train set
# - Generate additional train set
# 
# Re the first point, thinking about previous periods' churn got me thinking that transaction and log data perhaps should use  different time thresholds for the provided train vs test sets. Predicting March churners using transaction data up to only February means that the model trained on Feb churners should be based on transaction data up to Jan. I have not seen any baseline model taking that into account - am I missing something?
# 
# Re the last point, I would like to train a model on previous years' (2016, 2015) churn for March to factor in the potential cyclicality 
# 

# For convenience I include the organiser's rules for churn
# 
# The churn/renewal definition can be tricky due to KKBox's subscription model. Since the majority of KKBox's subscription length is 30 days, a lot of users re-subscribe every month. The key fields to determine churn/renewal are transaction date, membership expiration date, and is_cancel. Note that the is_cancel field indicates whether a user actively cancels a subscription. Note that a cancellation does not imply the user has churned. A user may cancel service subscription due to change of service plans or other reasons. **The criteria of "churn" is no new valid service subscription within 30 days after the current membership expires.**
# 
# The train and the test data are selected from users whose membership expire within a certain month. The train data consists of users whose subscription expires within the month of February 2017, and the test data is with users whose subscription expires within the month of March 2017. This means we are looking at user churn or renewal roughly in the month of March 2017 for train set, and the user churn or renewal roughly in the month of April 2017. Train and test sets are split by transaction date, as well as the public and private leaderboard data.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
trans = pd.read_csv('../input/transactions.csv')


# Let's take the second user from the train set -  'QA7uiXy8vIbUSPOkCf9RwQ3FsT8jVq2OxDr8zqa7bRQ='
# 
# I also take another random one because the function for churn identification requires at least two users

# In[2]:


sample = trans.loc[trans.msno.isin(['QA7uiXy8vIbUSPOkCf9RwQ3FsT8jVq2OxDr8zqa7bRQ=','waLDQMmcOu2jLDaV1ddDkgCrB/jl6sD66Xzs0Vqax1Y=']),:].copy(deep=True)


# Here's my modest attempts at identifying churn from previous months. I know it's not good and potentially shit, so please let me know any suggestions on how to improve it

# In[3]:


def identify_churn(trans):
    trans["transaction_date"] = pd.to_datetime(trans["transaction_date"], format='%Y%m%d')
    trans["membership_expire_date"] = pd.to_datetime(trans["membership_expire_date"], format='%Y%m%d')
    trans = trans.sort_values(by=['msno', 'transaction_date']).reset_index(drop=True)
    trans["next_trans"] =trans.groupby("msno")["transaction_date"].shift(-1)
    trans["day_diff"] = trans.groupby("msno").apply(lambda trans: trans["next_trans"] - trans["membership_expire_date"]).reset_index(drop=True)
    threshold = pd.Timedelta('31 days')
    trans["churn_flag"] = trans["day_diff"]>threshold
    #trans['churn_date'] = trans["membership_expire_date"] + pd.Timedelta('31 days')
    return trans
sample = identify_churn(sample)


# Below is what happens for the selected user. Column 'churn_flag' should be True when a transaction indicates that she churned. It happens on transaction dated '2016-01-31'. The expiration date becomes '2016-03-21', and it takes her until '2016-05-05', i.e. more than a month later, to renew her subscription.
# 
# By the way, this example user confuses me. Train set say she churned in Feb. By definition there should be "no new valid service subscription within 30 days after the [Feb 2017] membership expires". The transaction corresponding to membership expiring in Feb is '2016-12-31'. However another transaction happens a month later to renew the subscription to March. So I don't understand why she would be labeled as a Feb churner. Any ideas?
# 
# 

# In[4]:


sample[sample.msno=='QA7uiXy8vIbUSPOkCf9RwQ3FsT8jVq2OxDr8zqa7bRQ=']

