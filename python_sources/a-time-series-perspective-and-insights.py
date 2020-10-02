#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


# In[ ]:


# Loading transaction data
train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')


# ## Assuming the start date to be 1-Dec-2017 as hypothesized in the "TransactionDT startdate" kernel 

# In[ ]:


START_DATE = '2017-12-01'
startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

train_transaction['TransactionDT'] = train_transaction['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))
train_transaction['trans_yymm'] = train_transaction['TransactionDT'].map(lambda x: x.replace(day=1))
train_transaction['trans_date'] = train_transaction['TransactionDT'].map(lambda x: x.date())


# ## Observed fraud rates each day

# In[ ]:


fraud_rate_daily = train_transaction.groupby(['trans_date'])['isFraud'].mean()

#Determing 30 day rolling statistics
rolmean = fraud_rate_daily.rolling(30).mean()
rolstd = fraud_rate_daily.rolling(30).std()

#Plot rolling statistics:
orig = plt.plot(fraud_rate_daily, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('Fraud Rate: Rolling Mean & Standard Deviation')
plt.show(block=False)


# Fraud rates are lower in December with the lowest point around christmas day. The fraud rate from January onwards seems to hover ~4%. 
# 
# The lower fraud rates in December could be due to higher number of genuine transactions in December. Lets test this by looking at the number of fraud transactions.

# ## Number of fraud transactions each day 

# In[ ]:


num_frauds_daily = train_transaction.groupby(['trans_date'])['isFraud'].sum()

#Determing 30 day rolling statistics
rolmean = num_frauds_daily.rolling(30).mean()
rolstd = num_frauds_daily.rolling(30).std()

#Plot rolling statistics:
orig = plt.plot(num_frauds_daily, color='blue',label='Original')
mean = plt.plot(rolmean, color='red', label='Rolling Mean')
std = plt.plot(rolstd, color='black', label = 'Rolling Std')
plt.legend(loc='best')
plt.title('# Fraud transactions: Rolling Mean & Standard Deviation')
plt.show(block=False)


# Unlike the fraud rate, the rolling average seems pretty stable until May when we see a drop and possibly a shift in mean value?

# ## Checking stationarity 

# In[ ]:


# Fraud rate
dftest = adfuller(fraud_rate_daily, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# Fraud rate doesn't pass the ADF test for stationarity as expected from looking at the graph

# In[ ]:


# Number of Fraud transactions
dftest = adfuller(num_frauds_daily, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)


# Number of fraudulent activities passes the ADF test for stationarity despite the drop we observed around May-2018

# ## How do these insights help with my modeling?

# 1. If anyone is using undersampling/oversampling approaches to deal with the class imbalance. It could possibly be benefitial to consider the month/date of transactions, try balancing classes over each month rather than over the entire training set.  
# 
# 2. If the drop in number of fraudulent activies we observed in May-2018, does indicate a shift in mean value. Then the expected number of fraudlent activities in test set could be lower than what we observed in the train set. 
# A speculation but this could also possibly explain why models are getting higher LB AUC values than local CV AUC values.
