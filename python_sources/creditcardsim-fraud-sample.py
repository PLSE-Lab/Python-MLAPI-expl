#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.model_selection import train_test_split
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv('../input/PS_20174392719_1491204439457_log.csv')
print(df.shape)
df = df.rename(columns={'oldbalanceOrg':'oldBalanceOrigin', 'newbalanceOrig':'newBalanceOrigin',"step":"time",                         'oldbalanceDest':'oldBalanceDestination', 'newbalanceDest':'newBalanceDestination', "isFlaggedFraud":"flag_rule_fraud_largeTransaction"})
df.head()


# ## Convert steps to time
# * Each step is ~ 1 hour. Set arbitrary start time
# * https://stackoverflow.com/questions/46081457/pandas-add-seconds-from-a-column-to-datetime-in-other-column

# In[ ]:


df['time'] = pd.to_datetime("01/01/2017") + pd.to_timedelta(df['time'], unit='h')
df['time'].sample(5)


# ## Downsample data for modelling
# * We will keep a test set with the original distribution

# In[ ]:


df_train, df_test = train_test_split(df, stratify=df.isFraud, test_size=0.05)


# In[ ]:


print(df_test.shape)
print(df_test.isFraud.sum())
# df_test.isFraud.describe()


# In[ ]:


df_pos = df_train.loc[(df_train.isFraud == 1) | (df_train.flag_rule_fraud_largeTransaction == 1) ]
print(df_pos.shape)
df_neg = df_train.loc[(df_train.isFraud == 0) ]
print(df_neg.shape)

df_neg = df_neg.sample(frac=0.05)
print(df_neg.shape)


# In[ ]:


df_train = pd.concat([df_pos,df_neg]).sample(frac=0.3)
print("downsampled, subsampled train data size:", df_train.shape )
print("subsampled Fraud cases: ",df_train.isFraud.sum())


# ##### Save/export our data
# 
# * We may wish to get also the transactions for all customers +- merchants in the downsampled train/test data *only* for creating features. 

# In[ ]:


# customers OR merchants can be in orig or Dest for this dataset (unlike BankSim)
customer_sampled = set(df_train.nameOrig).union(set(df_test.nameOrig)).union(set(df_test.nameDest)).union(set(df_test.nameDest))
len(customer_sampled)


# In[ ]:


print("full data:",df.shape[0])
df_context = df.loc[(df.nameOrig.isin(customer_sampled)) | (df.nameDest.isin(customer_sampled))]
print(df_context.shape[0])


# In[ ]:


df_train.tail(3)


# In[ ]:


df_train.to_csv("Train-creditcardsfraud_sampled.csv.gz",index=False,compression="gzip")
df_test.to_csv("Test-creditcardsfraud.csv.gz",index=False,compression="gzip")


# In[ ]:


df_context.to_csv("context_filtered_OrigDest.csv.gz",index=False,compression="gzip")

