#!/usr/bin/env python
# coding: utf-8

# **Disclaimer: I am new to Python so use it on your own risk.**
# Please let me know if I forgot to reference somebody's work.
# 
# Shamelessly copied from Konrad Banachewicz and Inversion. Also deleted 'TransactionDT' according to Bojan Tunguz's adversarial result.
# 
# [Date hint](https://www.kaggle.com/c/ieee-fraud-detection/discussion/100071)
# 
# ['2017-12-01' date start is taken from this kernel](https://www.kaggle.com/kevinbonnes/transactiondt-startdate)
# 
# Converted 'TransactionDT' to day-of-week, the orignial 'TransactionDT'is removed again.
# 
# Changed encoding for day_of_week to one-hot

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import datetime
import math

import os
print(os.listdir("../input"))

START_DATE = '2017-12-01'


# In[ ]:


from sklearn import preprocessing
import xgboost as xgb


# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID')
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')


# In[ ]:


train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)

startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')

train['day_week'] = train['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x))).dt.day_name()
test['day_week'] = test['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x))).dt.day_name()

y_train = train['isFraud'].copy()

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_train.drop('TransactionDT', axis=1, inplace=True)
X_train.drop('id_03', axis=1, inplace=True)
X_test = test.drop('TransactionDT', axis=1)
X_test.drop('id_03', axis=1, inplace=True)

X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


# In[ ]:


del train, test, train_transaction, train_identity, test_transaction, test_identity


# In[ ]:


# Trying One-Hot on day_week

X_train = pd.concat([X_train, pd.get_dummies(X_train['day_week'], prefix='Day')], axis=1)
X_test = pd.concat([X_test, pd.get_dummies(X_test['day_week'], prefix='Day')], axis=1)
X_train.drop('day_week', axis=1, inplace=True)
X_test.drop('day_week', axis=1, inplace=True)
print(X_train.shape)
print(X_test.shape)


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[ ]:


clf = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.7,
                        colsample_bytree=0.7,
                        missing=-999, 
                        gamma = 0.1)

clf.fit(X_train, y_train)


# In[ ]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('simple_xgboost_dayweek_1hot.csv')

