#!/usr/bin/env python
# coding: utf-8

# # A wise person once said "Look at all of your data"
# 
# The goal of this kernel is to allow for visual inspection of all numeric features as a function of time. The hope is to:
# - Familiarize ourselves with the data
# - Identify any potential time-series related trends in the data that our models may not pick up on, in hopes of creating new features that capture this information.
# - This is especially important for features in the test set that display charachteristics in the test set that are not found in the training set.
# 
# Machine Learning is REALLY good at figuring out patterns. Machine Learning is NOT good at making predictions for a state it has never seen before.
# 
# Because the holdout set for this competition is from a moment in time later than the training data

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pylab as plt
import datetime


# In[ ]:


# Transaction CSVs
train_transaction = pd.read_csv('../input/train_transaction.csv')
test_transaction = pd.read_csv('../input/test_transaction.csv')
# Identity CSVs - These will be merged onto the transactions to create additional features
train_identity = pd.read_csv('../input/train_identity.csv')
test_identity = pd.read_csv('../input/test_identity.csv')
# Sample Submissions
ss = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


# Add the `isFraud` column for analysis
train_identity_ = train_identity.merge(train_transaction[['TransactionID',
                                                         'TransactionDT',
                                                         'isFraud']],
                                      on=['TransactionID'])

test_identity_ = test_identity.merge(test_transaction[['TransactionID',
                                                      'TransactionDT']],
                                    on=['TransactionID'])


# In[ ]:


# Idea from https://www.kaggle.com/kevinbonnes/transactiondt-starting-at-2017-12-01
# Read his kernel it's great!
def convert_TranactionDT(df):
    try:
        START_DATE = "2017-12-01"
        startdate = datetime.datetime.strptime(START_DATE, "%Y-%m-%d")
        df["TransactionDT"] = df["TransactionDT"].apply(
            lambda x: (startdate + datetime.timedelta(seconds=x))
        )
        return df
    except TypeError:
        """Already converted?"""
        return df

train_transaction = convert_TranactionDT(train_transaction)
test_transaction = convert_TranactionDT(test_transaction)
train_identity_ = convert_TranactionDT(train_identity_)
test_identity_ = convert_TranactionDT(test_identity_)


# In[ ]:


for i in train_identity_.columns:
    if i in ["isFraud", "TransactionDT", "TransactionID"]:
        continue
    try:
        train_identity_.loc[train_identity_["isFraud"] == 0].set_index("TransactionDT")[
            i
        ].plot(
            style=".",
            title=i,
            figsize=(15, 3),
            alpha=0.2,
            label="Not Fraud",
            rasterized=True,
        )
        train_identity_.loc[train_identity_["isFraud"] == 1].set_index("TransactionDT")[
            i
        ].plot(style=".", title=i, figsize=(15, 3), label="Fraud", alpha=0.5)
        test_identity_.set_index("TransactionDT")[i].plot(
            style=".",
            title=i,
            figsize=(15, 3),
            alpha=0.2,
            label="Test Data",
            rasterized=True,
        )
        plt.legend()
        plt.show()
    except TypeError:
        pass


# In[ ]:


for i in train_transaction.columns:
    if i in ["isFraud", "TransactionDT", "TransactionID"]:
        continue
    try:
        train_transaction.loc[train_transaction["isFraud"] == 0].set_index(
            "TransactionDT"
        )[i].sample(10000).plot(
            style=".",
            title=i,
            figsize=(15, 3),
            alpha=0.05,
            label="Not Fraud",
            rasterized=True,
        )
        train_transaction.loc[train_transaction["isFraud"] == 1].set_index(
            "TransactionDT"
        )[i].sample(10000).plot(
            style=".",
            title=i,
            figsize=(15, 3),
            label="Fraud",
            alpha=0.05,
            rasterized=True,
        )

        test_transaction.set_index("TransactionDT")[i].sample(10000).plot(
            style=".",
            title=i,
            figsize=(15, 3),
            alpha=0.05,
            label="Test Data",
            rasterized=True,
        )

        plt.legend()
        plt.show()
    except TypeError:
        pass

