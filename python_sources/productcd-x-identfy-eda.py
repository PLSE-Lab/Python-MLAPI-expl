#!/usr/bin/env python
# coding: utf-8

# There is no identify rows when ProductCD is 'W'

# In[ ]:


import pandas as pd


# In[ ]:


train_trans = pd.read_csv('../input/train_transaction.csv')
test_trans = pd.read_csv('../input/test_transaction.csv')
train_identity = pd.read_csv('../input/train_identity.csv')
test_identity = pd.read_csv('../input/test_identity.csv')


# In[ ]:


train_trans['has_identify'] = train_trans['TransactionID'].isin(
    train_identity['TransactionID']).astype(int)
test_trans['has_identify'] = test_trans['TransactionID'].isin(
    test_identity['TransactionID']).astype(int)


# In[ ]:


train_trans.groupby('ProductCD')['has_identify'].agg(
    ['count', 'sum', 'mean']
).rename({
    'count': 'all',
    'sum': 'number of has_identify',
    'mean': 'has_identify ratio'}, axis=1)


# In[ ]:


test_trans.groupby('ProductCD')['has_identify'].agg(
    ['count', 'sum', 'mean']
).rename({
    'count': 'all',
    'sum': 'number of has_identify',
    'mean': 'has_identify ratio'}, axis=1)


# isFraud ratio per ProductCD

# In[ ]:


train_trans.groupby('ProductCD')['isFraud'].agg(
    ['count', 'sum', 'mean']
).rename({
    'count': 'all',
    'sum': 'number of isFraud',
    'mean': 'isFraud ratio'}, axis=1)


# enjoy kaggling!

# In[ ]:




