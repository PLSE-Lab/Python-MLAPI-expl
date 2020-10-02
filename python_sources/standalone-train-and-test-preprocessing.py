#!/usr/bin/env python
# coding: utf-8

# Simple train and test preprocessing based on Inversion's script: https://www.kaggle.com/inversion/ieee-simple-xgboost

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing
import os
print(os.listdir("../input"))


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


# In[ ]:


train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)


# In[ ]:


train = train.drop('isFraud', axis=1)
train = train.fillna(-999)
test = test.fillna(-999)


# In[ ]:


object_columns = []

for f in train.columns:
    if train[f].dtype=='object' or test[f].dtype=='object': 
        object_columns.append(f)
        
np.save('object_columns', object_columns)


# In[ ]:




