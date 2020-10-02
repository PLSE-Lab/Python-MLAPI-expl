#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import xgboost as xgb
from sklearn import preprocessing
import gc


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ntrain_identity = pd.read_csv('../input/ieee-fraud-detection/train_identity.csv', index_col='TransactionID')\ntrain_transaction = pd.read_csv('../input/ieee-fraud-detection//train_transaction.csv', index_col='TransactionID')\ntest_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv', index_col='TransactionID')\ntest_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv', index_col='TransactionID')\nsample_submission = pd.read_csv('../input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')\n\ntrain = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)\ntest = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)\n\nprint(train.shape)\nprint(test.shape)\n\ny_train = train['isFraud'].copy()\ndel train_identity, train_transaction, test_identity, test_transaction\n\nX_train = train.drop('isFraud', axis=1)\nX_test = test.copy()\n\ndel train, test\n\nX_train = X_train.fillna(-999)\nX_test = X_test.fillna(-999)\n\nfor col in X_train.columns:\n    if X_train[col].dtype == 'object' or X_test[col].dtype == 'object':\n        lbl = preprocessing.LabelEncoder()\n        lbl.fit(list(X_train[col].values) + list(X_test[col].values))\n        X_train[col] = lbl.transform(list(X_train[col].values))\n        X_test[col] = lbl.transform(list(X_test[col].values))")


# In[ ]:


clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=9,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    missing=-999,
    random_state=2000
)


# In[ ]:


get_ipython().run_line_magic('time', 'clf.fit(X_train, y_train)')


# In[ ]:


sample_submission['isFraud'] = clf.predict_proba(X_test)[:, 1]
sample_submission.to_csv('xgboost.csv')


# In[ ]:




