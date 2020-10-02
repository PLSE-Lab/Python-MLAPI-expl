#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
print(os.listdir("../input"))


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
adversarial_features = ["TransactionDT"]
train = train.drop(adversarial_features, axis = 1)
test = test.drop(adversarial_features, axis = 1)
print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()
X_train = X_train.fillna(-999)
X_test = X_test.fillna(-999)


# In[ ]:


del train, test, train_transaction, train_identity, test_transaction, test_identity


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size = .1)


# In[ ]:


print(X_train.shape)
print(X_val.shape)
print(y_train.shape)
print(y_val.shape)


# In[ ]:


eval_set = [(X_val, y_val)]


# In[ ]:


clf = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=6,
                        learning_rate=0.1,
                        subsample=0.7,
                        colsample_bytree=0.7,
                        missing=-999,
                       )

clf.fit(X_train, y_train, eval_set = eval_set, early_stopping_rounds = 10)


# In[ ]:


from sklearn.metrics import roc_auc_score


# In[ ]:


xgb_pred_val = clf.predict_proba(X_val)[:,1]


# In[ ]:


print(roc_auc_score(y_val, xgb_pred_val))


# In[ ]:


import lightgbm as lgb


# In[ ]:


for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        X_train[f] = X_train[f].astype("category")
        X_train[f] = X_train[f].astype("category")
        X_test[f] = X_test[f].astype("category")  


# In[ ]:


params = {'application': 'xentropy',
          'boosting': 'gbdt',
          'learning_rate': 0.1,
          'bagging_fraction': 0.6,
          'feature_fraction': 0.6,
          'verbosity': -1,
          'data_random_seed': 24,
          'early_stop': 10,
          'verbose_eval': 50,
          'num_rounds': 10000}


# In[ ]:


d_train = lgb.Dataset(X_train, label=y_train)
d_valid = lgb.Dataset(X_val, label = y_val)
watchlist = [d_train, d_valid]
num_rounds = params.pop('num_rounds')
verbose_eval = params.pop('verbose_eval')
early_stop = None
if params.get('early_stop'):
    early_stop = params.pop('early_stop')
model = lgb.train(params,
                  train_set=d_train,
                  num_boost_round=num_rounds,
                  valid_sets=watchlist,
                  verbose_eval=verbose_eval,
                  early_stopping_rounds=early_stop)


# In[ ]:


lgb_pred_val = model.predict(X_val, num_iteration=model.best_iteration)
print(roc_auc_score(y_val, lgb_pred_val))


# In[ ]:


lgb_pred_val


# In[ ]:


print(roc_auc_score(y_val, (xgb_pred_val + lgb_pred_val)/2))


# In[ ]:


xgb_pred = clf.predict_proba(X_test)[:,1]
lgb_pred = model.predict(X_test, num_iteration=model.best_iteration)


# In[ ]:


((xgb_pred_val + lgb_pred_val)/2).mean()


# In[ ]:


sample_submission['isFraud'] = (xgb_pred + lgb_pred)/2
sample_submission.to_csv('simple_xgboost.csv')


# In[ ]:




