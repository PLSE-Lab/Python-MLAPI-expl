#!/usr/bin/env python
# coding: utf-8

# * XGBOost model hyperparam tuner
# * modified to use temporal split for CV. 
# * Added 3 of the features from my features kernel, and datetime : 
#     * https://www.kaggle.com/danofer/ieee-fraud-features-xgboost-0-934-lb
#     * https://www.kaggle.com/danofer/ieee-fraud-new-features-export-0-9359-lb
#     
# * Credit for the specific date (vs just 1.1.2018" goes to : https://www.kaggle.com/kevinbonnes/transactiondt-starting-at-2017-12-01

# In[ ]:


import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import xgboost as xgb
# import lightgbm as lgb
import optuna
import functools
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve,auc,accuracy_score,confusion_matrix,f1_score
import datetime


# In[ ]:


train_transaction = pd.read_csv('../input/train_transaction.csv', index_col='TransactionID') # ,nrows=12345
test_transaction = pd.read_csv('../input/test_transaction.csv', index_col='TransactionID')

train_identity = pd.read_csv('../input/train_identity.csv', index_col='TransactionID')
test_identity = pd.read_csv('../input/test_identity.csv', index_col='TransactionID')

sample_submission = pd.read_csv('../input/sample_submission.csv', index_col='TransactionID')

train = train_transaction.merge(train_identity, how='left', left_index=True, right_index=True)
test = test_transaction.merge(test_identity, how='left', left_index=True, right_index=True)

print(train.shape)
print(test.shape)

y_train = train['isFraud'].copy()
del train_transaction, train_identity, test_transaction, test_identity

# Drop target, fill in NaNs
X_train = train.drop('isFraud', axis=1)
X_test = test.copy()

del train, test


# In[ ]:


import gc
gc.collect()


# In[ ]:


def fraud_datetime(df):
    """
    Credit for picking 31.12
    """
    START_DATE = '2017-12-01'
    startdate = datetime.datetime.strptime(START_DATE, '%Y-%m-%d')
    df['TransactionDT'] = df['TransactionDT'].apply(lambda x: (startdate + datetime.timedelta(seconds = x)))

    # df['month'] = df['TransactionDT'].dt.month
    df['dow'] = df['TransactionDT'].dt.dayofweek
    df['hour'] = df['TransactionDT'].dt.hour
#     df['day'] = df['TransactionDT'].dt.day
    df.drop(['TransactionDT'],axis=1,inplace=True)
    return df


# In[ ]:


get_ipython().run_cell_magic('time', '', 'X_train["null_counts"] = X_train.isna().sum(axis=1)\nX_test["null_counts"] = X_test.isna().sum(axis=1)\n\n\n# nunique appears unstable\n# currently trying wit just numeric cols, may use less mem\nX_train["nuniques"] = X_train.select_dtypes(include=[np.number]).nunique(axis=1)\nX_test["nuniques"] = X_test.select_dtypes(include=[np.number]).nunique(axis=1)')


# In[ ]:


## note: we also drop the TransactionDT here
X_train = fraud_datetime(X_train)
X_test = fraud_datetime(X_test)


# In[ ]:


## probably better to impute NANS, as the distrib changes between train and test

# X_train = X_train.fillna(-999)
# X_test = X_test.fillna(-999)


# In[ ]:


# Label Encoding
for f in X_train.columns:
    if X_train[f].dtype=='object' or X_test[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values) + list(X_test[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))
        X_test[f] = lbl.transform(list(X_test[f].values))   


# In[ ]:


gc.collect()
X_train.head()


# In[ ]:


## random split: 
# (X_train,X_eval,y_train,y_eval) = train_test_split(X_train,y_train,test_size=0.15,random_state=0)

## temporal split (assuming data remains sorted): split by last records, e.g. by recordid or TransactionDT

## 80% of data
TR_ROW_CNT = int(X_train.shape[0]*0.8)
print(TR_ROW_CNT)

X_eval = X_train[TR_ROW_CNT:]
print("eval shape",X_eval.shape)

X_train = X_train[:TR_ROW_CNT]
print("new train shape",X_train.shape)

y_eval = y_train[TR_ROW_CNT:]
y_train = y_train[:TR_ROW_CNT]


# ## Training
# 
# * metric to  AUC

# In[ ]:


# ## fast AUC metric + calc, from : https://www.kaggle.com/c/microsoft-malware-prediction/discussion/76013#latest-556434
# ### gives errors, doesn't work ? 

# # import numpy as np 
# from numba import jit

# @jit
# def fast_auc(y_true, y_prob):
#     y_true = np.asarray(y_true)
#     y_true = y_true[np.argsort(y_prob)]
#     nfalse = 0
#     auc = 0
#     n = len(y_true)
#     for i in range(n):
#         y_i = y_true[i]
#         nfalse += (1 - y_i)
#         auc += y_i * nfalse
#     auc /= (nfalse * (n - nfalse))
#     return auc

# def eval_auc(preds, dtrain):
#     labels = dtrain.get_label()
#     return 'auc', fast_auc(labels, preds), True


# In[ ]:



from sklearn.metrics import roc_auc_score

def opt(X_train, y_train, X_test, y_test, trial):
    #param_list
    n_estimators = trial.suggest_int('n_estimators', 400, 900) # may relate to instability of kernel when predicting? 
    max_depth = trial.suggest_int('max_depth', 4, 20)
    min_child_weight = trial.suggest_int('min_child_weight', 1, 25)
    #learning_rate = trial.suggest_discrete_uniform('learning_rate', 0.01, 0.1, 0.01)
    scale_pos_weight = trial.suggest_int('scale_pos_weight', 1, 30)
    subsample = trial.suggest_discrete_uniform('subsample', 0.6, 1.0, 0.1)
    colsample_bytree = trial.suggest_discrete_uniform('colsample_bytree', 0.6, 1.0, 0.1)

    xgboost_tuna = xgb.XGBClassifier(n_jobs=1,
        random_state=41, 
        tree_method='gpu_hist',
        n_estimators = n_estimators,
        max_depth = max_depth,
        min_child_weight = min_child_weight,
        #learning_rate = learning_rate,
        scale_pos_weight = scale_pos_weight,
        subsample = subsample,
        colsample_bytree = colsample_bytree
    )
    xgboost_tuna.fit(X_train, y_train)

    tuna_pred_test_proba = xgboost_tuna.predict_proba(X_test)[:,1]
    return (1.0 - (roc_auc_score(y_test, tuna_pred_test_proba)))

#     tuna_pred_test = xgboost_tuna.predict(X_test)
#     return (1.0 - (accuracy_score(y_test, tuna_pred_test))) # default ,accuracy


# In[ ]:


study = optuna.create_study()
study.optimize(functools.partial(opt, X_train, y_train, X_eval, y_eval), n_trials=80)


# In[ ]:


print("Best score found:",1-study.best_value)
print(study.best_params)


# In[ ]:


clf = xgb.XGBClassifier(tree_method='gpu_hist',**study.best_params)
X_train = pd.concat([X_train,X_eval])
y_train = pd.concat([y_train,y_eval])
del X_eval,y_eval

print(X_train.shape)
print(y_train.shape)
clf.fit(X_train,y_train)


# In[ ]:


gc.collect()


# In[ ]:


## plot features importance
import matplotlib.pyplot as plt

fi = pd.DataFrame(index=X_train.columns)
fi['importance'] = clf.feature_importances_
fi.loc[fi['importance'] > 0.0005].sort_values('importance',ascending=False).head(32).plot(kind='barh', figsize=(8, 24), title='Feature Importance')
plt.show()


# In[ ]:


## kernel tends to run out of memory when doing predictions ?

sample_submission['isFraud'] = clf.predict_proba(X_test)[:,1]
sample_submission.to_csv('submission.csv')

