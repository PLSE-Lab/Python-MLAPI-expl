#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import sys,re

import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
import warnings

from tqdm import tqdm
warnings.filterwarnings('ignore')


# In[ ]:


train_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
print (train_identity.shape)
train_identity.head()


# In[ ]:


train_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
print (train_transaction.shape)
train_transaction.head()


# In[ ]:


test_identity = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_identity.csv')
print (test_identity.shape)
test_identity.head()


# In[ ]:


test_transaction = pd.read_csv('/kaggle/input/ieee-fraud-detection/test_transaction.csv')
print (test_transaction.shape)
test_transaction.head()


# In[ ]:


def train_data_processing(transaction_data, identity_data):
    data = pd.merge(transaction_data, identity_data, on='TransactionID', how = 'left', suffixes=('_x','_y'))
    del_columns = []
    for column in data.columns:
        if ((data[column].isnull().sum()) > (len(data)*0.60)):
            del_columns.append(column)
    catcols = ['ProductCD','card4','M4','V14','V41','V65','V88','V94','card6','M1', 'M2', 'M3', 'M5', 'M6', 'M7', 'M8', 'M9']
    del_columns_1 = ['addr1','addr2','C5','C9','V12','V13','V29','V30','V35','V36','V48','V49','V53','V54','V69','V70','V75','V76','V90','V91','V107','V305','P_emaildomain']
    
    del_columns.extend(del_columns_1)
    data = data.drop(columns = del_columns, axis = 1)
    # data['isFraud'] = data['isFraud'].apply(lambda x: -1 if x ==0 else 1)
    data = pd.get_dummies(columns = catcols, data = data)
    return data
    
data = train_data_processing(train_transaction, train_identity)


# In[ ]:


def test_data_processing(transaction_data, identity_data):
    data = pd.merge(transaction_data, identity_data, on='TransactionID', how = 'left', suffixes=('_x','_y'))
    del_columns = []
    for column in data.columns:
        if ((data[column].isnull().sum()) > (len(data)*0.60)):
            del_columns.append(column)
    catcols = ['ProductCD','card4','M4','V14','V41','V65','V88','V94','card6','M1', 'M2', 'M3', 'M6', 'M7', 'M8', 'M9']
    del_columns_1 = ['addr1','addr2','C5','C9','V12','V13','V29','V30','V35','V36','V48','V49','V53','V54','V69','V70','V75','V76','V90','V91','V107','V305','P_emaildomain']
    
    del_columns.extend(del_columns_1)
    data = data.drop(columns = del_columns, axis = 1)
    data = pd.get_dummies(columns = catcols, data = data)
    return data

test_data = test_data_processing(test_transaction, test_identity)


# In[ ]:


y_train = data['isFraud']
X_train = data.drop(['isFraud','TransactionID'], 1)

X_test = test_data.drop(['TransactionID'], 1)


# In[ ]:


X_train.drop(['card6_debit or credit','M5_T','M5_F'], axis=1, inplace = True)


# In[ ]:


from lightgbm import LGBMClassifier

lgbmclassifier = LGBMClassifier().fit(X_train, y_train)
lgbm_result = lgbmclassifier.predict_proba(X_test)
lgbm_result = lgbm_result[:,1]


# In[ ]:


lgbm_importance = lgbmclassifier.feature_importances_
lgbm_importance = pd.DataFrame(lgbm_importance, index = X_train.columns, columns=['Importance'])
lgbm_importance = lgbm_importance.sort_values('Importance', ascending = False)
lgbm_importance = lgbm_importance[:100]
lgbm_importance


# In[ ]:


from xgboost import XGBClassifier

xgbclassifier = XGBClassifier().fit(X_train, y_train)
xgb_result = xgbclassifier.predict_proba(X_test)
xgb_result


# In[ ]:


xgb_importance = xgbclassifier.feature_importances_
xgb_importance = pd.DataFrame(xgb_importance, index = X_train.columns, columns=['Importance'])
xgb_importance = xgb_importance.sort_values('Importance', ascending = False)
xgb_importance = xgb_importance[:100]
xgb_importance


# In[ ]:


common_importance_list = list(set(lgbm_importance.index).intersection(xgb_importance.index))


# In[ ]:


len(common_importance_list)


# In[ ]:


X_train = X_train[common_importance_list]
print (X_train.shape)
X_train.head()


# In[ ]:


X_test = X_test[common_importance_list]
print (X_test.shape)
X_test.head()


# In[ ]:


lgbmclassifier_modified = LGBMClassifier().fit(X_train, y_train)
lgbm_result_modified = lgbmclassifier_modified.predict_proba(X_test)
lgbm_result_modified = lgbm_result_modified[:,1]

xgbclassifier_modified = XGBClassifier().fit(X_train, y_train)
xgb_result_modified = xgbclassifier_modified.predict_proba(X_test)
xgb_result_modified = xgb_result_modified[:,1]


# In[ ]:


avg_model= (lgbm_result_modified+xgb_result_modified)/2


# In[ ]:


avg_model


# In[ ]:


submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv')
print (submission.shape)
submission.head()


# In[ ]:


submission['isFraud'] = avg_model


# In[ ]:


submission.head()


# In[ ]:


submission.to_csv('submission_ensemble_modified.csv', index = False)


# In[ ]:




