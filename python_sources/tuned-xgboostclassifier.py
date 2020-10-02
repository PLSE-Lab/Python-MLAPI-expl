#!/usr/bin/env python
# coding: utf-8

# This is my first notebook and submission to a kaggle competition. I took on this competition for a university project and have tested many different models including, logistic regression, random forest, svm and neural networks. I discovered out of these model tested my best result came from a random forest classifier. I tuned this model and improved it too a auc_roc_score of 0.92. Upon searching for more ways to improve this model I found gradient boosting models and tested both LGB and XGB with XGB providing larger auc scores. From here I began to tune the model by testing different stratified samples and parameters to come up with the result below. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.preprocessing import LabelEncoder,StandardScaler #(encoding and standardising data)
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# First Loading in the datasets

# In[ ]:


train_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_identity.csv")
train_tran = pd.read_csv("/kaggle/input/ieee-fraud-detection/train_transaction.csv")
test_id = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_identity.csv")
test_tran = pd.read_csv("/kaggle/input/ieee-fraud-detection/test_transaction.csv")
sample_submission = pd.read_csv('/kaggle/input/ieee-fraud-detection/sample_submission.csv', index_col='TransactionID')


# * Merging the transaction and identity dataset for test and train.
# * Converting to a dataframe
# * Deleting unused datasets to save memory

# In[ ]:


train = pd.merge(train_tran, train_id, on="TransactionID", how="left")
test = pd.merge(test_tran, test_id, on="TransactionID", how="left")
train = pd.DataFrame(train)
test = pd.DataFrame(test)
del train_id, train_tran, test_id, test_tran


# Splitting the target variables for the trainning set & deleting unused dataset 

# In[ ]:


train_y = train.isFraud
train_x=train.drop(["isFraud"], axis=1)
del train


# To apply the data to xgboost I label encode the data using sklearns labelencoder() method

# In[ ]:


categorical = ['ProductCD','card1' ,'card2' ,'card3' ,'card4' ,'card5' ,'card6', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1'
               ,'M2', 'M3', 'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'DeviceType', 'DeviceInfo', 'id_12', 'id_13', 'id_14', 'id_15', 'id_16', 'id_17', 
               'id_18', 'id_19', 'id_20', 'id_21', 'id_22', 'id_23', 'id_24', 'id_25', 'id_26', 'id_27', 'id_28', 'id_29', 'id_30', 
               'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38']



lb_make = LabelEncoder()

length_cat = len(categorical)
for i in range(length_cat):
    train_x[categorical[i]] = lb_make.fit_transform(train_x[categorical[i]].astype(str))
    test[categorical[i]] = lb_make.fit_transform(test[categorical[i]].astype(str))
   


# Next I standardise the data to equalise the range and variability of the data 

# In[ ]:


train_x = StandardScaler().fit_transform(train_x)
test = StandardScaler().fit_transform(test)


# I have tuned a model by selecting optimal, max_depth, min_child_weight, sub_sample and colsample_bytree I selected random forest as my boosting alogrithm as apart of my report process I explored the use of random forest and wanted to apply the xgboosting alogrithm to strengthen its learning ability. This model gave my best result so far. 

# In[ ]:


model = XGBClassifier(boosting="gbdt", max_depth=10, min_child_weight=3, subsample=0.9, colsample_bytree=0.9, gamma=0.4)
model.fit(train_x, train_y)
pred = model.predict_proba(train_x)[:, 1]
print("Tuned")
print("AUC", roc_auc_score(train_y, pred))


# Creating a submission for the competition

# In[ ]:


sample_submission['isFraud'] = model.predict_proba(test)[:, 1]
sample_submission.to_csv('tuned_xgb3.csv')


# I am completing a more in-depth report for my university project and this is just the model I have found to be the strongest so far and I will continue to explore different models. Any tips for improving my model or problems please comment as I wish to gain as much knowledge from this experience as possible.
