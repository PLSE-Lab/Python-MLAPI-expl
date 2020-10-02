#!/usr/bin/env python
# coding: utf-8

# Previous kernel - https://www.kaggle.com/priteshshrivastava/ieee-pipeline-1-create-validation-set
# 
# Input - Train & val, test CSVs
# 
# Output - Val & Test preds
# 
# Next kernel - Meta model

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from catboost import CatBoostClassifier, Pool, cv
import os
import math
from sklearn.metrics import roc_auc_score
import pickle

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


train_X = pd.read_pickle("/kaggle/input/ieee-pipeline-1-create-validation-set/train_X.pkl")
train_y = pd.read_csv("/kaggle/input/ieee-pipeline-1-create-validation-set/train_y.csv")


# In[ ]:


train_y.head()


# In[ ]:


val_X = pd.read_pickle("/kaggle/input/ieee-pipeline-1-create-validation-set/val_X.pkl")
val_y = pd.read_csv("/kaggle/input/ieee-pipeline-1-create-validation-set/val_y.csv")


# In[ ]:


test_df = pd.read_pickle("/kaggle/input/ieee-pipeline-1-create-validation-set/test_df.pkl")
test_df.head()


# ### Specify & fit models on training set

# In[ ]:


print(f"Before dropna, top missing columns:\n{train_X.isna().sum().sort_values(ascending = False).head(5)}\n")

thresh = 0.60 #how many NA values (%) I think anything more than 80% is a bit too much. This is of course only my opinion

train_X_less_nas = train_X.dropna(thresh=train_X.shape[0]*(1-thresh), axis='columns')

cols_dropped  = list(set(train_X.columns)-set(train_X_less_nas.columns))

test_df.drop(cols_dropped, axis=1, inplace=True)
val_X.drop(cols_dropped, axis=1, inplace=True)

print(f"After dropna, top missing columns:\n{train_X_less_nas.isna().sum().sort_values(ascending = False).head(5)}")

print(f"\nNo. of cols dropped = {len(set(train_X.columns)-set(train_X_less_nas.columns))}")


# In[ ]:


cat_params = {
    'loss_function': 'Logloss',
    'custom_loss':['AUC'],
    'logging_level':'Silent',
    'task_type' : 'CPU',
    'early_stopping_rounds' : 100
}

modelA = CatBoostClassifier(**cat_params)


# In[ ]:


train_X_less_nas.fillna(-10000, inplace=True)
test_df.fillna(-10000, inplace=True)


# In[ ]:


Catfeats = ['ProductCD'] +            ["card"+f"{i+1}" for i in range(6)] +            ["addr"+f"{i+1}" for i in range(2)] +            ["P_emaildomain", "R_emaildomain"] +            ["M"+f"{i+1}" for i in range(9)] +            ["DeviceType", "DeviceInfo"] +            ["id_"+f"{i}" for i in range(12, 39)]

Catfeats = list(set(Catfeats)- set(cols_dropped))


# ### Defining function to calculate the evaluation metric

# In[ ]:


def auc(x,y): 
    return roc_auc_score(x,y)
def print_score(m):
    res = [auc(m.predict(train_X_less_nas), train_y), auc(m.predict(val_X), val_y)]
    print(res)


# In[ ]:


modelA.fit(train_X_less_nas, train_y, cat_features = Catfeats)


# In[ ]:


print_score(modelA)


# ### Make predictions on validation AND test set

# In[ ]:


predsA = pd.Series(modelA.predict(val_X))   ## or predict_proba ?? Not sklearn package


# In[ ]:


test_predsA = pd.Series(modelA.predict(test_df))


# ### Storing val & test pred

# In[ ]:


predsA.to_csv("predsA.csv", index = False, header = True)
test_predsA.to_csv("test_predsA.csv", index = False, header = True)


# ### Create submission file for single model

# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")
#sample_submission['isFraud'] = modelA.predict_proba(test_df)[:,1]
sample_submission['isFraud'] = modelA.predict(test_df)              ## Does this give continuous response ??
sample_submission.to_csv('catboost+ftsel.csv', index=False)

