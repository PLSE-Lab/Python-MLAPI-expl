#!/usr/bin/env python
# coding: utf-8

# Previous kernel - https://www.kaggle.com/priteshshrivastava/ieee-pipeline-1-create-validation-set
# 
# Input - Train & val, test CSVs
# 
# Output - Val & Test preds
# 
# Next kernel - Meta model https://www.kaggle.com/priteshshrivastava/ieee-pipeline-3-stacking-with-meta-model

# This one is based on Inversion's simple xgb kernel : https://www.kaggle.com/inversion/ieee-simple-xgboost/output

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import math
from sklearn.metrics import roc_auc_score
import pickle
from sklearn import preprocessing
import xgboost as xgb

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


# ### Handling missing values & categorical variables

# In[ ]:


train_X = train_X.fillna(-999)
val_X = val_X.fillna(-999)
test_df = test_df.fillna(-999)


# In[ ]:


# Label Encoding
for f in train_X.columns:
    if train_X[f].dtype=='object' or val_X[f].dtype=='object' or test_df[f].dtype=='object': 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_X[f].values) + list(val_X[f].values) + list(test_df[f].values))
        train_X[f] = lbl.transform(list(train_X[f].values))
        val_X[f] = lbl.transform(list(val_X[f].values))
        test_df[f] = lbl.transform(list(test_df[f].values))


# ### Defining function to calculate the evaluation metric

# In[ ]:


def auc(x,y): 
    return roc_auc_score(x,y)
def print_score(m):
    #res = [auc(m.predict_proba(train_X)[:,1], train_y), auc(m.predict_proba(val_X)[:,1], val_y)]  ## continuous not supported
    res = [auc(m.predict(train_X), train_y), auc(m.predict(val_X), val_y)]
    print(res)


# In[ ]:


modelC = xgb.XGBClassifier(n_estimators=500,
                        n_jobs=4,
                        max_depth=9,
                        learning_rate=0.05,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        missing=-999)


# In[ ]:


modelC.fit(train_X, train_y)


# In[ ]:


print_score(modelC)


# ### Make predictions on validation AND test set

# In[ ]:


predsC = pd.Series(modelC.predict_proba(val_X)[:,1])


# In[ ]:


test_predsC = pd.Series(modelC.predict_proba(test_df)[:,1])


# ### Storing val & test pred

# In[ ]:


predsC.to_csv("predsC.csv", index = False, header = True)
test_predsC.to_csv("test_predsC.csv", index = False, header = True)


# ### Creating a submission file for the single model

# In[ ]:


sample_submission = pd.read_csv("/kaggle/input/ieee-fraud-detection/sample_submission.csv")
sample_submission['isFraud'] = modelC.predict_proba(test_df)[:,1]
sample_submission.to_csv('simple_xgboost.csv', index=False)

