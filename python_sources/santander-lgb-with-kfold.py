#!/usr/bin/env python
# coding: utf-8

# # Santander Customet Transaction prediction by applying LGB with K-Fold
# 
# This kernel show some visualization of the data and applied Light Gradient Boosting (LGB) algorithm over K-Fold cross validation. After applying different algorithms (Decision Tree, Logistic regression, PCA to diemention reduction; which will discuss later) this approaches gives better accuracy.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Load data

# In[ ]:


# read train.csv file
df = pd.read_csv("../input/train.csv")
df.head()


# # Data check, visualization and preprocessing

# In[ ]:


# check if there is any empty cell or not
df.isnull().any().any()


# In[ ]:


#check classification distribution
df['target'].value_counts().plot.bar()


# In[ ]:


#check classification distribution in percentage. So, target are not balanced
df['target'].value_counts(normalize=True)


# In[ ]:


# separate target column and feature columns
labels = df.pop('target')
data = df.drop('ID_code', axis=1)


# In[ ]:


# load test data
test_data = pd.read_csv("../input/test.csv")
test_data.head()


# In[ ]:


# keep only features columns
test_data = test_data.drop('ID_code', axis=1)
test_data.head()


# In[ ]:


# K-fold corss validation with 10 fold
n_splits = 10
kf = KFold(n_splits=n_splits)


# In[ ]:


# model with LGB
import warnings
warnings.filterwarnings('ignore')

param = {'objective': 'binary', 'metric': 'auc', 'learning_rate': 0.01, 'num_rounds': 6000, 'verbose': 1}
test_pred = np.zeros(len(test_data))
for fold, (train_indx, val_indx) in enumerate(kf.split(labels)):
    print("Fold {}".format(fold+1))
    train_set = lgb.Dataset(data.iloc[train_indx], label=labels.iloc[train_indx])
    val_set = lgb.Dataset(data.iloc[val_indx], label=labels.iloc[val_indx])
    model = lgb.train(param, train_set, valid_sets=val_set, verbose_eval=500)
    test_pred += model.predict(test_data)/n_splits


# In[ ]:


# save data
save = pd.read_csv('../input/sample_submission.csv')
save['target'] = test_pred
save.to_csv('LGB_kfold.csv', index=False)
save.head()


# A simple model but great works. upvote if you like :):)

# In[ ]:




