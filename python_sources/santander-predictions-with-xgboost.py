#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


################
# Santander Customer Transaction
# Using XGBoost
# Hemal Vakharia
# Mar 27, 2019
#################
import pandas as pan # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import xgboost as xgb


# In[ ]:


#Given: Train and Test Data set - separately

#Reading Train Data set

def readfile(filename):
    filename = "../input/" + filename
    return(pan.read_csv(filename))

def checkfornull(dataset):
    return(dataset.columns[dataset.isnull().any()])

data_set_train = readfile("train.csv")
data_set_testing = readfile("test.csv")


# In[ ]:


target_train_Y = data_set_train['target'].values
train_X = data_set_train.drop(['target', 'ID_code'], axis=1)
test_X = data_set_testing.drop(['ID_code'], axis=1)

sc=StandardScaler()
train=sc.fit_transform(train_X)
test=sc.transform(test_X)


# In[ ]:


xgb_prediction = []


# In[ ]:


K = 5
kf = KFold(n_splits = K, random_state = 3228, shuffle = True)


# In[ ]:


for train_index, test_index in kf.split(train):
    train_X, valid_X = train[train_index], train[test_index]
    train_y, valid_y = target_train_Y[train_index], target_train_Y[test_index]
    xgb_params = {'max_depth': 8,'objective': 'binary:logistic','eval_metric':'auc'}

    d_train = xgb.DMatrix(train_X, train_y)
    d_valid = xgb.DMatrix(valid_X, valid_y)
    d_test = xgb.DMatrix(test)
    watchlist = [(d_train, 'train'), (d_valid, 'valid')]
    model = xgb.train(xgb_params, d_train,500,watchlist,early_stopping_rounds=20) #500, #150
    xgb_pred = model.predict(d_test)
    xgb_prediction.append(list(xgb_pred))


# In[ ]:


fig, ax = plt.subplots(1,1,figsize=(10,12))
xgb.plot_importance(model, max_num_features=30, ax=ax, importance_type="cover", xlabel="Cover")
print("---- drawing features by importance in descending order---")
plt.show()


# In[ ]:


preds=[]
for i in range(len(xgb_prediction[0])):
    sum=0
    for j in range(K):
        sum+=xgb_prediction[j][i]
    preds.append(sum / K)


# In[ ]:


result = pan.DataFrame({"ID_code": data_set_testing['ID_code'], "target": preds})
print(result.head())

result.to_csv("submission.Hemal.Mar282019.4.csv", index=False)

