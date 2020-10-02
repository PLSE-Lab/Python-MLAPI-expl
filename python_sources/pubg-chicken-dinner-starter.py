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


df_train=pd.read_csv('../input/train.csv')
df_train.head()


# In[ ]:


import matplotlib.pyplot as plt


# In[ ]:


print('Distribution of damage dealt')
print('{0:.4f}% players dealt zero damage'.format((df_train['damageDealt'] == 0).sum()/ df_train.shape[0]))
plt.hist(df_train['damageDealt'], bins=40);


# In[ ]:


print('Distribution of damage dealt')
print('{0:.4f}% players dealt zero damage'.format((df_train['DBNOs'] == 0).sum()/ df_train.shape[0]))
plt.hist(df_train['damageDealt'], bins=40);


# In[ ]:


df_train=df_train.drop(['groupId'],axis=1)
df_train.head()


# In[ ]:


y_train=df_train['winPlacePerc']
df_train=df_train.drop(['winPlacePerc'],axis=1)
df_train.shape


# In[ ]:


y_train.head()


# In[ ]:


#from sklearn.ensemble import RandomForestRegressor
import os


# In[ ]:


n=os.cpu_count()
#regr = RandomForestRegressor( n_jobs=n)
#regr.fit(df_train, y_train)


# In[ ]:


from sklearn.model_selection import train_test_split
X_train, X_val, y_Train, y_val = train_test_split(df_train, y_train, test_size=0.3)
del df_train,y_train


# In[ ]:


from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler().fit(X_train)


# In[ ]:


X_train = scaler.transform(X_train)


# In[ ]:


X_val = scaler.transform(X_val)


# #import xgboost as xgb
# #import lightgbm as lgb
# #dtrain = xgb.DMatrix(X_train.values, label=y_train.values)
# #dval= xgb.DMatrix(X_val.values, label=y_val.values)
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import AdaBoostRegressor
# '''
# params = {
# "max_depth" : 10,
# 'booster':'dart',
# 'eval_metric': ['mae'],
# }
# 
# param = {'num_leaves':31, 'num_trees':100, 'objective':'binary'}
# param['metric'] = 'auc'
# 
# train_data = lgb.Dataset(df_train, label=y_train)
# 
# num_round = 100
# '''
# regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),
#                           n_estimators=300, random_state=42)
# clf = regr_2.fit(df_train,y_train)

# 

# In[ ]:





# In[ ]:


df_test=pd.read_csv('../input/test.csv')


# In[ ]:


df_test.head()


# id=df_test['Id']
# df_test=df_test.drop(['Id'], axis=1)
# df_test.head()

# #dtest = xgb.DMatrix(df_test.values)
# preds= regr_2.predict(df_test)

# df_submission=pd.read_csv('../input/sample_submission.csv')
# 

# df_submission.head()

# df_submission.shape

# df_submission['Id']=id
# df_submission['winPlacePerc']=preds

# df_submission.head()

# df_submission.to_csv('subs.csv', index=False)

# In[ ]:




