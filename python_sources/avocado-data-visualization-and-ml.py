#!/usr/bin/env python
# coding: utf-8

# # Price analysis of Avocado
# Analysis of Avocado data based on date, month, years, type.

# ![![image.png](attachment:image.png)](https://i.pinimg.com/originals/b5/3c/06/b53c061ae4ee0f1e3f34c84657702468.gif)

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


# Importing the data

# In[ ]:


df = pd.read_csv("../input/avocado.csv")
df.head()


# In[ ]:


df.shape


# In[ ]:


df.info()


# In[ ]:


df.columns


# In[ ]:


df.describe()


# In[ ]:


df.isnull().sum()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
from pandas import datetime

df['Date'] = pd.to_datetime(df['Date'])

mask = df['type']=='conventional'
g = sns.factorplot('Date','AveragePrice',data=df[mask],
                   hue='year',
                   size=10,
                   aspect=0.8,
                   palette='vlag',
                   join = False
              )


# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])

mask = df['type']=='organic'
g = sns.factorplot('Date','AveragePrice',data=df[mask],
                   hue='year',
                   size=10,
                   aspect=0.8,
                   palette='RdGy',
                   join = False
                  )


# In[ ]:


df['Month'] = df['Date'].dt.month
mask = df['type']=='conventional'
g = sns.factorplot('Month','region',data=df[mask],
                   hue='year',
                   size=10,
                   aspect=0.8,
                   palette='RdBu_r',
                   join = False
              )


# In[ ]:


df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
mask = df['type']=='organic'
g = sns.factorplot('Month','region',data=df[mask],
                   hue='year',
                   size=10,
                   aspect=0.8,
                   palette='rainbow',
                   join = False
              )


# In[ ]:


g = sns.factorplot('AveragePrice','region',data = df,
                   hue='type',
                   size=13,
                   aspect=0.8,
                   palette='winter',
                   join=False,
              )


# In[ ]:


mask = df['type']=='conventional'
g = sns.factorplot('AveragePrice','region',data=df[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='prism',
                   join=False,
              )


# In[ ]:


mask = df['type']=='organic'
g = sns.factorplot('AveragePrice','region',data=df[mask],
                   hue='year',
                   size=13,
                   aspect=0.8,
                   palette='magma',
                   join=False,
              )


# In[ ]:


df_org = df.copy()
df_org["type"] = df_org["type"].replace({"conventional":1, "organic":0})
df_org = df_org.drop(columns = 'region')
df_org.head()


# In[ ]:


X = df_org.drop(columns = ['type','Date','Unnamed: 0'])
y = df_org.type


# # Train Test Split

# In[ ]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


# # RandomForest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
model_rf = RandomForestClassifier(n_estimators=1000 , oob_score = True, n_jobs = -1,
                                  random_state =50, max_features = "auto",
                                  max_leaf_nodes = 30)
model_rf.fit(x_train, y_train)

# Make predictions
prediction_test = model_rf.predict(x_test)
print (metrics.accuracy_score(y_test, prediction_test))


# In[ ]:


importances = model_rf.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')


# # XGBoost Classifier

# In[ ]:


from xgboost import XGBClassifier
xgb = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
                    colsample_bytree=1, gamma=0, learning_rate=0.9, max_delta_step=0,
                    max_depth = 7, min_child_weight=1, missing=None, n_estimators=100,
                    n_jobs=1, nthread=None, objective='binary:logistic', random_state=0,
                    reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
                    silent=True, subsample=1)
xgb.fit(x_train, y_train)
prediction_test1 = xgb.predict(x_test)
print (metrics.accuracy_score(y_test, prediction_test1))


# In[ ]:


importances = xgb.feature_importances_
weights = pd.Series(importances,
                 index=X.columns.values)
weights.sort_values()[-10:].plot(kind = 'barh')

