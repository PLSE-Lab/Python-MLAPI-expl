#!/usr/bin/env python
# coding: utf-8

# Lightgbm model - roc_auc 0.815 , feature importance visualization 
# 

# In[ ]:



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style(style='whitegrid')
sns.set(font_scale=1.5);
import re

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/clinvar-conflicting/clinvar_conflicting.csv', dtype={0: object, 38: str, 40: object})
print(df.columns)
print(df.shape)


# CLASS is target variable 
# 
# The CLASS distribution is skewed a bit to the 0 class, meaning there are fewer variants with conflicting submissions.
# 

# In[ ]:


ax = sns.countplot(x="CLASS", data=df)
ax.set(xlabel='CLASS', ylabel='Number of Variants');


# In[ ]:


import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


# In[ ]:


X = df.drop('CLASS',axis = 1)
y = df['CLASS']
categorical_features = [col for c, col in enumerate(X.columns)                         if not ( np.issubdtype(X.dtypes[c], np.number )  )  ]

len(categorical_features), X.shape, y.shape, y.mean() 


# In[ ]:


for f in categorical_features:
    X[f] = X[f].astype('category')

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 0, stratify = y  )
print(X_train.shape, X_test.shape)


# In[ ]:


# technical things: transform data into internal lightgbm format
train_data = lgb.Dataset(X_train, label=y_train , categorical_feature=categorical_features)
test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=categorical_features)


# In[ ]:


# Create lightgbm model builder.
# I found params by tuning 'num_leaves': 500, 'learning_rate': 0.0015, 

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 500,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.0015,
    'verbose': 0
}

model = lgb.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=5000,
                       early_stopping_rounds=100)


# In[ ]:


p = model.predict(X_test)
print('Test roc_auc_score = ', roc_auc_score(y_test, p ))    


# In[ ]:


#ax = plt.figure(figsize = (20,5))
fig, ax = plt.subplots(figsize=(20, 15))
lgb.plot_importance(model,ax.axes,  height = 1.6)
plt.show()


# In[ ]:


pd.Series(index = X.columns, data = model.feature_importance() , name = 'Importance').sort_values(ascending = False)


# In[ ]:




