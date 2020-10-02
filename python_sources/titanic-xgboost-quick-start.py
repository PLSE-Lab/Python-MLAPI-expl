#!/usr/bin/env python
# coding: utf-8

# # Quick Start with XGBoost
# * Beginner's guide to getting up and running with XGBoost
# * Please leave a like or comment!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.

import os
        
os.listdir('../input')


# In[ ]:


RANDOM_SEED = 42
nan_replace = -999


# ## Import data

# In[ ]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
sample_submission = pd.read_csv('../input/gender_submission.csv', index_col='PassengerId')

train.head()


# ## Drop features
# * Just keep simplest, most obvious features

# In[ ]:


train = train[['Survived','Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]
test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch']]

# 0/1 encode sex
train['Sex'] = train['Sex'].map({'female': 1, 'male': 0})
test['Sex'] = test['Sex'].map({'female': 1, 'male': 0})


train.head()


# ## Missing values (NaNs)
# * XGBoost handles missing values well, just replace with -999

# In[ ]:


#train = train.dropna()


# In[ ]:


train = train.fillna(nan_replace)
test = test.fillna(nan_replace)

assert(train.isnull().sum().sum() == 0)
assert(test.isnull().sum().sum() == 0)


# ## Fit XGBoost
# * [Docs](https://xgboost.readthedocs.io/en/latest/parameter.html)

# In[ ]:


y_train = train['Survived']
train = train.drop(['Survived'], axis=1)


# In[ ]:


import xgboost as xgb


# In[ ]:


clf = xgb.XGBClassifier(missing=nan_replace) # default parameters for now
clf.fit(train, y_train)


# ## Predict and submit

# In[ ]:


pred = clf.predict(test)

sample_submission['Survived'] = pred

sample_submission.to_csv('xgboost_baseline2.csv')


# In[ ]:




