#!/usr/bin/env python
# coding: utf-8

# In[14]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[5]:


get_ipython().run_line_magic('cd', '../input')
get_ipython().run_line_magic('ls', '')


# In[41]:


train = pd.read_csv('train.csv')
train.info()


# In[42]:


train.head()


# In[43]:


train.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train.head()


# In[44]:


train.isnull().sum()


# In[45]:


Counter(train['SibSp'])


# In[46]:


train['Pclass'] = train['Pclass'].astype('category')
train['Sex'] = train['Sex'].astype('category')
train['Pclass'] = train['Pclass'].astype('category')
train['Embarked'] = train['Embarked'].astype('category')


# In[47]:


import lightgbm as lgb
train_dataset = lgb.Dataset(train.drop('Survived', axis=1), label=train['Survived'])


# In[48]:


params_tuned = {
#     'bagging_freq': 5,
#    'bagging_fraction': 0.9,
#    'boost_from_average':'false',
    'boost': 'gbdt',
#     'feature_fraction': 0.05,
    'learning_rate': 0.2,
    'max_depth': -1,  
    'metric':'auc',
#    'min_data_in_leaf': 80,
#     'min_sum_hessian_in_leaf': 10.0,
#     'num_leaves': 13,
    'num_threads': -1,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1,
    'num_leaves': 10,
    "device" : "cpu"
#     ,"gpu_platform_id" : 0,
#     "gpu_device_id" : 0
}


# In[49]:


model = lgb.train(params_tuned, train_dataset,  verbose_eval = 100, num_boost_round=1000)


# In[51]:


pred = model.predict(train.drop('Survived', axis=1))


# In[57]:


train['pred'] = pred


# In[55]:


from sklearn import metrics
metrics.roc_auc_score(train['Survived'], pred)


# In[72]:


acc = []
max_ = 0
ind = 0
for i in range(0,101):
    th = i/100
    label = train['pred'].apply(lambda p: 1 if p > th else 0)
    acc.append(metrics.accuracy_score(train['Survived'], label))
    if acc[i] > max_:
        ind, max_ = i, acc[i]
print(ind, acc[ind])
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
plt.figure(figsize=(12,6))
plt.plot(range(0,101), acc)
plt.scatter(range(0,101), acc, s = 2.8,c = 'red')
plt.show()


# In[73]:


test = pd.read_csv('test.csv')
print(test.shape)
test.head()


# In[75]:


test['Pclass'] = test['Pclass'].astype('category')
test['Sex'] = test['Sex'].astype('category')
test['Pclass'] = test['Pclass'].astype('category')
test['Embarked'] = test['Embarked'].astype('category')


# In[77]:


test['Survived'] = model.predict(test.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis = 1))


# In[79]:


test['Survived'] = test['Survived'].apply(lambda p: 1 if p>0.4 else 0)


# In[81]:


Counter(test['Survived'])


# In[86]:


get_ipython().run_line_magic('cd', '')
get_ipython().run_line_magic('mkdir', 'results')
get_ipython().run_line_magic('cd', 'results')


# In[87]:



test[['PassengerId', 'Survived']].to_csv('submission.csv')

