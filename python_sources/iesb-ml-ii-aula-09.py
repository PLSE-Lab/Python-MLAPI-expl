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


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[ ]:


train.shape, test.shape


# In[ ]:


train.head(9).T


# In[ ]:


# Datatypes
train.dtypes.value_counts()


# In[ ]:


# Remove number types
train.select_dtypes(exclude='number').count()


# In[ ]:


train['target'].value_counts(normalize=True)


# In[ ]:


feats = [c for c in train.columns if c not in ['id', 'target']]


# In[ ]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# In[ ]:


#rf = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1)
#rf.fit(train[feats], train['target'])


# In[ ]:


df = train.sample(30000)
rf = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1)
rf.fit(df[feats], df['target'])


# In[ ]:


rf.oob_score_


# In[ ]:


preds = [1 if x > 0.5 else 0 for x in rf.oob_decision_function_[:, 1] ]


# In[ ]:


accuracy_score(df['target'], preds)


# In[ ]:


accuracy_score(df['target'], rf.oob_decision_function_[:, 1] >= .5)


# In[ ]:


pd.Series(rf.oob_decision_function_[:, 1]>=.5).value_counts()


# In[ ]:


from sklearn.metrics import roc_auc_score
roc_auc_score(df['target'], rf.oob_decision_function_[:, 1])


# In[ ]:


df = train.query('target == 0').sample(10000).append(train.query('target == 1').sample(10000))
df.target.value_counts()


# In[ ]:


rf = RandomForestClassifier(n_estimators=200, oob_score=True, n_jobs=-1)
rf.fit(df[feats], df['target'])


# In[ ]:


accuracy_score(df['target'], rf.oob_decision_function_[:, 1]>=.5) 


# In[ ]:


roc_auc_score(df['target'], rf.oob_decision_function_[:, 1])


# In[ ]:




