#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import os
import pandas_profiling


# In[ ]:


train = pd.read_csv("../input/train.csv")
train.head()


# In[ ]:


test = pd.read_csv("../input/test.csv")
test.head()


# In[ ]:


pandas_profiling.ProfileReport(train)


# In[ ]:


pandas_profiling.ProfileReport(test)


# In[ ]:


feat_with_missing = ['meaneduc', 'rez_esc', 'v18q1', 'v2a1', 'SQBmeaned']


# In[ ]:


train[feat_with_missing].isnull().sum()/train.shape[0]


# In[ ]:


test[feat_with_missing].isnull().sum()/test.shape[0]


# In[ ]:


columns = train.columns[1:-1]
train_test = pd.concat([train[columns], test[columns]], axis=0)


# In[ ]:


train .fillna({'meaneduc': train_test.meaneduc.mean()}, inplace=True)
test.fillna({'meaneduc': train_test.meaneduc.mean()}, inplace=True)


# In[ ]:


train.fillna({'SQBmeaned': train_test.SQBmeaned.mean()}, inplace=True)
test.fillna({'SQBmeaned': train_test.SQBmeaned.mean()}, inplace=True)


# In[ ]:


train.drop(columns=['rez_esc', 'v18q1', 'v2a1'], inplace=True)
test.drop(columns=['rez_esc', 'v18q1', 'v2a1'], inplace=True)

