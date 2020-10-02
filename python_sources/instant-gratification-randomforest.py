#!/usr/bin/env python
# coding: utf-8

# In[16]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestClassifier
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[17]:


df_train = pd.read_csv("../input/train.csv")


# In[32]:


df_test = pd.read_csv("../input/test.csv")


# In[19]:


df_train.head()


# In[20]:


df_train.shape


# In[21]:


X_train = df_train.iloc[:,:-1]
y_train = df_train['target']


# In[22]:


X_train.info()


# In[23]:


X_train.describe()


# In[24]:


for col,dt in X_train.dtypes.items():
    if dt == 'object':
        print(col)


# In[25]:


X_train = X_train.drop(columns=['id'])


# In[26]:


rnd = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, bootstrap = True, oob_score = True, n_jobs =-1)


# In[27]:


rnd.fit(X_train, y_train)


# In[33]:


X_test = df_test.drop(columns=['id'])
y_test = rnd.predict(X_test)


# In[29]:


y_test


# In[34]:


my_submission = pd.DataFrame({'Id': df_test['id'], 'Target': y_test})
my_submission.to_csv('submission.csv', index=False)


# In[ ]:




