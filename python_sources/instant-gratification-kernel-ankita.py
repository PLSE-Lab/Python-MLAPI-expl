#!/usr/bin/env python
# coding: utf-8

# In[13]:


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


# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


# In[3]:


pd.set_option('display.max_columns', None)  # or 1000
pd.set_option('display.max_rows', None)  # or 1000
pd.set_option('display.max_colwidth', -1)  # or 199


# In[15]:


df_train = pd.read_csv("../input/train.csv")
df_train.head()


# In[16]:


df_train["target"].value_counts()


# In[17]:


df_train.shape


# In[18]:


df_test = pd.read_csv("../input/test.csv")
df_test.head()


# In[19]:


df_train.describe()


# In[20]:


X_train = df_train.iloc[:,1:-1]
y_train = df_train['target']


# In[21]:


randomforest = RandomForestClassifier(n_estimators = 500, max_leaf_nodes = 16, bootstrap = True, oob_score = True, n_jobs =-1)


# In[ ]:


randomforest.fit(X_train, y_train)


# In[ ]:


X_test = df_test.drop(columns=['id'])

X_test.shape
y_test = randomforest.predict(X_test)


# In[ ]:


y_test


# In[ ]:


my_submission = pd.DataFrame({'Id': df_test['id'], 'Target': y_test})


# In[ ]:


my_submission.to_csv('submission.csv', index=False)


# In[ ]:




