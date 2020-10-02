#!/usr/bin/env python
# coding: utf-8

# In[10]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.ensemble import RandomForestRegressor
import gc
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[11]:


train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")


# In[12]:


train.head()


# In[13]:


train.describe()


# In[14]:


cols = ['parent_category_name', 'category_name', 'price', 'user_type', 'item_seq_number', 'image_top_1']
dummy_cols = ['parent_category_name', 'category_name','user_type']
y = train['deal_probability'].copy()
x_train = train[cols].copy().fillna(0)
x_test  = test[cols].copy().fillna(0)
del train, test; gc.collect()

n = len(x_train)
x = pd.concat([x_train, x_test])
x = pd.get_dummies(x, columns=dummy_cols)
x.head()


# In[ ]:


x_train = x.iloc[:n, :]
x_test = x.iloc[n:, :]
del x; gc.collect()


# In[ ]:


clf = RandomForestRegressor()
clf.fit(x_train, y)
y_pred = clf.predict(x_test)
y_pred


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


sub['deal_probability'] = y_pred


# In[ ]:


sub.to_csv('simple_mean_benchmark.csv', index=False)

