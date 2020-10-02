#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.metrics import mean_squared_error # metrics

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# In[5]:


train.shape


# In[6]:


test.shape


# In[7]:


pred_test = np.array([train['target'].mean() for i in range(test.shape[0])])


# In[9]:


pred_train = np.array([train['target'].mean() for i in range(train.shape[0])])


# In[10]:


mean_squared_error(y_pred=pred_train, y_true= train['target'])


# In[4]:


np.save('prediction.npy',pred_train)


# In[ ]:




