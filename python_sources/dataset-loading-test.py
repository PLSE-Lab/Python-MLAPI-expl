#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[10]:


# train dataset load using pandas
tr_numeric = pd.read_csv('../input/Bosch_train_numeric.csv')
tr_numeric.describe()


# In[9]:


tr_date = pd.read_csv('../input/Bosch_train_date.csv')
tr_date.describe()


# In[14]:


tr_categorical = pd.read_csv('../input/Bosch_train_categorical.csv', dtype='category')
tr_categorical.describe()


# In[11]:


# test dataset load using pandas
test_numeric = pd.read_csv('../input/Bosch_test_numeric.csv')
test_numeric.describe()


# In[12]:


test_date = pd.read_csv('../input/Bosch_test_date.csv')
test_date.describe()


# In[13]:


test_categorical = pd.read_csv('../input/Bosch_test_categorical.csv', dtype='category')
test_categorical.describe()

