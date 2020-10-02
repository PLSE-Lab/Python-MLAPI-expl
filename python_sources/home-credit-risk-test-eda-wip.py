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


# In[2]:


import os
print(os.listdir("../input"))


# import libraries

# In[4]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[5]:


test = pd.read_csv('../input/application_test.csv')


# In[6]:


test.columns.values


# In[8]:


test.head()


# In[9]:


test.info()


# In[10]:


test.describe()


# snapshot

# In[11]:


print("test -  rows:",test.shape[0]," columns:", test.shape[1])


# Missing data

# In[12]:


def missing_data(data):
    total = data.isnull().sum().sort_values(ascending = False)
    percent = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)
    return pd.concat([total, percent], axis=1, keys=['Total', 'Percent'])


# print missing data

# In[13]:


missing_data(test)


# In[ ]:




