#!/usr/bin/env python
# coding: utf-8

# Country having maximum number of accidents

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('/kaggle/input/us-accidents/US_Accidents_Dec19.csv')


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


df.info()


# In[ ]:


null_count = df.isnull().sum()
null_count[null_count > 0]


# In[ ]:


state_wise_count = df.groupby('State')['ID'].count().reset_index()
state_wise_count.head()


# In[ ]:


state_wise_count.shape


# In[ ]:


state_wise_count = state_wise_count.sort_values(by = 'ID', ascending = False)


# In[ ]:


state_wise_count.head()


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
sns.set(style = 'whitegrid')


# In[ ]:


f, ax = plt.subplots(figsize = (6,15))
sns.barplot(y = 'State', x = 'ID', data = state_wise_count)


# In[ ]:




