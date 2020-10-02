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


# **when** the earthquake will take place  
# predict the time remaining (**time_to_failure**) from real-time seismic data (**acoustic_data**)  
# evaluated using the **mean absolute error**  
# the time between the **last** row of the segment (**seg_id**) and the next laboratory earthquake

# In[ ]:


# first 3 segments
train = pd.read_csv('../input/train.csv', nrows=450000)


# In[ ]:


pd.options.display.precision


# In[ ]:


pd.options.display.precision = 18


# In[ ]:


train.head(150000) #.info()


# In[11]:


import seaborn as sns
sns.set()


# In[ ]:


# 1st segment
ax = sns.lineplot(x=np.arange(150000), y=train['acoustic_data'][:150000])


# In[ ]:


# 2nd segment
ax = sns.lineplot(x=np.arange(150000), y=train['acoustic_data'][150000:300000])


# In[ ]:


# 3rd segment
ax = sns.lineplot(x=np.arange(150000), y=train['acoustic_data'][300000:450000])


# In[ ]:


train.dtypes #.value_counts()


# In[ ]:


train.nunique() #.sort_values()


# In[ ]:


train.acoustic_data.value_counts()


# In[ ]:


pd.options.display.max_rows


# In[ ]:


pd.options.display.max_rows = 250


# In[ ]:


train.acoustic_data.value_counts().sort_index(ascending=False)


# In[ ]:


ax = sns.lineplot(train.acoustic_data.value_counts().index, train.acoustic_data.value_counts())


# In[ ]:


ax = sns.distplot(train.acoustic_data, bins=20, kde=False)


# In[ ]:


train.duplicated().sum()


# In[ ]:


train.isna().sum() #.sort_values(ascending=False)


# In[ ]:


train.eq(0).sum() #.sort_values(ascending=False)


# In[ ]:


train.lt(0).sum() #.sort_values(ascending=False)


# In[ ]:


train.min() #.sort_values(ascending=False)


# In[ ]:


train.max() #.sort_values(ascending=False)


# In[ ]:


del train


# In[2]:


# all targets
rows = 150000
target = pd.read_csv('../input/train.csv', usecols=[1]).iloc[rows-1::rows, 0]


# In[22]:


ax = sns.lineplot(x=np.arange(target.size), y=target)


# In[23]:


target.diff().gt(0).sum()


# In[24]:


target.mean()


# In[18]:


target.min()


# In[25]:


target.max()


# In[26]:


ax = sns.distplot(target, bins=20, kde=False)


# In[27]:


ax = sns.boxplot(target)

