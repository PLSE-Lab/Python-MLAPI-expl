#!/usr/bin/env python
# coding: utf-8

# In[37]:


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


# In[38]:


# Carregar os arquivos
train = pd.read_csv('../input/train.csv', parse_dates=[0])
test = pd.read_csv('../input/test.csv', parse_dates=[0])

train.shape, test.shape


# In[39]:


train['count'] = np.log(train['count'])


# In[40]:


df = pd.concat([train, test], sort=False)


# In[41]:


df.info()


# In[42]:


df.loc[2]


# In[43]:


# Resetando o indice
df = df.reset_index(drop=True)


# In[44]:


df.loc[2]


# In[45]:


df['year'] = df.datetime.dt.year
df['month'] = df.datetime.dt.month
df['weekday'] = df.datetime.dt.weekday
df['day'] = df.datetime.dt.day
df['hour'] = df.datetime.dt.hour


# In[46]:


train = df[~df['count'].isnull()]
test = df[df['count'].isnull()]


# In[47]:


train.shape, test.shape


# In[48]:


from sklearn.model_selection import train_test_split

train, valid = train_test_split(train, random_state=42)

train.shape, valid.shape


# In[49]:


removed_col = ['count', 'casual', 'registered', 'datetime']
feats = [c for c in train.columns if c not in removed_col]


# In[50]:





# In[ ]:




