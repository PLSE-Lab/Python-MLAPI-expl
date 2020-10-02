#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


train = pd.read_csv('../input/nfl-big-data-bowl-2020/train.csv')


# In[ ]:


train['NflId'] = train['NflId'].astype('|S')

train = pd.concat([train.drop(['NflId'], axis=1), pd.get_dummies(train['NflId'], prefix='NflId')], axis=1)
dummy_col = train.columns


# In[ ]:


train.shape


# So, we can see that this simple procedure cleans errors in the 'Stadium' field and reduces the number of unique stadiums from virtual 55 to the real 37.
