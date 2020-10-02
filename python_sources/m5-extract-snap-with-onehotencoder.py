#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import preprocessing


# Based on [my previous notebook](https://www.kaggle.com/sergioli212/m5-all-feature-engineering-ready-to-use) for Ready-to-use features. I make correction for SNAP.

# For each state, the policy for SNAP is different. So just keep the value of the corresponding state for each observation. After I try out the conditional statements which is too slow. Just thinking to do it using One hot encoder plus matrix multiplication.

# In[ ]:


# Conditional statements(Slow)
# def get_SNAP_for_store(row):
#     print(row)
#     if row['state_id'] == 'CA':
#         return row['snap_CA']
#     if row['state_id'] == 'TX':
#         return row['snap_TX']
#     if row['state_id'] == 'WI':
#         return row['snap_WI']
# grid_df['snap'] = grid_df.apply(lambda row: get_SNAP_for_store(row), axis=1)


# In[ ]:


grid_df = pd.read_pickle('../input/m5-sample50/grid_df50.pkl')
grid_df = grid_df[['snap_CA', 'snap_TX', 'snap_WI', 'state_id']]
grid_df.head()


# In[ ]:


enc = preprocessing.OneHotEncoder()
state = enc.fit_transform(grid_df[['state_id']]).toarray()
state[:5]


# In[ ]:


grid_df['snap'] = np.multiply(grid_df[['snap_CA', 'snap_TX', 'snap_WI']].values,state).sum(axis=1)
grid_df

