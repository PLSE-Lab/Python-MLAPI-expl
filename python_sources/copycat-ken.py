#!/usr/bin/env python
# coding: utf-8

# ![Copycat Ken](https://i.ytimg.com/vi/ByH13tdosnM/maxresdefault.jpg)

# Why is the inclusion of an introductory header image a 'soft requirement' for kernels these days, lol? Guess the anime.

# In[ ]:


import numpy as np
import pandas as pd
import gc

tr = pd.read_csv('/kaggle/input/ashrae-energy-prediction/train.csv')
te = pd.read_csv('/kaggle/input/ashrae-energy-prediction/test.csv')

a = set(tr.building_id)
b = set(te.building_id)
len(a-b), len(b-a)


# > Cool, each test building exists in train.

# In[ ]:


del a,b; gc.collect()
tr.timestamp.min(), tr.timestamp.max()


# If we drop year, we don't have to worry about collisions... since we only have a max of 1 unique reading per hour per meter per building.

# In[ ]:


tr.timestamp = tr.timestamp.map(lambda x: x[5:])
te.timestamp = te.timestamp.map(lambda x: x[5:])
te = te.merge(
    tr[['building_id','meter','timestamp','meter_reading']],
    how='left',
    on=['building_id','meter','timestamp']
)


# In[ ]:


del tr; gc.collect()
(te.meter_reading.isna().sum() / te.shape[0] * 100).astype(np.int)


# Small amount of nans we need to handle...

# In[ ]:


fillna = te.groupby(['building_id', 'meter']).meter_reading.mean().reset_index()
fillna.rename(columns={'meter_reading':'missing'}, inplace=True)
te = te.merge(fillna, how='left', on=['building_id', 'meter'])

mask = te.meter_reading.isna()
te.loc[mask, 'meter_reading'] = te[mask].missing
(te.meter_reading.isna().sum() / te.shape[0] * 100).astype(np.int)


# In[ ]:


te[['row_id', 'meter_reading']].to_csv('./submission.csv', index=False)


# In[ ]:




