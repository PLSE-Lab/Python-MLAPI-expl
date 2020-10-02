#!/usr/bin/env python
# coding: utf-8

# In [Atomic Distance Benchmark](https://www.kaggle.com/inversion/atomic-distance-benchmark/output) kernel by [inversion](https://www.kaggle.com/inversion), <br/> I found '#(there's ways to speed this up!)'.
# I tried to find the faster way to calculate distance and share it. <br/>
# Let's research FASTER :)
# 
# From
# > CPU times: user 7min 19s, sys: 9.25 s, total: 7min 28s <br/>
# > Wall time: 7min 28s
# 
# To
# > CPU times: user 412 ms, sys: 828 ms, total: 1.24 s <br/>
# > Wall time: 1.23 s

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

import os
print(os.listdir("../input"))

from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

train = pd.read_csv('../input/train.csv', index_col='id')
test = pd.read_csv('../input/test.csv', index_col='id')

structures = pd.read_csv('../input/structures.csv')

def map_atom_info(df, atom_idx):
    df = pd.merge(df, structures, how = 'left',
                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],
                  right_on = ['molecule_name',  'atom_index'])
    
    df = df.drop('atom_index', axis=1)
    df = df.rename(columns={'atom': f'atom_{atom_idx}',
                            'x': f'x_{atom_idx}',
                            'y': f'y_{atom_idx}',
                            'z': f'z_{atom_idx}'})
    return df

train = map_atom_info(train, 0)
train = map_atom_info(train, 1)

test = map_atom_info(test, 0)
test = map_atom_info(test, 1)


# # Calculate distance

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Engineer a single feature: distance vector between atoms\n#  (there's ways to speed this up!)\n\ndef dist(row):\n    return ( (row['x_1'] - row['x_0'])**2 +\n             (row['y_1'] - row['y_0'])**2 +\n             (row['z_1'] - row['z_0'])**2 ) ** 0.5\n\ntrain['dist'] = train.apply(lambda x: dist(x), axis=1)\ntest['dist'] = test.apply(lambda x: dist(x), axis=1)")


# In[ ]:


get_ipython().run_cell_magic('time', '', "# This block is SPPED UP\n\ntrain_p_0 = train[['x_0', 'y_0', 'z_0']].values\ntrain_p_1 = train[['x_1', 'y_1', 'z_1']].values\ntest_p_0 = test[['x_0', 'y_0', 'z_0']].values\ntest_p_1 = test[['x_1', 'y_1', 'z_1']].values\n\ntrain['dist_speedup'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)\ntest['dist_speedup'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)")


# In[ ]:


train.head()


# In[ ]:




