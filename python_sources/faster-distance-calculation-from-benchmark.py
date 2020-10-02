#!/usr/bin/env python
# coding: utf-8

# *not that we need to, but....
# *
# ### It seems we can go even faster with [numpy.einsum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.einsum.html)....
# from:
# > CPU times: user 140 ms, sys: 84 ms, total: 224 ms <br/>
# > Wall time: 220 ms
# 
# to:
# > CPU times: user 108 ms, sys: 4 ms, total: 112 ms <br/>
# > Wall time: 107 ms
# 
# ---------------------------------------------------------------------
# 
# 
# **borrowing from** [chanran's kernel](https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark)
# 
# ---------------------------------------------------------------------
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

# In[48]:


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


# In[49]:


train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
test_p_0 = test[['x_0', 'y_0', 'z_0']].values
test_p_1 = test[['x_1', 'y_1', 'z_1']].values

tr_a_min_b = train_p_0 - train_p_1
te_a_min_b = test_p_0 - test_p_1


# ### Using np.linalg.norm

# In[50]:


get_ipython().run_cell_magic('time', '', "# This block is SPPED UP\ntrain['dist_speedup'] = np.linalg.norm(tr_a_min_b, axis=1)\ntest['dist_speedup'] = np.linalg.norm(te_a_min_b, axis=1)")


# ### Using np.einsum

# In[51]:


get_ipython().run_cell_magic('time', '', "# This block is SPED UP a little more\n# np.sqrt(np.einsum('ij,ij->i', a_min_b, a_min_b))\ntrain['dist_speedup_einsum'] = np.sqrt(np.einsum('ij,ij->i', tr_a_min_b, tr_a_min_b))\ntest['dist_speedup_einsum'] = np.sqrt(np.einsum('ij,ij->i', te_a_min_b, te_a_min_b))")


# ### Verify the Results

# In[ ]:


train.head()


# In[ ]:




