#!/usr/bin/env python
# coding: utf-8

# In [Just speed up calculate distance from Benchmark](https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark) kernel by [Chanran Kim](https://www.kaggle.com/seriousran), <br/> a faster way to calculate distance is provided. <br/>
# 
# However, the computation requires data preprocessing.  If we include data preprocessing then there is a way to gain about 2x speedup again.
# 
# From
# > CPU times: user 10 s, sys: 8.75 s, total: 18.8 s
# > Wall time: 18.8 s
# 
# To
# 
# > CPU times: user 6.43 s, sys: 2.82 s, total: 9.25 s
# > Wall time: 9.14 s

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


# # Calculate distance Chanran Kim way
# 

# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='id')
test = pd.read_csv('../input/test.csv', index_col='id')

structures = pd.read_csv('../input/structures.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# This block is SPPED UP\n\ndef map_atom_info(df, atom_idx):\n    df = pd.merge(df, structures, how = 'left',\n                  left_on  = ['molecule_name', f'atom_index_{atom_idx}'],\n                  right_on = ['molecule_name',  'atom_index'])\n    \n    df = df.drop('atom_index', axis=1)\n    df = df.rename(columns={'atom': f'atom_{atom_idx}',\n                            'x': f'x_{atom_idx}',\n                            'y': f'y_{atom_idx}',\n                            'z': f'z_{atom_idx}'})\n    return df\n\ntrain = map_atom_info(train, 0)\ntrain = map_atom_info(train, 1)\n\ntest = map_atom_info(test, 0)\ntest = map_atom_info(test, 1)\n\ntrain_p_0 = train[['x_0', 'y_0', 'z_0']].values\ntrain_p_1 = train[['x_1', 'y_1', 'z_1']].values\ntest_p_0 = test[['x_0', 'y_0', 'z_0']].values\ntest_p_1 = test[['x_1', 'y_1', 'z_1']].values\n\ntrain['dist_speedup'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)\ntest['dist_speedup'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)")


# # Calculate distance via faster method

# Let's reload data to ensure fair comparison

# In[ ]:


train = pd.read_csv('../input/train.csv', index_col='id')
test = pd.read_csv('../input/test.csv', index_col='id')

structures = pd.read_csv('../input/structures.csv')


# In[ ]:


get_ipython().run_cell_magic('time', '', "# This block is SPPED UP\ndef add_dist(train, structures=structures):\n    dist = (train[['molecule_name', 'atom_index_0']].merge(structures, how='left', \n                    left_on=['molecule_name', 'atom_index_0'], \n                    right_on=['molecule_name', 'atom_index'])[['x', 'y', 'z'] ]\n        -\n        train[['molecule_name', 'atom_index_1']].merge(structures, how='left', \n                    left_on=['molecule_name', 'atom_index_1'], \n                    right_on=['molecule_name', 'atom_index'])[['x', 'y', 'z'] ]\n       )\n    train['dist_speed'] = np.linalg.norm(dist, axis=1)\n    \nadd_dist(train)\nadd_dist(test)")


# It is always possible to improve code efficiency ;)
