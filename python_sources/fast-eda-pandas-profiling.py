#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.prototype._should_scroll = function(lines) {\n    return false;\n}')


# In[ ]:



# Importing the Required Libraries

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling

# Reading all the files


# In[ ]:



train = pd.read_csv("../input/train.csv")

test = pd.read_csv("../input/test.csv")

structures = pd.read_csv('../input/structures.csv')

scalar_coupling_contributions = pd.read_csv('../input/scalar_coupling_contributions.csv')

train.head(5)

structures.head(5)

scalar_coupling_contributions.head(5)


# In[ ]:



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


# In[ ]:



# This block is SPPED UP

train_p_0 = train[['x_0', 'y_0', 'z_0']].values
train_p_1 = train[['x_1', 'y_1', 'z_1']].values
test_p_0 = test[['x_0', 'y_0', 'z_0']].values
test_p_1 = test[['x_1', 'y_1', 'z_1']].values

train['dist_speedup'] = np.linalg.norm(train_p_0 - train_p_1, axis=1)
test['dist_speedup'] = np.linalg.norm(test_p_0 - test_p_1, axis=1)


# In[ ]:


pandas_profiling.ProfileReport(train)


# In[ ]:




