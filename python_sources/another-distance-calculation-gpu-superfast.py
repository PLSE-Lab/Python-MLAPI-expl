#!/usr/bin/env python
# coding: utf-8

# This code builds on these two improvements to Inversion's benchmark kernel:
#   - https://www.kaggle.com/seriousran/just-speed-up-calculate-distance-from-benchmark
# 
#   - https://www.kaggle.com/rakibilly/faster-distance-calculation-from-benchmark
# 
# Not that we need anything faster than 400ms, but I wanted to see what a GPU could do. I used [RAPIDS](https://rapids.ai) cuDF, which is a GPU DataFrame library. It's pretty cool from what I've seen so far. You can use an API similar to Pandas and get a 16x speedup over numpy functions!

# ### Setup

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

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


# ### Speed Tests
# 
# #### 0. Original Benchmark code - pandas.DataFrame.apply()

# In[ ]:


# Engineer a single feature: distance vector between atoms
#  (there's ways to speed this up!)

# def dist(row):
#     return ( (row['x_1'] - row['x_0'])**2 +
#              (row['y_1'] - row['y_0'])**2 +
#              (row['z_1'] - row['z_0'])**2 ) ** 0.5

# train['dist'] = train.apply(lambda x: dist(x), axis=1)
# test['dist'] = test.apply(lambda x: dist(x), axis=1)

# takes 7+ minutes per run


# #### 1. numpy.linalg()

# In[ ]:


get_ipython().run_cell_magic('timeit', '', "# This block is SPED UP\n\ntrain_p_0 = train[['x_0', 'y_0', 'z_0']].values\ntrain_p_1 = train[['x_1', 'y_1', 'z_1']].values\ntest_p_0 = test[['x_0', 'y_0', 'z_0']].values\ntest_p_1 = test[['x_1', 'y_1', 'z_1']].values\n\ntr_a_min_b = train_p_0 - train_p_1\nte_a_min_b = test_p_0 - test_p_1\ntrain['dist_speedup'] = np.linalg.norm(tr_a_min_b, axis=1)\ntest['dist_speedup'] = np.linalg.norm(te_a_min_b, axis=1)")


#    #### 2. numpy.einsum()

# In[ ]:


get_ipython().run_cell_magic('timeit', '', "# This block is SPED UP a little more\n\ntrain_p_0 = train[['x_0', 'y_0', 'z_0']].values\ntrain_p_1 = train[['x_1', 'y_1', 'z_1']].values\ntest_p_0 = test[['x_0', 'y_0', 'z_0']].values\ntest_p_1 = test[['x_1', 'y_1', 'z_1']].values\n\ntr_a_min_b = train_p_0 - train_p_1\nte_a_min_b = test_p_0 - test_p_1\ntrain['dist_speedup_einsum'] = np.sqrt(np.einsum('ij,ij->i', tr_a_min_b, tr_a_min_b))\ntest['dist_speedup_einsum'] = np.sqrt(np.einsum('ij,ij->i', te_a_min_b, te_a_min_b))")


# #### 3. cudf.DataFrame

# In[ ]:


# Install rapids component, cuDF
# Takes a few minutes...
get_ipython().system(' conda update --force-reinstall conda -y')

get_ipython().system('conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults     cudf=0.7 python=3.6 cudatoolkit=10.0 -y')

import cudf

# Convert existing Pandas df since it exists already
gputrain = cudf.DataFrame.from_pandas(train)


# In[ ]:


get_ipython().run_cell_magic('timeit', '', "#This block is STRAIGHT UP ACCELERATED\n\ndef get_dist(df):\n    return np.sqrt((df.x_1-df.x_0)**2 +\n                   (df.y_1-df.y_0)**2 +\n                   (df.z_1-df.z_0)**2)\n\ngputrain['dist_rapids'] = get_dist(gputrain)")


# ### Verify the Results

# In[ ]:


print(gputrain[['dist_speedup', 'dist_speedup_einsum','dist_rapids']].head())


# Good stuff! Anybody else using Rapids?
