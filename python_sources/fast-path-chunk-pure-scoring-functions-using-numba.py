#!/usr/bin/env python
# coding: utf-8

# ### 1. Define scoring functions

# #### Import all the stuff

# In[ ]:


import numpy as np
import pandas as pd
import numba
from sympy import isprime
from math import sqrt


# #### Read the cities

# In[ ]:


cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])


# #### Define scoring functions

# In[ ]:


XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
is_not_prime = np.array([not isprime(city_id) for city_id in cities.index], dtype=np.int32)

@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def pure_score(path):
    '''Pure path score without penalties.'''
    dist = 0.0
    for i in numba.prange(path.shape[0] - 1):
        a, b = XY[path[i]], XY[path[i+1]]
        dx, dy = a[0] - b[0], a[1] - b[1]
        dist += sqrt(dx * dx + dy * dy)
    return dist


@numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
def chunk_score(start_offset, chunk):
    '''Score of path's chunk that starts at index 'start_offset'.'''
    dist = 0.0
    penalty = 0.0
    penalty_modulo = 9 - start_offset % 10
    for i in numba.prange(chunk.shape[0] - 1):
        id_a = chunk[i]
        a, b = XY[id_a], XY[chunk[i+1]]
        dx, dy = a[0] - b[0], a[1] - b[1]
        d = sqrt(dx * dx + dy * dy)
        dist += d
        if i % 10 == penalty_modulo and is_not_prime[id_a]:
            penalty += d
    return dist + 0.1 * penalty


@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def path_score(path):
    return chunk_score(0, path)


# ### 2. Test scoring functions' performance

# #### Define some silly Rudolph's path

# In[ ]:


path = np.concatenate([cities.index, [0]])


# #### Measure functions' performance
# 

# In[ ]:


get_ipython().run_line_magic('timeit', 'pure_score(path)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'path_score(path)')


# In[ ]:


get_ipython().run_line_magic('timeit', 'chunk_score(42, path[42:2019])')

