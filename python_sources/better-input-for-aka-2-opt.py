#!/usr/bin/env python
# coding: utf-8

# I'm going to feed a better initial tour found by [LKH](http://akira.ruc.dk/~keld/research/LKH-3/) on my local machine to [my previous kernel](https://www.kaggle.com/kostyaatarik/close-ends-chunks-optimization-aka-2-opt).

# 1. Import all that we need.

# In[ ]:


import numpy as np
import pandas as pd
import numba
from sympy import isprime, primerange
from math import sqrt
from sklearn.neighbors import KDTree
from tqdm import tqdm
from itertools import combinations


# 2. Read input data and define some arrays that we'll need later.

# In[ ]:


cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)


# 3. Define fast scoring functions using numba.

# In[ ]:


@numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
def cities_distance(offset, id_from, id_to):
    xy_from, xy_to = XY[id_from], XY[id_to]
    dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
    distance = sqrt(dx * dx + dy * dy)
    if offset % 10 == 9 and is_not_prime[id_from]:
        return 1.1 * distance
    return distance


@numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
def score_chunk(offset, chunk):
    pure_distance, penalty = 0.0, 0.0
    penalty_modulo = 9 - offset % 10
    for path_index in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[path_index], chunk[path_index+1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
        if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
            penalty += distance
    return pure_distance + 0.1 * penalty


@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def score_path(path):
    return score_chunk(0, path)


# 4. Precompute close cities pairs using KDTree.

# In[ ]:


kdt = KDTree(XY)


# In[ ]:


pairs = set()
for city_id in tqdm(cities.index):
    dists, neibs = kdt.query([XY[city_id]], 31)
    for neib_id in neibs[0][1:]:
        if city_id and neib_id:  # skip pairs that include starting city 
            pairs.add(tuple(sorted((city_id, neib_id))))
    neibs = kdt.query_radius([XY[city_id]], 31, count_only=False, return_distance=False)
    for neib_id in neibs[0]:
        if city_id and neib_id and city_id != neib_id:
            pairs.add(tuple(sorted((city_id, neib_id))))

print(f'{len(pairs)} cities pairs are selected.')
# sort pairs by distance
pairs = np.array(list(pairs))
distances = np.sum((XY[pairs.T[0]] - XY[pairs.T[1]])**2, axis=1)
order = distances.argsort()
pairs = pairs[order]


# 5. Load the initial path to start optimization from. I'll use the one with a pure score of 1502650 found by [LKH](http://akira.ruc.dk/~keld/research/LKH-3/) on my local machine.

# In[ ]:


path = np.array(pd.read_csv('../input/traveling-santa-lkh-solution/pure1502650.csv').Path)
initial_score = score_path(path)


# 6. Use optimization described in [my previous kernel](https://www.kaggle.com/kostyaatarik/close-ends-chunks-optimization-aka-2-opt).

# In[ ]:


path_index = np.argsort(path[:-1])

total_score = initial_score
print(f'Total score is {total_score:.2f}.')
for _ in range(3):
    for step, (id1, id2) in enumerate(tqdm(pairs), 1):
        if step % 10**6 == 0:
            new_total_score = score_path(path)
            print(f'Score: {new_total_score:.2f}; improvement over last 10^6 steps: {total_score - new_total_score:.2f}; total improvement: {initial_score - new_total_score:.2f}.')
            total_score = new_total_score
        i, j = path_index[id1], path_index[id2]
        i, j = min(i, j), max(i, j)
        chunk, reversed_chunk = path[i-1:j+2], np.concatenate([path[i-1:i], path[j:i-1:-1], path[j+1:j+2]])
        chunk_score, reversed_chunk_score = score_chunk(i-1, chunk), score_chunk(i-1, reversed_chunk)
        if j - i > 2:
            chunk_abc = np.concatenate([path[i-1:i+1], path[j:i:-1], path[j+1:j+2]])
            chunk_acb = np.concatenate([path[i-1:i], path[j:j+1], path[i:j], path[j+1:j+2]])
            chunk_abcb = np.concatenate([path[i-1:i+1], path[j:j+1], path[i+1:j], path[j+1:j+2]])
            abc_score, acb_score, abcb_score = map(lambda chunk: score_chunk(i-1, chunk), [chunk_abc, chunk_acb, chunk_abcb])
            for chunk, score, name in zip((chunk_abc, chunk_acb, chunk_abcb), (abc_score, acb_score, abcb_score), ('abc', 'acb', 'abcb')):
                if score < chunk_score:
                    path[i-1:j+2] = chunk
                    path_index = np.argsort(path[:-1])  # update path index
                    chunk_score = score
        if reversed_chunk_score < chunk_score:
            path[i-1:j+2] = reversed_chunk
            path_index = np.argsort(path[:-1])  # update path index


# In[ ]:


print(f'Total improvement is {initial_score - total_score:.2f}.')


# 7. Save the result path.

# In[ ]:


def make_submission(name, path):
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)


# In[ ]:


make_submission(score_path(path), path)
