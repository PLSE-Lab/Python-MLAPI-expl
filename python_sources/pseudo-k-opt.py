#!/usr/bin/env python
# coding: utf-8

# <h3>Introduction</h3>
# 
# In this kernel we search for combinations of K cities that are close to each node and split the paths between them. After that, the idea is to try different permutations of the resulting chunks and check if the sub-tour is shorter. This work is a small improvement over [Atarik's kernel](https://www.kaggle.com/kostyaatarik/not-a-3-and-3-halves-opt) and I've also included some comments with the code.

# In[ ]:


import numpy as np
import pandas as pd
import numba
from sympy import isprime, primerange
from math import sqrt
from sklearn.neighbors import KDTree
from tqdm import tqdm
from itertools import combinations, permutations
from functools import lru_cache


# Read input data and set parameters:

# In[ ]:


# Parameters
K = 4
NUM_NEIGHBORS = 5
RADIUS = 0 # not using due to time limit

cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)


# Define fast scoring functions using numba.

# In[ ]:


@numba.jit('f8(i8, i8)', nopython=True, parallel=False)
def euclidean_distance(id_from, id_to):
    xy_from, xy_to = XY[id_from], XY[id_to]
    dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
    return sqrt(dx * dx + dy * dy)


@numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
def cities_distance(offset, id_from, id_to):
    """Euclidean distance with prime penalty."""
    distance = euclidean_distance(id_from, id_to)
    if offset % 10 == 9 and is_not_prime[id_from]:
        return 1.1 * distance
    return distance


@numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
def score_chunk(offset, chunk):
    """Return the total score (distance) for a chunk (array of cities)."""
    pure_distance, penalty = 0.0, 0.0
    penalty_modulo = 9 - offset % 10
    for path_index in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[path_index], chunk[path_index+1]
        distance = euclidean_distance(id_from, id_to)
        pure_distance += distance
        if path_index % 10 == penalty_modulo and is_not_prime[id_from]:
            penalty += distance
    return pure_distance + 0.1 * penalty


@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def score_path(path):
    return score_chunk(0, path)


@numba.jit
def chunk_scores(chunk):
    scores = np.zeros(10)
    pure_distance = 0
    for i in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[i], chunk[i+1]
        distance = euclidean_distance(id_from, id_to)
        pure_distance += distance
        if is_not_prime[id_from]:
            scores[9-i%10] += distance
    scores *= 0.1
    scores += pure_distance
    return scores


def score_compound_chunk(offset, head, chunks, tail, scores, indexes_permutation=None):
    """
    Return the total distance for the path formed by all chunks in the
    order defined by the last argument.
    """
    if indexes_permutation is None:
        indexes_permutation = range(len(chunks))
    score = 0.0
    last_city_id = head
    for index in indexes_permutation:
        chunk, chunk_scores = chunks[index], scores[index]
        score += cities_distance(offset % 10, last_city_id, chunk[0])
        score += chunk_scores[(offset + 1) % 10]
        last_city_id = chunk[-1]
        offset += len(chunk)
    return score + cities_distance(offset % 10, last_city_id, tail)


# Find close neighbors for each city using KDTree and add all possible combinations with size K to the candidates set structure

# In[ ]:


kdt = KDTree(XY)
candidates = set()
for city_id in tqdm(cities.index):
    # Find N nearest neighbors
    dists, neibs = kdt.query([XY[city_id]], NUM_NEIGHBORS)
    for candidate in combinations(neibs[0], K):
        if all(candidate):
            candidates.add(tuple(sorted(candidate)))
    # Also add all cities in a given range (radius)
    neibs = kdt.query_radius([XY[city_id]], RADIUS, count_only=False, return_distance=False)
    for candidate in combinations(neibs[0], K):
        if all(candidate):
            candidates.add(tuple(sorted(candidate)))

print("{} groups of {} cities are selected.".format(len(candidates), K))

# sort candidates by distance
@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def sum_distance(ids):
    res = 0
    for i in numba.prange(len(ids)):
        for j in numba.prange(i + 1, len(ids)):
            res += cities_distance(0, ids[i], ids[j])
    return res

candidates = np.array(list(candidates))
distances = np.array(list(map(sum_distance, tqdm(candidates))))
order = distances.argsort()
candidates = candidates[order]


# Read the initial tour from [this kernel](https://www.kaggle.com/kostyaatarik/not-a-5-and-5-halves-opt).

# In[ ]:


path = pd.read_csv('../input/santatour1515602/1515602.86080513.csv', squeeze=True).values
initial_score = score_path(path)
print("Initial tour distance (score): {:.2f}".format(initial_score))


# Loop trough all candidates groups and try to permutate the path between the cities in each one of them.

# In[ ]:


def not_trivial_permutations(iterable):
    perms = permutations(iterable)
    next(perms)
    yield from perms

@lru_cache(maxsize=None)
def not_trivial_indexes_permutations(length):
    return np.array([list(p) for p in not_trivial_permutations(range(length))])


path_index = np.argsort(path[:-1])
for _ in range(1):
    for ids in tqdm(candidates):
        # Index for each city in the order they appear in tour
        idx = sorted(path_index[ids])
        head, tail = path[idx[0] - 1], path[idx[-1] + 1]
        # Split the path between the candidate cities
        chunks = [path[idx[0]:idx[0]+1]]
        for i in range(len(idx) - 1):
            chunks.append(path[idx[i]+1:idx[i+1]])
            chunks.append(path[idx[i+1]:idx[i+1]+1])
        # Remove empty chunks and calculate score for each remaining chunk
        chunks = [chunk for chunk in chunks if len(chunk)]
        scores = [chunk_scores(chunk) for chunk in chunks]
        # Distance (score) for all chunks in the current order
        default_score = score_compound_chunk(idx[0]-1, head, chunks, tail, scores)
        best_score = default_score
        for indexes_permutation in not_trivial_indexes_permutations(len(chunks)):
            # Get score for all chunks when permutating the order
            score = score_compound_chunk(idx[0]-1, head, chunks, tail, scores, indexes_permutation)
            if score < best_score:
                permutation = [chunks[i] for i in indexes_permutation]
                best_chunk = np.concatenate([[head], np.concatenate(permutation), [tail]])
                best_score = score
        if best_score < default_score:
            # Update tour if an improvement has been found
            path[idx[0]-1:idx[-1]+2] = best_chunk
            path_index = np.argsort(path[:-1])
            improvement = True


# I'm using only one iteration since the kernel's running time is short.

# In[ ]:


final_score = score_path(path)
print("Final score is {:.2f}, improvement: {:.2f}".format(final_score, initial_score - final_score))


# Save the result path.

# In[ ]:


def make_submission(name, path):
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)
    
make_submission(score_path(path), path)

