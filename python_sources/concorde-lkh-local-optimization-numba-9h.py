#!/usr/bin/env python
# coding: utf-8

# This kernel provides an insight of the first part of my solution. It's a CPU kernel that computes everything without loading external data from any other kernel.  Main idea is to run local optimizations on initial path found by off-the-shelf solvers. It is designed for 9 hours runtime. In this part, we try to improve score as much as possible without any fine tuning. We limit and cache possible permutations. 
# 
# * First, we run [Concorde](http://www.math.uwaterloo.ca/tsp/concorde/index.html) TSP solver followed by [LKH](http://akira.ruc.dk/~keld/research/LKH-3/) solver to get a pure TSP path without taking into account the prime penalty. 
# * Second, we try several local combinations (pseudo K-opt) with penalty included. It is inspired from [this kernel](https://www.kaggle.com/kostyaatarik/not-a-3-and-3-halves-opt) that gives the basics.
# 
#  CPU native flags and Numba are used to speed up computations. Real benefits from Numba are only with **nopython** mode so all local optimization functions are designed with this mode. It makes python code not super readable but it's really faster. That's the way we can improve score by almost 1000 in just around 2 hours.
#  
# Second part (not in this Kernel) of the solution is fine tuning. We try to increase combinatory but on less data such as:
# * Area high density points (with HDBScan clusters) 
# * Area with high penalties
# * ...
# 

# In[ ]:


import time
t0 = time.time()


# In[ ]:


# First install recent version of Numba and HDBScan


# In[ ]:


get_ipython().system('conda install -y --channel numba numba')


# In[ ]:


get_ipython().system('pip install hdbscan')


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import time
from sympy import isprime, primerange
from math import sqrt
from tqdm import tqdm
import numba
import hdbscan
from sklearn.neighbors import KDTree
from itertools import combinations, permutations
from functools import lru_cache
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import collections  as mc
import warnings
warnings.filterwarnings('ignore')
seed = 2019


# In[ ]:


# Part1: Find good score quickly
# Wee need Numba 0.41 or higher.
print(numba.__version__)


# In[ ]:


# Read cities and find primes
cities = pd.read_csv('../input/cities.csv')
cities["X"] = cities["X"].astype(np.float64)
cities["Y"] = cities["Y"].astype(np.float64)
is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)
cities["is_not_prime"] = is_not_prime
cities.head(10)


# In[ ]:


# Fast distance and score thanks to Numba
# Updated from: https://www.kaggle.com/kostyaatarik/not-a-3-and-3-halves-opt
XY = np.stack((cities.X, cities.Y), axis=1)

@numba.jit('f8(i4, i8[:])', nopython=True, parallel=False)
def distance_chunk(offset, chunk):
    pure_distance = 0.0
    for path_index in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[path_index], chunk[path_index+1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
    return pure_distance


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


@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def distance_path(path):
    return distance_chunk(0, path)


# In[ ]:


# Basic visualizations
# Run DBScan to find out parts with similar densities
# https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
# We expect long path within each cluster.
clusterer = hdbscan.HDBSCAN(min_cluster_size=600, min_samples=1)
clusterer.fit(cities[['X', 'Y']].values)
cities['hc'] = clusterer.labels_
# Display primes/non primes and all
fig, ax = plt.subplots(2, 2, figsize=(24, 18))
d = cities[cities["is_not_prime"] == 1].plot(kind="scatter", x="X", y="Y", s=1, c="blue", alpha=0.25, ax=ax[0][0], title="non primes: %d" % (len(cities[cities["is_not_prime"] == 1])))
d = cities[cities["is_not_prime"] == 0].plot(kind="scatter", x="X", y="Y", s=1, c="r", alpha=0.25, ax=ax[0][1], title="primes: %d" % (len(cities[cities["is_not_prime"] == 0])))
d = cities.plot(kind="scatter", x="X", y="Y", s=1, alpha=0.9, ax=ax[1][0], title="All: %d" % (len(cities)))
d = cities.plot(kind="scatter", x="X", y="Y", s=1, c=cities['hc'], cmap=plt.cm.tab20, alpha=0.9, ax=ax[1][1], title="All with cluster by density: %d" % (len(cities)))


# In[ ]:


# Functions to read/write LKH tour and TSP file.
def write_lkh_files(lkh_df, filename_par, filename_tsp, filename_tour, filename_initial_tour=None, real=False, bw=False):
    l = len(lkh_df)
    
    with open(filename_par, "w") as f:
        f.write("PROBLEM_FILE = %s\n" % filename_tsp)
        f.write("MOVE_TYPE = 8\n")
        f.write("PATCHING_C = 3\n")
        f.write("PATCHING_A = 2\n")
        f.write("RUNS = 2\n")
        f.write("OUTPUT_TOUR_FILE = %s\n" % filename_tour)
        f.write("SEED = %d\n" % seed)
        f.write("CANDIDATE_SET_TYPE = POPMUSIC\n")
        f.write("POPMUSIC_SOLUTIONS = 12\n")   
        f.write("POPMUSIC_SAMPLE_SIZE = 14\n")
        f.write("POPMUSIC_TRIALS = 1000\n")
        f.write("POPMUSIC_MAX_NEIGHBORS = 14\n")
        f.write("INITIAL_PERIOD = 2500\n")
        f.write("MAX_TRIALS = 1000\n")
        f.write("INITIAL_TOUR_ALGORITHM = QUICK-BORUVKA\n")
        if filename_initial_tour is not None:
            f.write("INITIAL_TOUR_FILE = %s\n" % filename_initial_tour)
        f.write("TRACE_LEVEL = 1\n")
        if bw == True:
            f.write("BWTSP = %d %d %d\n" % (len(lkh_df[lkh_df["is_not_prime"] == 0]), 9, 9999999999))
    
    # LKH start with node 1
    with open(filename_tsp, "w") as f:
        f.write("NAME : SOLVER\n")
        f.write("COMMENT : %d points\n" % l)
        f.write("TYPE : TSP\n")
        f.write("DIMENSION : %d\n" % l)
        f.write("EDGE_WEIGHT_TYPE : EUC_2D\n")
        f.write("NODE_COORD_TYPE : TWOD_COORDS\n")
        f.write("NODE_COORD_SECTION\n")
        for idx, row in lkh_df.iterrows():
            if real:
                f.write("%d %.12f %.12f\n" % (row["CityId"] + 1, (row["X"]), (row["Y"])))
            else:
                f.write("%d %.9f %.9f\n" % (row["CityId"] + 1, (row["X"])*1000.0, (row["Y"]*1000.0)))
            
# Read LKH with additional mapping
def read_lkh_tour(filename, mapping = None):
    with open(filename, 'r') as f:
        tour = []
        i = 0
        for line in f:
            i = i + 1
            if i > 6:
                tour.append(line.rstrip('\n'))
        tour = [int(t) for t in tour if t != 'EOF']
        tour = tour[:-1]
    tour_map = [mapping[int(t)] for t in tour]
    return tour_map

# Back to original points for basic TSP, needed for BW
def lkh_ids_mapping(lkh_df):
    mapping = {}
    for idx, row in lkh_df.iterrows():
        mapping[int(row["CityId"] + 1)] = idx
    return mapping

def read_cnc_tour(filename):
    tour = open(filename).read().split()[1:]
    tour = list(map(int, tour))
    if tour[-1] == 0: tour.pop()
    return tour

def write_lkh_tour(path, filename_tour, comment="None"):
    with open(filename_tour, "w") as f:
        f.write("NAME : %s\n" % filename_tour)
        f.write("COMMENT : %s\n" % comment)
        f.write("COMMENT :\n")
        f.write("TYPE : TOUR\n")
        f.write("DIMENSION : %d\n" % len(path))
        f.write("TOUR_SECTION\n")
        for p in path:
            f.write("%d\n" % p) 
        f.write("-1\n")
        f.write("EOF\n")


# In[ ]:


# LKH3 supports constrained TSP
# Download and Install LKH3 with gcc and CFLAGS for optimization. 


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', "wget http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.5.tgz\n(tar xvfz LKH-3.0.5.tgz)\n(cd LKH-3.0.5 && make CC=gcc CFLAGS='-IINCLUDE -Ofast -march=native -mtune=native -fPIC')\n(cd LKH-3.0.5 && cp LKH ../)\nrm -rf LKH-3.0.5 LKH-3.0.5.tgz")


# In[ ]:


# Generate TSP and LKH files
write_lkh_files(cities, "cities.lkh.par", "cities.lkh.tsp", "cities.lkh.tour",
                filename_initial_tour="cities.cnc.lkh.tour", real=False, bw=False)


# In[ ]:


# Download and Install concorde for an initial run that will generate: cities.cnc.lkh.tour
# Concorde is fast and provides good initial tour for LKH


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', "if ! [[ -f ./linkern ]]; then\n  wget http://www.math.uwaterloo.ca/tsp/concorde/downloads/codes/src/co031219.tgz\n  echo 'c3650a59c8d57e0a00e81c1288b994a99c5aa03e5d96a314834c2d8f9505c724  co031219.tgz' | sha256sum -c\n  tar xf co031219.tgz\n  (cd concorde && CFLAGS='-Ofast -march=native -mtune=native -fPIC' ./configure)\n  (cd concorde/LINKERN && make -j && cp linkern ../../)\n  rm -rf concorde co031219.tgz\nfi")


# In[ ]:


# Run concorde for 30 minutes


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'head cities.lkh.tsp')


# In[ ]:


get_ipython().run_cell_magic('bash', '', 'time ./linkern -K 1 -s 2019 -S cities.cnc.tour -R 999999999 -t 1800 cities.lkh.tsp >cities.cnc.log')


# In[ ]:


# Results from concorde
cnc_tour = np.array(read_cnc_tour("cities.cnc.tour"))
cnc_score = score_path(cnc_tour)
print(f'Concorde distance is {distance_path(cnc_tour):.2f} and score is {score_path(cnc_tour):.2f} in {1800/60 :.1f} minutes')


# In[ ]:


# Save concorde tour as initial tour for LKH
write_lkh_tour([p+1 for p in cnc_tour], "cities.cnc.lkh.tour")


# In[ ]:


# Run LKH with POPMUSIC initializer for 5h55min


# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'timeout 21300s ./LKH cities.lkh.par')


# In[ ]:


# Results from LKH
lkh_tour = np.array(read_lkh_tour("cities.lkh.tour", mapping=lkh_ids_mapping(cities)))


# In[ ]:


# Go back to North pole
path = np.concatenate([lkh_tour.copy(), np.array([0])])
print(path[0:10])
print(path[-10:])
print(f'LKH distance is {distance_path(path):.2f} and score is {score_path(path):.2f} in {21300/60 :.1f} minutes')


# In[ ]:


t1 = time.time()


# In[ ]:


# Numba optimized functions.
# From: https://www.kaggle.com/kostyaatarik/not-a-3-and-3-halves-opt
@numba.jit('f8(i8, i8, i8)', nopython=True, parallel=False)
def cities_distance(offset, id_from, id_to):
    xy_from, xy_to = XY[id_from], XY[id_to]
    dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
    distance = sqrt(dx * dx + dy * dy)
    if offset % 10 == 9 and is_not_prime[id_from]:
        return 1.1 * distance
    return distance

@numba.jit
def chunk_scores(chunk):
    scores = np.zeros(10)
    pure_distance = 0
    for i in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[i], chunk[i+1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
        if is_not_prime[id_from]:
            scores[9-i%10] += distance
    scores *= 0.1
    scores += pure_distance
    return scores

# sort by distance
@numba.jit('f8(i8[:])', nopython=True, parallel=False)
def sum_distance(ids):
    res = 0
    for i in numba.prange(len(ids)):
        for j in numba.prange(i + 1, len(ids)):
            res += cities_distance(0, ids[i], ids[j])
    return res


# In[ ]:


# Precomputes all permutations. Memory consuming but better for execution.
def not_trivial_permutations(iterable):
    perms = permutations(iterable)
    next(perms)
    yield from perms

def _not_trivial_indexes_permutations(length):
    return np.array([list(p) for p in not_trivial_permutations(range(length))], dtype=np.int64)

MAX_PERMS=11

PERMS_01 = _not_trivial_indexes_permutations(1)
PERMS_02 = _not_trivial_indexes_permutations(2)
PERMS_03 = _not_trivial_indexes_permutations(3)
PERMS_04 = _not_trivial_indexes_permutations(4)
PERMS_05 = _not_trivial_indexes_permutations(5)
PERMS_06 = _not_trivial_indexes_permutations(6)
PERMS_07 = _not_trivial_indexes_permutations(7)
PERMS_08 = _not_trivial_indexes_permutations(8)
PERMS_09 = _not_trivial_indexes_permutations(9)
PERMS_10 = _not_trivial_indexes_permutations(10)
PERMS_11 = _not_trivial_indexes_permutations(11)

@numba.jit(nopython=True, parallel=False)
def not_trivial_indexes_permutations(length):
    if length == 2:
        return PERMS_02
    elif length == 3:
        return PERMS_03
    elif length == 4:
        return PERMS_04  
    elif length == 5:
        return PERMS_05  
    elif length == 6:
        return PERMS_06
    elif length == 7:
        return PERMS_07
    elif length == 8:
        return PERMS_08    
    elif length == 9:
        return PERMS_09
    elif length == 10:
        return PERMS_10 
    elif length == 11:
        return PERMS_11


# In[ ]:


@numba.jit
def chunk_scores2(chunk):
    scores = np.zeros(10)
    pure_distance = 0
    for i in numba.prange(chunk.shape[0] - 1):
        id_from, id_to = chunk[i], chunk[i+1]
        xy_from, xy_to = XY[id_from], XY[id_to]
        dx, dy = xy_from[0] - xy_to[0], xy_from[1] - xy_to[1]
        distance = sqrt(dx * dx + dy * dy)
        pure_distance += distance
        if is_not_prime[id_from]:
            scores[9-i%10] += distance
    scores *= 0.1
    scores += pure_distance
    return scores

@numba.jit(nopython=True, parallel=False)
def range_length(length):
    return [p for p in range(length)]


# In[ ]:


# Run KDTree for nearest neighbours or each city by Nuplets
kdt = KDTree(XY)
# Stage 1 is around 13min, Stage2 is around 15min, Stage3 is around 100min
cs = [2, 3, 4]
cs_nearest = [17, 10, 8]
cs_radius = [12, 10, 6]
tuples = []
for c, n, r in zip(cs, cs_nearest, cs_radius):
    print(f'\nGenerating tuples for N={c} with nearest={n} and radius={r} ...')
    Nuplets = set()
    for city_id in tqdm(cities.index, mininterval=2.0):
        dists, neibs = kdt.query([XY[city_id]], k=n)
        for Nuplet in combinations(neibs[0], c):
            if all(Nuplet):
                Nuplets.add(tuple(sorted(Nuplet)))
        if r > 0:
            neibs = kdt.query_radius([XY[city_id]], r=r, count_only=False, return_distance=False)
            for Nuplet in combinations(neibs[0], c):
                if all(Nuplet):
                    Nuplets.add(tuple(sorted(Nuplet)))
    tuples.append(Nuplets)
    print(f'{len(Nuplets)} cities for N={c} are selected.')

for t,i in zip(tuples, range(len(tuples))):
    t = np.array(list(t))
    distances = np.array(list(map(sum_distance, tqdm(t))))
    order = distances.argsort()
    tuples[i] = t[order]


# In[ ]:


# Main function for local optimizations. It's a bit flat but it helps Numba to speed up.
# It builds path chunks from Nuplets combinations, then tries chunk content permutations until score improves. 
@numba.jit(numba.int64[:](numba.int64[:], numba.int64[:,:], numba.int64, numba.int64), nopython=True, parallel=False)
def optN(path, tmptuples, max_chunk_len, full_reverse_rank):
    path_index = np.argsort(path[:-1])
    for iids in numba.prange(len(tmptuples)):
        ids = tmptuples[iids]        
        s = sorted(path_index[ids])
        head, tail = path[s[0]-1], path[s[-1]+1]
        # Build chunks
        if len(s) == 2:
            chunks = [path[s[0]:s[0]+1], 
                      path[s[0]+1:s[1]], 
                      path[s[1]:s[1]+1]]              
        elif len(s) == 3:
            chunks = [path[s[0]:s[0]+1], 
                      path[s[0]+1:s[1]], 
                      path[s[1]:s[1]+1], 
                      path[s[1]+1:s[2]],
                      path[s[2]:s[2]+1]]             
        elif len(s) == 4:         
            chunks = [path[s[0]:s[0]+1], 
                      path[s[0]+1:s[1]], 
                      path[s[1]:s[1]+1], 
                      path[s[1]+1:s[2]],
                      path[s[2]:s[2]+1],
                      path[s[2]+1:s[3]],
                      path[s[3]:s[3]+1]]
        elif len(s) == 5:         
            chunks = [path[s[0]:s[0]+1], 
                      path[s[0]+1:s[1]], 
                      path[s[1]:s[1]+1], 
                      path[s[1]+1:s[2]],
                      path[s[2]:s[2]+1],
                      path[s[2]+1:s[3]],
                      path[s[3]:s[3]+1],
                      path[s[3]+1:s[4]],
                      path[s[4]:s[4]+1]]            
        # Drop empty chunks        
        chunks = [chunk for chunk in chunks if len(chunk)]
        
        if len(chunks) > max_chunk_len:
            continue
        
        scores = [chunk_scores2(chunk) for chunk in chunks]
        
        lindexes_permutation = range_length(len(chunks))
        # Inline score
        lscore = 0.0
        offset = s[0]-1
        last_city_id = head
        for iindex in numba.prange(len(lindexes_permutation)):
            index = lindexes_permutation[iindex]
            chunk = chunks[index]
            chunk_scores_ = scores[index]
            lscore += cities_distance(offset % 10, last_city_id, chunk[0])
            lscore += chunk_scores_[(offset + 1) % 10]
            last_city_id = chunk[-1]
            offset += len(chunk)
        default_score =  lscore + cities_distance(offset % 10, last_city_id, tail)
        
        best_score = default_score
        nti = not_trivial_indexes_permutations(len(chunks))
        for iindexes_permutation in numba.prange(len(nti)):
            indexes_permutation = nti[iindexes_permutation]
            # Inline score
            lscore = 0.0
            offset = s[0]-1
            last_city_id = head
            for iindex in numba.prange(len(indexes_permutation)):
                index = indexes_permutation[iindex]
                chunk = chunks[index]
                chunk_scores_ = scores[index]
                lscore += cities_distance(offset % 10, last_city_id, chunk[0])
                lscore += chunk_scores_[(offset + 1) % 10]
                last_city_id = chunk[-1]
                offset += len(chunk)
            score =  lscore + cities_distance(offset % 10, last_city_id, tail)
            if score < best_score:
                permutation = [chunks[i] for i in indexes_permutation]
                # Concatenate permutation
                tl = 0
                for t in range(len(permutation)):
                    tl = tl + len(permutation[t])
                z = np.zeros(tl, dtype=np.int64)
                tl = 0
                for t in range(len(permutation)):
                    z[tl:tl+len(permutation[t])] = permutation[t]
                    tl = tl + len(permutation[t])
                # Concatenate head/tail
                best_chunk = np.zeros(len(z) + 2, dtype=np.int64)
                best_chunk[0] = head
                best_chunk[1:-1] = z[:]
                best_chunk[-1] = tail
                best_score = score
                
        if best_score < default_score:
            path[s[0]-1:s[-1]+2] = best_chunk
            path_index = np.argsort(path[:-1])
            print('New total score is ', score_path(path), ', Permutating path at indexes ', s, 'Progress: ', int(iids*100.0/len(tmptuples)))
            
    return path.copy()


# In[ ]:


# Check if reverse path scores better
def check_reverse(p):
    sr = score_path(p[::-1])
    s = score_path(p)
    if sr < s:
        print("Reverse better than initial: %.1f vs %.1f" % (sr, s))
        return p[::-1]
    else:
        print("Reverse not better than initial: %.1f vs %.1f" % (sr, s))
        return p


# In[ ]:


# Run local optimizations now
initial_path = check_reverse(path.copy())


# In[ ]:


new_path = optN(initial_path, tuples[0], 11, 0)


# In[ ]:


print(f'Total score after stage 1 is {score_path(new_path):.2f} in {(time.time()-t1)/60.0:.1f} min')


# In[ ]:


t2 = time.time()
new_path1 = optN(new_path.copy(), tuples[1], 11, 0)


# In[ ]:


print(f'Total score after stage 2 is {score_path(new_path1):.2f} in {(time.time()-t2)/60.0:.1f} min')


# In[ ]:


t3 = time.time()
new_path2 = optN(new_path1.copy(), tuples[2], 11, 0)


# In[ ]:


print(f'Total score after stage 3 is {score_path(new_path2):.2f} in {(time.time()-t3)/60.0:.1f} min')


# In[ ]:


print(f'Total local optimizations time is {(time.time()-t1)/60.0:.1f} min')


# In[ ]:


print(f'Final score is {score_path(new_path2):.2f}')


# In[ ]:


# Output
pd.DataFrame({'Path': new_path2}).to_csv('submission.csv', index=False)


# In[ ]:


# Path vizualization with clusters and starting tour in red
def plot_tour_extended(cities, ax, b=None, path=None, chunk_path=None):
    ax.scatter(cities.X, cities.Y, c=cities.hc, cmap=plt.cm.tab20, s=1)
    if b is not None:
        cities_b = cities[cities["CityId"].isin(b)]
        ax.scatter(cities_b.X, cities_b.Y, s=16, c="orange")
    if path is not None:
        lines = [[(cities.X[path[i]],cities.Y[path[i]]),(cities.X[path[i+1]],cities.Y[path[i+1]])] for i in range(0,len(path)-1)]
        lc = mc.LineCollection(lines, linewidths=1, colors="gray", alpha=0.7)
        ax.add_collection(lc)
        ax.autoscale()
        # Start point
        ax.scatter(cities.X[path[0]], cities.Y[path[0]], s=28, c="black")
    if chunk_path is not None:
        lines = [[(cities.X[chunk_path[i]],cities.Y[chunk_path[i]]),(cities.X[chunk_path[i+1]],cities.Y[chunk_path[i+1]])] for i in range(0,len(chunk_path)-1)]
        lc = mc.LineCollection(lines, linewidths=1, colors="r")
        ax.add_collection(lc)     
    plt.show()
    
fig, ax = plt.subplots(1, 1, figsize=(20, 14))
plot_tour_extended(cities, ax, b=None, path=new_path2, chunk_path=new_path2[0:500])


# In[ ]:


print(f'Total kernel time is {(time.time()-t0)/60.0:.1f} min')


# In[ ]:


# Part 2: Fine tuning (to be continued)

