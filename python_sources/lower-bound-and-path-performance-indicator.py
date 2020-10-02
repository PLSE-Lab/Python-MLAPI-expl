#!/usr/bin/env python
# coding: utf-8

# ## Lower Bound and Path Performance Indicator

# In this kernel I will try to analyze path performance by establishing a lower bound and comparing paths against it. Ideally, this could be used to identify regions of the path with low performance and focus on them for optimization.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sympy import isprime, primerange
from math import sqrt
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from matplotlib.path import Path
# For busy visualizations
plt.rcParams['agg.path.chunksize'] = 10000

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# First step is to identify the 2 closest neighbours for each city.
# To do so, I am honestly copying the methods from Kostya Atarik (thanks for sharing your work).

# In[ ]:


# Loading cities and defining primes
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv', index_col=['CityId'])
XY = np.stack((cities.X.astype(np.float32), cities.Y.astype(np.float32)), axis=1)
is_not_prime = np.array([0 if isprime(i) else 1 for i in cities.index], dtype=np.int32)
is_prime = np.array([1 if isprime(i) else 0 for i in cities.index], dtype=np.int32)


# In[ ]:


# Using a KD Tree to identify nearest neighbours:
kdt = KDTree(XY)


# In[ ]:


# Find 3 nearest neighbors (including city itself)
dists, neibs = kdt.query(XY, 3)


# In[ ]:


# List of neighbours
neibs


# In[ ]:


# List of distances
dists


# I am calculating the Lower Bound for the total path as the sum for each city of the shortest way in and shortest way out divided by 2. It assumes only prime cities are visited on 10th step so penalties are not considered.

# In[ ]:


## Lower bound per city
arr_LB = 0.5 * (dists[:, 1] + dists[:, 2])


# In[ ]:


## Lower path distance
path_LB_score = np.sum(arr_LB)
print('Theoretical Lower Bound path would score {}.'.format(path_LB_score))


# ## Path evaluation

# Based on this lower bound, path performance can be reviewed in terms of efficiency.

# Path tested: DP Shuffle by blacksix

# In[ ]:


# Loading a path from public kernels as an example
path = np.array(pd.read_csv('../input/dp-shuffle-by-blacksix/DP_Shuffle.csv').Path)


# In[ ]:


# Because I am not very efficient and piecing up parts of different kernels together I will reload primes
# Load the prime numbers we need in a set with the Sieve of Eratosthenes
def eratosthenes(n):
    P = [True for i in range(n+1)]
    P[0], P[1] = False, False
    p = 2
    l = np.sqrt(n)
    while p < l:
        if P[p]:
            for i in range(2*p, n+1, p):
                P[i] = False
        p += 1
    return P

def load_primes(n):
    return set(np.argwhere(eratosthenes(n)).flatten())

PRIMES = load_primes(cities.shape[0])


# In[ ]:


# Running the list of distances in & out for each city in the path (as well as overall score to double check)
coord = cities[['X', 'Y']].values
score = 0
arr_perfo = np.copy(arr_LB)
for i in range(1, len(path)):
    begin = path[i-1]
    end = path[i]
    distance = np.linalg.norm(coord[end] - coord[begin])
    if i%10 == 0:
        if begin not in PRIMES:
            distance *= 1.1
    score += distance
    arr_perfo[begin] -= distance/2
    arr_perfo[end] -= distance/2
print('Path score: {}.'.format(score))


# In[ ]:


# This gives a list of "inefficiencies" per city
arr_perfo


# Probably because of rounding issues, some cities have positive scores meaning they are beating the Lower Bound, which should not be possible.

# In[ ]:


# Difference between "Lower Bound" path and current path
np.sum(arr_perfo)


# Data can then be used in a plot to illustrate cities where path is less efficient.
# Plot will represent absolute differences as a small improvement on a long leg is much more important than a great improvement on a short leg.
# I'm using the square "efficiency" to accentuate the differences.

# In[ ]:


sq_perfo = arr_perfo * arr_perfo


# In[ ]:


## Scatter Plot
cities.plot.scatter(x='X', y='Y', s=sq_perfo , figsize=(15, 10), c=sq_perfo, cmap='Reds' )
north_pole = cities.iloc[0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.axis('off')
plt.show()


# Quick check on a less efficient path: LKH Solver by Aguiar

# In[ ]:


# Loading a path from public kernels as an example
path = np.array(pd.read_csv('../input/lkh-solver-by-aguiar/LKH_Solver.csv').Path)
score = 0
arr_perfo = np.copy(arr_LB)

for i in range(1, len(path)):
    begin = path[i-1]
    end = path[i]
    distance = np.linalg.norm(coord[end] - coord[begin])
    if i%10 == 0:
        if begin not in PRIMES:
            distance *= 1.1
    score += distance
    arr_perfo[begin] -= distance/2
    arr_perfo[end] -= distance/2
print('Path score: {}.'.format(score))

sq_perfo = arr_perfo * arr_perfo

## Scatter Plot
cities.plot.scatter(x='X', y='Y', s=sq_perfo , figsize=(15, 10), c=sq_perfo, cmap='Reds' )
north_pole = cities.iloc[0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.axis('off')
plt.show()


# Interesting to see how similar are those red circles, isn't it ?
# Next step would be to focus on reducing their diameter / color...

# In[ ]:





# In[ ]:





# In[ ]:




