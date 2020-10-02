#!/usr/bin/env python
# coding: utf-8

# # 0. Setup + Exploration
# 
# Using `sympy` for primality test and prime finding functions. Installed package from GitHub repo **jvkersch/pyconcorde** for the Concorde TSP Solver. `numpy` and `pandas` for basic numerical and data processing needs.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
from sympy import isprime, primerange
import matplotlib.pyplot as plt
from matplotlib import collections as mc
from concorde.tsp import TSPSolver
import time


# Setting up dataframes with an additional `isPrime` tag

# In[ ]:


cities = pd.read_csv('../input/cities.csv')
cities['isPrime'] = cities.CityId.apply(isprime)
prime_cities = cities.loc[cities.isPrime]


# ## Visualizing the World
# 
# The world goes from (0,0) to about (5100, 3400) and consists of 197769 cities (with 17802 prime cities). 9% of cities are prime.
# There appears to be no correlation between `CityId` and the `X, Y` coordinates of a city.

# In[ ]:


plt.figure(figsize=(16,10))
plt.subplot(111, adjustable='box', aspect=1.0)
plt.plot(cities.X, cities.Y, 'k,', alpha=0.3)
plt.plot(cities.X[0], cities.Y[0], 'bx')
plt.xlim(0, 5100)
plt.ylim(0, 3400)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (North Pole = Blue X)', fontsize=18)
plt.show()


# The prime cities seem to be randomly spread around as well.

# In[ ]:


plt.figure(figsize=(16,10))
plt.subplot(111, adjustable='box', aspect=1.0)
plt.plot(cities.X, cities.Y, 'k,', alpha=0.3)
plt.plot(prime_cities.X, prime_cities.Y, 'r.', markersize=4, alpha=0.3)
plt.plot(cities.X[0], cities.Y[0], 'bx')
plt.xlim(0, 5100)
plt.ylim(0, 3400)
plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (Primes = Red Dots, North Pole = Blue X)', fontsize=18)
plt.show()


# ## Helper Functions

# In[ ]:


# This function will submit a path to name.csv (with some validity tests)
def make_submission(name, path):
    assert path[0] == path[-1] == 0
    assert len(set(path)) == len(path) - 1 == 197769
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)

# Fast score calculator given a path
def score_path(path):
    cities = pd.read_csv('../input/cities.csv', index_col=['CityId'])
    pnums = [i for i in primerange(0, 197770)]
    path_df = cities.reindex(path).reset_index()
    
    path_df['step'] = np.sqrt((path_df.X - path_df.X.shift())**2 + 
                              (path_df.Y - path_df.Y.shift())**2)
    path_df['step_adj'] = np.where((path_df.index) % 10 != 0,
                                   path_df.step,
                                   path_df.step + 
                                   path_df.step*0.1*(~path_df.CityId.shift().isin(pnums)))
    return path_df.step_adj.sum()


# # 1. Nearest Neighbour
# 
# Starting from the North Pole, travel to the nearest city (without concern for prime-ness of a city).

# In[ ]:


def nearest_neighbour():
    cities = pd.read_csv("../input/cities.csv")
    ids = cities.CityId.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = ((xy - np.array([last_x, last_y]))**2).sum(-1)
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
    path.append(0)
    make_submission('nearest_neighbour', path)
    return path

#path_nn = nearest_neighbour()


# This approach achieves a score of **1812602.18**.

# # 2. Concorde TSP Solver

# In[ ]:


def concorde_tsp(seed=42):
    cities = pd.read_csv('../input/cities.csv')
    solver = TSPSolver.from_data(cities.X, cities.Y, norm="EUC_2D")
    tour_data = solver.solve(time_bound=60.0, verbose=True, random_seed=seed)
    if tour_data.found_tour:
        path = np.append(tour_data.tour,[0])
        make_submission('concorde', path)
        return path
    else:
        return None

path_cc = concorde_tsp()


# The Concorde solver achieves a score of **1533176.80** (only using the initial steps of the Concorde solver, because the subsequent steps take too long).
# 
# ***Note:*** The Concorde solver doesn't like halting once it goes past the initial solving stage, regardless of the time bound set. However, the difference between letting it run and halting it in 60 seconds is a matter of a few percent in efficiency.

# # 3. Concorde Solver for only Prime Cities
# 
# Perhaps, we can get a more efficient TSP solution if we only use the prime cities (since there are fewer prime cities) and then look into filling in the gaps with non-prime cities.

# In[ ]:


cities = pd.read_csv('../input/cities.csv')
cities['isPrime'] = cities.CityId.apply(isprime)
prime_cities = cities.loc[(cities.CityId == 0) | (cities.isPrime)]
solver = TSPSolver.from_data(prime_cities.X, prime_cities.Y, norm="EUC_2D")
tour_data = solver.solve(time_bound=5.0, verbose=True, random_seed=42)
prime_path = np.append(tour_data.tour,[0])


# In[ ]:


plt.figure(figsize=(16,10))
ax = plt.subplot(111, adjustable='box', aspect=1.0)
ax.plot(cities.X, cities.Y, 'k,', alpha=0.3)

lines = [[(prime_cities.X.values[prime_path[i]],
           prime_cities.Y.values[prime_path[i]]),
          (prime_cities.X.values[prime_path[i+1]],
           prime_cities.Y.values[prime_path[i+1]])]
         for i in range(0, len(prime_cities))]
lc = mc.LineCollection(lines, linewidths=1, colors='r')
ax.add_collection(lc)

ax.plot(cities.X[0], cities.Y[0], 'bx')
plt.xlim(0, 5100)
plt.ylim(0, 3400)

plt.xlabel('X', fontsize=16)
plt.ylabel('Y', fontsize=16)
plt.title('All cities (Prime Path = Red, North Pole = Blue X)', fontsize=18)
plt.show()

