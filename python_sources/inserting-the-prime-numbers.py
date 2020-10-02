#!/usr/bin/env python
# coding: utf-8

# Hello kagglers! In this kernel, I try to take into account the prime penalty by **inserting prime IDs cities to 10th steps of the path optimized between other cities**. I begin with reminding you different methods to optimize a normal TSP and then (section 3), I try to insert primes! Happy reading! :)

# *You need to install the package from GitHub repo jvkersch/pyconcorde for the Concorde TSP Solver.*

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from concorde.tsp import TSPSolver


# ## Load Data

# In[ ]:


cities = pd.read_csv('../input/cities.csv')


# In[ ]:


# Uncomment if you want to work with less cities
# n_cities = 5000
# cities = cities.sample(n=n_cities, replace=False)
# cities.CityId = np.arange(n_cities)
# cities = cities.reset_index().drop(columns='index')


# In[ ]:


cities.head()


# In[ ]:


cities.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))
north_pole = cities[cities.CityId==0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=30)
plt.axis('off')
plt.show()


# ## Distance and other functions

# In[ ]:


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


# Total distance taking into account the prime numbers
def total_distance(path):
    coord = cities[['X', 'Y']].values
    score = 0
    for i in range(1, len(path)):
        begin = path[i-1]
        end = path[i]
        distance = np.linalg.norm(coord[end] - coord[begin])
        if i%10 == 0:
            if begin not in PRIMES:
                distance *= 1.1
        score += distance
    return score


# In[ ]:


# Path plotting
def plot_path(path):
    coords = cities[['X', 'Y']].values
    ordered_coords = coords[np.array(path)]
    codes = [Path.MOVETO] * len(ordered_coords)
    path = Path(ordered_coords, codes)
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111)
    xs, ys = zip(*ordered_coords)
    ax.plot(xs, ys,  lw=1., ms=10)
    plt.axis('off')
    
    north_pole = cities[cities.CityId==0]
    plt.scatter(north_pole.X, north_pole.Y, c='red', s=10)
    
    plt.show()
    
# Count the number of prime numbers reached
def count_primes_path(path):
    mask_iter = np.array([True if (i+1)%10==0 else False for i in range(len(path))])
    mask_primes = np.isin(path, list(PRIMES))
    return np.sum(mask_iter & mask_primes)


# ## Algorithms

# ### 1. Greedy

# In[ ]:


def greedy(verbose=True, k_iter=10000):
    ID = cities.CityId.values
    coord = cities[['X', 'Y']].values
    id = 0
    pos = coord[0]
    path = [0]
    
    ID = np.delete(ID, 0)    
    coord = np.delete(coord, 0, axis=0)
    
    # Prime penalisation matrix
    primes = -(0.1) * np.isin(ID, PRIMES) + 1.1
    
    it = 1
    
    while len(path) != cities.shape[0]:
        # Compute the distance matrix
        dist_matrix = np.linalg.norm(coord - pos, axis=1)
        
        # Add a penalisation for non-prime numbers
        if (it+1) % 10 == 0 and id not in PRIMES:
            dist_matrix *= primes
        
        # Find the nearest city
        i_min = dist_matrix.argmin()
        
        id = ID[i_min]
        path.append(id)
        pos = coord[i_min]
        
        # Delete it
        coord = np.delete(coord, i_min, axis=0)
        ID = np.delete(ID, i_min)
        primes = np.delete(primes, i_min)
        
        it += 1
        
        if verbose and it%k_iter == 0:
            print('{} iterations, {} remaining cities.'.format(it, len(ID)))
    
    # Don't forget to add the north pole at the end!
    path.append(0)
    
    return path


# In[ ]:


#greedy_path = greedy(verbose=False, k_iter=10000)


# This approach gives a score of **1812602**.

# ### 2. Concorde solver

# I took the following function from [this](https://www.kaggle.com/thexyzt/xyzt-s-visualizations-and-various-tsp-solvers) kernel.

# In[ ]:


def concorde_tsp(c, seed=42):
    solver = TSPSolver.from_data(c.X, c.Y, norm="EUC_2D")
    tour_data = solver.solve(time_bound=60.0, verbose=True, random_seed=seed)
    if tour_data.found_tour:
        path = np.append(tour_data.tour,[0])
        return path
    else:
        return None

#concorde_path = concorde_tsp()


# This approach gives us a score of **1533177**.

# ## Approach by inserting the prime numbers 

# An idea that has been mentionned among several kernels is to resolve the TSP only on prime numbers and then try to interpole 9 cities between those prime numbers ID cities. I will try to do the opposite here : **solve the TSP problem only on non-prime numbers** and **insert the remaining prime numbers into 10th iterations of the path**.

# In[ ]:


cities['prime_id'] = cities.CityId.apply(lambda p: p in PRIMES)
# prime cities
cities_p = cities[cities.prime_id]
# and the others
cities_np = cities[~cities.prime_id]


# We resolve on non-prime cities with the concorde solver as it gives better results than the greedy approach.

# In[ ]:


path_np = concorde_tsp(cities_np)


# In[ ]:


# Here is the incomplete path containing only non-prime numbers
path_incomplete = cities_np.CityId.values[path_np]
plot_path(path_incomplete)


# Then, we have to find a method to insert the remaining prime cities into the incomplete path. This problem is harder than it looks, because insert a city will shift all the next ones in the path. So once we insert a prime city into a 10th index, we can no longer insert cities before it! 
# Here is a naive algorithm that inserts the nearest prime city in the 10th emplacement of the incomplete path until there are'nt prime cities left.

# In[ ]:


def naive_insert(path, prime_cities, non_prime_cities, verbose=False):
    comp_path = list(path.copy())
    k = 1
    remaining_prime_cities_id = prime_cities.CityId.values
    remaining_prime_cities_coord = prime_cities[['X', 'Y']].values
    
    while len(remaining_prime_cities_id) > 0:
        id1, id2 = comp_path[10 * k - 2], comp_path[10 * k - 1] # Cities ID between which we will insert the prime city
        p1, p2 = non_prime_cities[non_prime_cities.CityId==id1][['X', 'Y']].values, non_prime_cities[non_prime_cities.CityId==id2][['X', 'Y']].values
        d12 = np.linalg.norm(p1 - p2)
        
        # Compute all the distances between the city id1 and the remaining prime cities
        dist_c1_matrix = np.linalg.norm(p1 - remaining_prime_cities_coord, axis=1)
        dist_c2_matrix = np.linalg.norm(p2 - remaining_prime_cities_coord, axis=1)
        
        # Find the prime city that minimize the detour
        dist_matrix = dist_c1_matrix + dist_c2_matrix - d12
        i_opt = np.argmin(dist_matrix)
        id_opt = remaining_prime_cities_id[i_opt]
        comp_path.insert(10 * k - 1, id_opt)
        
        # Delete the prime city inserted
        remaining_prime_cities_id = np.delete(remaining_prime_cities_id, i_opt)
        remaining_prime_cities_coord = np.delete(remaining_prime_cities_coord, i_opt, axis=0)
        
        # Go to the next 10th location
        k += 1
        
        # verbose
        if verbose and k%1000 == 0:
            print('{} remaining cities to insert'.format(len(remaining_prime_cities_id)))
    
    return comp_path


# Before we run this algorithm, let's check out if there are enough 10th steps to store all the prime cities !

# In[ ]:


print('Number of prime cities:', len(cities_p))
print('Number of 10th steps:', len(cities) // 10)


# Ok so we are good!

# In[ ]:


get_ipython().run_line_magic('time', 'path = naive_insert(path_incomplete, cities_p, cities_np)')


# Ok now let's see if it has worked!

# In[ ]:


print('Total distance of the completed path:', total_distance(path))


# OOF ! Something went wrong... Let's plot the path :

# In[ ]:


plot_path(path)


# Well, in fact this is understandable : Let's take a closer look  the first steps of the path by highlighting the prime numbers

# In[ ]:


N = 200

coords = cities[['X', 'Y']].values
ordered_coords_complete = coords[np.array(path)][:N]
ordered_coords_incomplete = coords[np.array(path_incomplete)][:N-N//10]

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
xp, yp = zip(*ordered_coords_complete)
xnp, ynp = zip(*ordered_coords_incomplete)
ax.plot(xp, yp,  lw=1., ms=10, label='with primes')
ax.plot(xnp, ynp,  lw=1., alpha=0.8, ls='--', ms=10, label='without primes')

plt.axis('off')

mask_10 = (np.arange(len(ordered_coords_complete))%10==9) | (np.arange(len(ordered_coords_complete))%10==0).astype(int)

north_pole = cities[cities.CityId==0]
plt.scatter(north_pole.X, north_pole.Y, s=30, c='red', label='north pole')

plt.legend()
plt.show()


# At the beginning of the insertions, we see (in blue) little jumps to go to the nearest prime ID, but nothing alarming. But let's take a look at the middle insertions :

# In[ ]:


N = 200

coords = cities[['X', 'Y']].values
ordered_coords_complete = coords[np.array(path)][100000:100000+N]
ordered_coords_incomplete = coords[np.array(path_incomplete)][100000-100000//10:100000+N-(100000+N)//10]

fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(111)
xp, yp = zip(*ordered_coords_complete)
xnp, ynp = zip(*ordered_coords_incomplete)
ax.plot(xp, yp,  lw=1., ms=10, label='with primes')
ax.plot(xnp, ynp,  lw=1., alpha=0.8, ls='--', ms=10, label='without primes')

plt.axis('off')

mask_10 = (np.arange(len(ordered_coords_complete))%10==9) | (np.arange(len(ordered_coords_complete))%10==0).astype(int)

plt.legend()
plt.show()


# Here is the problem! When most of the prime IDs are used, the nearest cities with prime IDs become too distant from the path we are completing.

# A possible solution to this problem is to insert a prime ID city only if it is near enough (by defining a distance threshold), and add the remaining ones to the end. Anyway, I'm not sure it will offer a better result than the concorde solver, given that the prime penalization is too insignificant...

# Well, at least I tried! Thank you for reading!
