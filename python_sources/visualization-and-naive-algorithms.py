#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path


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


# Let's plot the cities! The single red dot is the north pole, the start and the end of our trip.

# In[ ]:


cities.plot.scatter(x='X', y='Y', s=0.07, figsize=(15, 10))
north_pole = cities[cities.CityId==0]
plt.scatter(north_pole.X, north_pole.Y, c='red', s=15)
plt.axis('off')
plt.show()


# What a surprise! The problem could rather be seen as drawing this reindeer picture in a single trait by minizing its length.

# ## Distance function

# Now let's look at the particular distance we want to minimize. It's just the common euclidean distance, but every 10-th step, the path to the next city will be 10% longer, unless out current city's id is a prime number...

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


# Let's test this function on a unoptimized path, for example [0, 1, 2, ..., 0]

# In[ ]:


unopt_path = list(range(cities.shape[0])) + [0]

get_ipython().run_line_magic('time', 'print("Distance of the naive path:" , total_distance(unopt_path))')


# We define a function that plot the path between cities, and an other that count the number of primes reached on 10th iterations, in order to visualize our later results :

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


# ## Naive algorithm

# A simple and greedy approach to minimize the total distance is to look for the nearest city at each iteration. We can either minimize among all the remaining cities but also among a random batch of cities to speed up the computations.

# In[ ]:


# Greedy algorithm without prime numbers
def greedy_whp(verbose=True, k_iter=10000):
    ID = cities.CityId.values
    coord = cities[['X', 'Y']].values
    pos = coord[0]
    path = [0]
    
    ID = np.delete(ID, 0)
    coord = np.delete(coord, 0, axis=0)
    
    it = 0
    
    while len(path) != cities.shape[0]:
        # Compute the distance matrix
        dist_matrix = np.linalg.norm(coord - pos, axis=1)
        
        # Find the nearest city
        i_min = dist_matrix.argmin()
        
        path.append(ID[i_min])
        pos = coord[i_min]
        
        # Delete it
        coord = np.delete(coord, i_min, axis=0)
        ID = np.delete(ID, i_min)
        
        it += 1
        
        if verbose and it%k_iter == 0:
            print('{} iterations, {} remaining cities.'.format(it, len(ID)))
    
    # Don't forget to add the north pole at the end!
    path.append(0)
    
    return path


# I tried to optimize the function with numpy operations, and it takes 6s/1000 iterations on a Intel(R) Xeon(R) CPU E5-1650 v4, 3.60GHz. This time decreases with the iterations as visited cities are removed.

# In[ ]:


get_ipython().run_line_magic('time', 'greedy_path_whp = greedy_whp(verbose=True)')


# In[ ]:


# Compute the total distance
print("Total distance:", total_distance(greedy_path_whp))

# Count the number of prime numbers reached
print("Number of prime numbers reached: {}/{}".format(count_primes_path(greedy_path_whp), len(PRIMES)))

# Visualize it
plot_path(greedy_path_whp)


# It looks more presentable than the previous one, but there is still some big jumps between cities...
# In addition, we reached ~10% of the available prime numbers, but this is a coincidence!

# ### with batch

# Let's try to speed up our naive algorithm by choosing next cities among a random batch each turn.

# In[ ]:


def greedy_whp_with_batch(batch_size=10000, verbose=True, k_iter=10000):
    ID = cities.CityId.values
    coord = cities[['X', 'Y']].values
    pos = coord[0]
    path = [0]
    
    ID = np.delete(ID, 0)
    coord = np.delete(coord, 0, axis=0)
    it = 0
    
    while len(path) != cities.shape[0]:
        # Choose randomly a batch of cities
        n_remaining = ID.shape[0]
        batch = np.random.choice(np.arange(n_remaining), size=batch_size)

        select_ID = ID[batch]
        select_coord = coord[batch, :]
        
        # Compute the distance matrix
        dist_matrix = np.linalg.norm(select_coord - pos, axis=1)
        
        # Find the nearest city
        i_min = dist_matrix.argmin()

        path.append(select_ID[i_min])
        pos = select_coord[i_min]
        
        # Delete it
        coord = np.delete(coord, batch[i_min], axis=0)
        ID = np.delete(ID, batch[i_min])
        it += 1
        
        if verbose and it%k_iter == 0:
            print('{} iterations, {} remaining cities.'.format(it, len(ID)))
    
    # Don't forget to add the north pole at the end!
    path.append(0)
    
    return path


# It goes faster with the batch, but not as fast as expected (because of the additional numpy operations to choose a batch). 

# In[ ]:


get_ipython().run_line_magic('time', 'batch_path = greedy_whp_with_batch(batch_size=1000)')


# In[ ]:


# Compute the total distance
print("Total distance:", total_distance(batch_path))

# Count the number of prime numbers reached
print("Number of prime numbers reached: {}/{}".format(count_primes_path(batch_path), len(PRIMES)))

# Visualize the solution
plot_path(batch_path)


# The batch adds randomness in the plot, and give us a greater distance than the previous one, but it runs faster. However, the distance is too high because at each iteration we miss a lot of near cities that we haven't picked in the batch. Anyway, this is not a viable solution!

# ### With prime numbers

# In[ ]:


# Greedy algorithm with prime numbers

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


get_ipython().run_line_magic('time', 'greedy_path = greedy()')


# In[ ]:


# Compute the total distance
print("Total distance:", total_distance(greedy_path))

# Count the number of prime numbers reached
print("Number of prime numbers reached: {}/{}".format(count_primes_path(greedy_path), len(PRIMES)))

# Visualize the solution
plot_path(greedy_path)


# Let's submit this first solution!

# In[ ]:


s = pd.read_csv('../input/sample_submission.csv')
s['Path'] = np.array(greedy_path)

s.to_csv('simple_nearest.csv', index=False)

