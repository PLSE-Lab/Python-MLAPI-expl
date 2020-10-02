#!/usr/bin/env python
# coding: utf-8

# # Nearest Neighbours (greedy) approach

# I made a kernel explaining the problem and some [visualizations](https://www.kaggle.com/wizmik12/visualization).
# 
# The first and most naive approach to the TSP problem is the nearest neighbour method. This heuristic is a greedy approach since in every step looks for the best option. Usually, this do not give us a optimal solution. Although in every step we are doing our best globablly do not need to be the best option. This kind of algorithm are called greedy, they are easy to implement and very fast. So let's look what happens. In TSP problem this method looks for the next nearest city that we have not been yet. 
# 
# This method is explained in the kernel [XYZT's Visualizations and Various TSP Solvers](https://www.kaggle.com/thexyzt/xyzt-s-visualizations-and-various-tsp-solvers) but it ignores the prime constraint. With the prime constraint our metric space is a little weird strange and different so we have to take it into account but... taking it into account improve the results? as nearest neighbours do not return the optimal solution it is possible that the solution could be worse.

# In[ ]:


import numpy as np
import pandas as pd
import time
from sympy import isprime 

df_cities = pd.read_csv('../input/cities.csv')
df_cities.loc[:,'prime'] = df_cities.loc[:,'CityId'].apply(isprime)

# calculate the value of the objective function (total distance)
def pair_distance(x,y):
    x1 = (df_cities.X[x] - df_cities.X[y]) ** 2
    x2 = (df_cities.Y[x] - df_cities.Y[y]) ** 2
    return np.sqrt(x1 + x2)

def total_distance(path):
    distance = [pair_distance(path[x], path[x+1]) + 0.1 * pair_distance(path[x], path[x+1])
                if (x+1)%10 == 0 and df_cities.prime[path[x]] == False else pair_distance(path[x], path[x+1]) for x in range(len(path)-1)]
    return np.sum(distance)

def make_submission(name, path):
    assert path[0] == path[-1] == 0
    assert len(set(path)) == len(path) - 1 == 197769
    pd.DataFrame({'Path': path}).to_csv(f'{name}.csv', index=False)


# In[ ]:


def nearest_neighbour():
    cities = df_cities.copy()
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
    make_submission('nearest_neighbour.csv', path)
    return path

def nearest_neighbour_prim():
    cities = df_cities.copy()
    ids = cities.CityId.values[1:]
    prim = cities.prime.values[1:]
    xy = np.array([cities.X.values, cities.Y.values]).T[1:]
    path = [0,]
    step = 1
    while len(ids) > 0:
        last_x, last_y = cities.X[path[-1]], cities.Y[path[-1]]
        dist = np.sqrt(((xy - np.array([last_x, last_y]))**2).sum(-1))
        if step % 10 == 0:
            dist = np.array([dist[x] if prim[x] else dist[x] * 1.1 for x in range(len(dist))])
        nearest_index = dist.argmin()
        path.append(ids[nearest_index])
        ids = np.delete(ids, nearest_index, axis=0)
        prim = np.delete(prim, nearest_index, axis=0)
        xy = np.delete(xy, nearest_index, axis=0)
        step +=1 
    path.append(0)
    make_submission('nearest_neighbour_prime.csv', path)
    return path


# In[ ]:


path = nearest_neighbour()
path_prime = nearest_neighbour_prim()


# In[ ]:


distance = total_distance(path)
prime_distance = total_distance(path_prime)
print('The distance of NN is', distance, 'but taking into account the prime cities:', prime_distance)


# We have concluded that we can improve the result of nearest neighbours of TSP **1812602.18** if we have into account the prime constraint **1812550.50**. This together with the visualizations suggests us that prime numbers have a great role in this problem so ignoring them and executing classical TSP solvers will not give us feasible solutions of the Santa Problem.
