#!/usr/bin/env python
# coding: utf-8

# This kernel implements (or at least attempts to implement) one of the insertion methods for TSP tour construction. These methods start with a simple tour joining two nodes. The remaining nodes are inserted into the path according to some selection process. The process shown here is known as Farthest Insertion, where at each selection step, the node found to be the most distant from any node on the current path is selected to be inserted next. While this sounds counterintuitive, it has been empirically found to perform better than other insertion methods as well as simple nearest neighbor construction algorithms. 
# 
# The algorithm proceeds as follows:
# 
# 1.  Start with an initial high cost edge between two nodes.
# 
# 2. From the remaining nodes available, find the node most distant from the two nodes found in Step 1. Add this node to your path, making a complete 3-node tour.
# 
# 3. With the remaining nodes, continue this selection process. Find the node most distant from all of your already-selected nodes. Insert this node (*not append*) between the nearest edge {i, j} on the tour, essentially breaking it up. When there are no more nodes left to select, you're done.
# 
# A visual example of this algorithm can be found at the website linked below.
# 
# [https://users.cs.cf.ac.uk/C.L.Mumford/howard/FarthestInsertion.html](https://users.cs.cf.ac.uk/C.L.Mumford/howard/FarthestInsertion.html)
# 
# With that, let's get to coding

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sympy as sym

import time

from scipy.spatial import ConvexHull, distance_matrix
from scipy.spatial.distance import cdist, pdist


# In[ ]:


cities = pd.read_csv("../input/cities.csv")
available_nodes = set(cities.CityId.values)
positions = np.array([cities.X.values, cities.Y.values]).T


# **Step 1** 
# 
# In this kernel, I initialize the tour using the two most distant cities in the dataset. The way I do this is to first find the convex hull of the cities, since the two most distant points are guaranteed to be on the convext hull and the convex hull algorithm is an O(nlogn). From there select the two most-distant using a brute force search. I can get away with this because the convex hull consists of only 20-something points. If it were much larger you could do some kind of rotating calipers algorithm to select the points. I didn't feel like doing that here, plus it's not necessary.

# In[ ]:


hull = ConvexHull(positions)

# Finding the two most distant positions on the hull
mat = distance_matrix(positions[hull.vertices], positions[hull.vertices])
i, j = np.unravel_index(mat.argmax(), mat.shape)

# Initializing our tour, and removing those cities from the set of available nodes
tour =  np.array([hull.vertices[i], hull.vertices[j]])
available_nodes.remove(hull.vertices[i])
available_nodes.remove(hull.vertices[j])


# **Step 2**
# 
# As explained above, we look at the nodes we have left and select the most distant node from the nodes on our 2-node tour. To keep track of the candidate nodes, I'm using a numpy masked array, and just masking out as I go. I'm also keeping track of the closest distances between the remaining nodes and the points on the tour. That way, as will be seen in the next step, we will only need to find the closest distance to the remaining points and the newest point we add to the tour. This will save time.

# In[ ]:


nodes_arr = np.ma.masked_array([i for i in available_nodes])
best_distances = np.ma.masked_array(cdist(positions[nodes_arr], positions[tour], 'euclidean').min(axis=1))

# We want the most distant node, so we get the max
index_to_remove = best_distances.argmax()
next_id = nodes_arr[index_to_remove]

# Add the most distant point, as well as the first point to close the tour, we'll be inserting from here
tour = np.append(tour, [next_id, tour[0]])

available_nodes.remove(next_id)
nodes_arr[index_to_remove] = np.ma.masked
best_distances[index_to_remove] = np.ma.masked


# **Step 3**
# Continue with the selection process until you've exhausted all of the remaining candidate cities. The city k must be inserted between edge {i, j} such that $c_{ik} + c_{jk} - c_{ij}$ is minimized where $c_{ab}$ is the cost (distance) of edge {a, b}. This whole process takes about an hour. If you're reading this and you're an expert at python/numpy I'd be happy to take suggestions on ways to improve performance if there are any.
# 
# First some convenience methods we will be using

# In[ ]:


# Takes two arrays of points and returns the array of distances
def dist_arr(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum(axis=1))

# This is our selection method we will be using, it will give us the index in the masked array of the selected node,
# the city id of the selected node, and the updated distance array.
def get_next_insertion_node(nodes, positions, prev_id, best_distances):
    best_distances = np.minimum(cdist(positions[nodes], positions[prev_id].reshape(-1, 2), 'euclidean').min(axis=1), best_distances)
    max_index = best_distances.argmax()
    return max_index, nodes[max_index], best_distances


# In[ ]:


start_time = time.time()
progress = 3
while len(available_nodes) > 0:
    index_to_remove, next_id, best_distances = get_next_insertion_node(nodes_arr, positions, next_id, best_distances)
    progress += 1
    
    # Finding the insertion point
    c_ik = cdist(positions[tour[:-1]], positions[next_id].reshape(-1, 2))
    c_jk = cdist(positions[tour[1:]], positions[next_id].reshape(-1, 2))
    c_ij = dist_arr(positions[tour[:-1]],positions[tour[1:]]).reshape(-1, 1)
    i = (c_ik + c_jk - c_ij).argmin()
    
    tour = np.insert(tour, i+1, next_id)

    available_nodes.remove(next_id)
    nodes_arr[index_to_remove] = np.ma.masked
    best_distances[index_to_remove] = np.ma.masked
    
    if progress % 1000 == 0:
        print(f'Progress: {progress}, Remaining: {len(available_nodes)}')
elapsed_time = time.time() - start_time
time.strftime("%H:%M:%S", time.gmtime(elapsed_time))


# Ok so we have a tour, let's look at it

# In[ ]:


print(tour)


# Whoops, doesn't start and end at 0, so it's not a valid submission. Let's fix that

# In[ ]:


tour = np.delete(tour, -1)
tour = np.roll(tour, -tour.argmin())
tour = np.append(tour, 0)
print(tour)


# That's better, now let's have a look at the tour

# In[ ]:


plt.figure(figsize=(20,20))
plt.title("Farthest Insertion Method")
plt.plot(*zip(*positions[tour]), '-r')
plt.scatter(*zip(*positions), c="b", s=10, marker="s")
plt.show()


# That looks pretty good. You don't see any giant edges that you sometimes get from a greedy algorithm. Let's see the score.

# In[ ]:


def dist_1d(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

def score_tour(tour, positions):
    primes = set(sym.sieve.primerange(1, tour.max()+1))
    score = 0
    for i, (j, k) in enumerate(zip(tour[:-1], tour[1:])):
        score += dist_1d(positions[j], positions[k]) * (1.1 if (i+1) % 10 == 0 and j not in primes else 1)
    return score

score_tour(tour, positions)


# Not bad. Not as good as the state of the art solvers, of course, but a definite improvement on the other tour construction algorithms. Now time to save and submit.

# In[ ]:


pd.DataFrame(data={'Path':tour}).to_csv('submission.csv', index=False)

