#!/usr/bin/env python
# coding: utf-8

# Concorde's EUC_2D norm rounds the distances between cities to the nearest integer ([source](https://github.com/matthelb/concorde/blob/master/UTIL/edgelen.c#L299)) whereas competition metric doesn't. This significantly hurts quality as you get closer to TSP optimum. Simply scaling the coordinates up by a few orders of magnitude lets you get quite a bit better solution.

# * This kernel hands off the cities to the very fast Concorde TSP solver
# * Ignores the prime twist on this problem
# * You must have https://github.com/jvkersch/pyconcorde installed in Kernels to run this
# 

# In[ ]:


from concorde.tsp import TSPSolver
from matplotlib import collections as mc
import numpy as np
import pandas as pd
import time
import pylab as pl


# In[ ]:


cities = pd.read_csv('../input/cities.csv')


# In[ ]:


# Instantiate solver
solver = TSPSolver.from_data(
    cities.X * 1000,
    cities.Y * 1000,
    norm="EUC_2D"
)

t = time.time()
tour_data = solver.solve(time_bound = 60.0, verbose = True, random_seed = 42)
print(time.time() - t)
print(tour_data.found_tour)


# In[ ]:


pd.DataFrame({'Path': np.append(tour_data.tour,[0])}).to_csv('submission.csv', index=False)


# In[ ]:


# Plot tour
lines = [[(cities.X[tour_data.tour[i]],cities.Y[tour_data.tour[i]]),(cities.X[tour_data.tour[i+1]],cities.Y[tour_data.tour[i+1]])] for i in range(0,len(cities)-1)]
lc = mc.LineCollection(lines, linewidths=2)
fig, ax = pl.subplots(figsize=(20,20))
ax.set_aspect('equal')
ax.add_collection(lc)
ax.autoscale()

