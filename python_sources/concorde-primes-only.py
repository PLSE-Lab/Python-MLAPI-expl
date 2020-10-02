#!/usr/bin/env python
# coding: utf-8

# ## Concorde - Primes only
# 
# Towards creating an "All the Carrots" path, where all primes are at 1/10th-cities locations, this notebook gets the Prime cities (and North Pole) and finds a TSP solution for them. The result is written to a csv file with columns CityId and Path.
# 
# This primes-only path was then used as a starting backbone in the kernel
# [All the Carrots: Traveling Santa 2018](https://www.kaggle.com/dan3dewey/all-the-carrots-traveling-santa-2018) 
# (v2) where it was expanded into a dense path by adding 9 (nearby) cities between each prime city. The remaining unused cites were then sampled and 1/10 of them chosen (systematically but randomly) to be added to the primes (and NP) to create a full-length backbone.
# 
# Now, in (v5, v6, v8) of this kernel that list of full length backbone cities, new_backbone_fromt60.csv,  is the new input and a TSP for them is formed and output for subsequent use by "All the Carrots" (v3 etc.) to generate a full all-city path with all prime cities at 1/10 locations (though not all 1/10th locations are primes.)
# 
# In (v9), the "remaining" cities are input and put in TSP order. <br>
# In (v11) the TSP-remaining cities are sampled at 1/10th and those cities are added to the primes-only TSP to make an improved(?!) full backbone. 
# 
# In (v13) the "remaining" cities, from
# [All the Carrots...](https://www.kaggle.com/dan3dewey/all-the-carrots-traveling-santa-2018) , (v7),
# are again input and TSP'ed (with NP), but with the idea of a "two loop" solution: a penalty-free primes-at-10th-cities path and a no-primes path (t_bound=2500 to avoid Kaggle memory issue.)
# 
# ### Forked
# from William Cukierski's ["Concorde solver"](https://www.kaggle.com/wcukierski/concorde-solver); and then under Settings/Packages installed from GitHub: jvkersch/pyconcorde .
# * This kernel hands off the cities to the very fast Concorde TSP solver
# * Ignores the prime twist on this problem
# * You must have https://github.com/jvkersch/pyconcorde installed in Kernels to run this
# 
# 
# 
# ### Notes/Diary: <br>
# 
# * 6--7 Dec 2018: Start on finding an all-carrots solution(v1,2); output a good TSP for NP + prime cities(v3).<br>
# * 8 Dec 2018: Use the "All the Carrots" full backbone list as input to do another TSP solution and output it (v5).  <br>
# * 11 Dec 2018: Redo the TSP solution of full backbone starting with it in 'random' (i.e., CityId) order, in case initial ordering matters. Makes some differnece, though TSP could have randomness in its solution. Run for 500, 5000 s bound starting with the random order (v6, v7).
# * 13 Dec 2018: Redo the TSP for NP + prime cities and run with a 5000 s bound (v8)
# * 15-16 Dec 2018: Get all the "remaining" (19740) cities and do a TSP on them (v9), to select a better set of backbone cities from/for them. Add those to the primes-only to make new full backbone (v11.)
# * 21 Dec 2018: Get all the "remaining" (19740) cities and do a TSP on them + NP (v13).
# * 27 Dec 2018: Get the v8 "All the Carrots" remaining cities and do a TSP on them + NP (v14).

# In[ ]:


from concorde.tsp import TSPSolver
from matplotlib import collections  as mc
import numpy as np
import pandas as pd
import time
import pylab as pl


# In[ ]:


# Read all the Cities
cities = pd.read_csv('../input/traveling-santa-2018-prime-paths/cities.csv')


# In[ ]:


max_city = max(cities.CityId)
print("Max CityId in list : ", max_city)

def sieve_eratosthenes(n):
    prime_flags = [False, False] + [True for i in range(n-1)]
    p = 2
    while (p * p <= n):
        if (prime_flags[p] == True):
            for i in range(p * 2, n + 1, p):
                prime_flags[i] = False
        p += 1
    return prime_flags

prime_flags = np.array(sieve_eratosthenes(max_city)).astype(int)
cities['Prime'] = prime_flags


# In[ ]:


cities.tail()


# ## Select input for TSP ...
# This kernel is used to TSP-order several different (related) inputs...

# In[ ]:


# Just NP + Prime cities:
#
# Select just the prime cities and include leading 0 (North Pole)
##prime_tour_select = (cities['CityId'] == 0) | (cities['Prime'] == 1)
# Down-select to the desired cities
##cities = cities[prime_tour_select]


# In[ ]:


# Full backbone:
#
# The NP + Primes have been augmented by other 'backbone' cities:
# (e.g., the full backbone list as output from "All the Carrots" kernel.)
# Dataframe with CityId column:
##full_bb = pd.read_csv('../input/santa-18-primes-tsp60/new_backbone_fromt60.csv')

# Remaining cities:
#
# Do TSP on the a sample of the "remaining" cities
##full_bb = pd.read_csv('../input/santa-18-primes-tsp60/remaining_cities_v4B_t5000.csv')
#
# Remaining cities from v7B of All the Carrots...
##full_bb = pd.read_csv('../input/santa-18-primes-tsp60/remaining_cities_v7B.csv')
full_bb = pd.read_csv('../input/santa-18-primes-tsp60/remaining_cities_v8C.csv')


# Final processing for either of the inputs in this cell:
#
# Replace the cities df with just the cities that are in the selected list:
full_bb_list = list(full_bb['CityId'])
# Put the bb cities in sorted order, i.e., 'random' in terms of the TSP path
full_bb_list.sort()
# Include the NP, if desired (yes for v7B)
full_bb_list = [0] + full_bb_list
# and down select to just these:
cities = cities.loc[full_bb_list]


# In[ ]:


# Full backbone: NP + Primes plus every 10th of the TSP-ordered remaining cities
if False:
    
    # Get the remaining cities TSP output
    df_bbtsp = pd.read_csv('../input/santa-18-primes-tsp60/remaining_v4B_t5000rand.csv')
    # There are 19740 of these cities, put them in path-order in a list
    all_remaining = list( df_bbtsp.loc[list(df_bbtsp['Path']), 'CityId'])
    # Choose every 10th to add to backbone
    bb_remaining = all_remaining[1: :10]
    # There are 1974 of these

    # Get the 17803 NP+primes
    prime_tour_select = (cities['CityId'] == 0) | (cities['Prime'] == 1)
    prime_cities = list(cities[prime_tour_select]['CityId'])

    # Combine them for 19777 full bb cities
    full_bb_list = prime_cities + bb_remaining
    # Put the bb cities in sorted order, i.e., 'random' in terms of the TSP path
    full_bb_list.sort()
    # and down-select to just these chosen bb cities:
    cities = cities.loc[full_bb_list]


# In[ ]:


# Reset the df index (new column 'index' is the old index i.e. the CityId, can ignore it)
cities = cities.reset_index()


# In[ ]:


# Check the TSP-input cities df and length
#  Primes-only length is  17803
#  Full backbone len is  19777
#  Remaining cities len is 19740
# Remaining cities plus NP is 19741
print(len(cities))
##cities.describe()
cities.head(20)


# In[ ]:


cities.tail(20)


# ## Do the TSP ...

# In[ ]:


# Instantiate solver
solver = TSPSolver.from_data(
    cities.X,
    cities.Y,
    norm="EUC_2D"
)

# Set the time bound - 5000 seems to be 5000-ish seconds
##t_bound = 5.0      # for testing
t_bound = 3000.0  #  1000, 2500 etc for actual run if 5000 uses too much memory
##t_bound = 5000.0   # for actual run
t = time.time()
tour_data = solver.solve(time_bound = t_bound, verbose = True, random_seed = 42)

print(time.time() - t)
print(tour_data.found_tour)


# In[ ]:


# Put the tour sequence in a new Path column
##print( len(cities), len(tour_data.tour) )
cities['Path'] = tour_data.tour

cities.head(15)


# In[ ]:


cities.tail(15)


# ## Output file

# In[ ]:


# Select output file name etc

# Primes: Write out the two columns, CityId and Path, from the prime cities dataframe:
##cities[['CityId','Path']].to_csv('primes_path_t'+str(int(t_bound))+'.csv', index=False)

# FULL BACKBONE: Write out the two columns, CityId and Path, from the cities dataframe:
# suffix 'rand' to indicate random order going into the TSP solver
##cities[['CityId','Path']].to_csv('backbonet60_path_t'+
##                                 str(int(t_bound))+'rand.csv', index=False)

# Remaining cities:
##cities[['CityId','Path']].to_csv('remaining_v4B_t'+
##                                     str(int(t_bound))+'rand.csv', index=False)

# # FULL BACKBONE: Using 10th of path-ordered remaining cities:
##cities[['CityId','Path']].to_csv('backbone_primesTSPremaining_t'+
##                                     str(int(t_bound))+'rand.csv', index=False)

# Remaining cities + NP, from All the Carrots v7B:
cities[['CityId','Path']].to_csv('remaining_v8C_t'+
                                     str(int(t_bound))+'rand.csv', index=False)


# In[ ]:


# Plot tour
lines = [[(cities.X[tour_data.tour[i]],cities.Y[tour_data.tour[i]]),           (cities.X[tour_data.tour[i+1]],cities.Y[tour_data.tour[i+1]])]          for i in range(0,len(cities)-1)]
lc = mc.LineCollection(lines, linewidths=2)
fig, ax = pl.subplots(figsize=(20,20))
ax.set_aspect('equal')
ax.add_collection(lc)
ax.autoscale()


# In[ ]:


get_ipython().system('ls')


# In[ ]:




