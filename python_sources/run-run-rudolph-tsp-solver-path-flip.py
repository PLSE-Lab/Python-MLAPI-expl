#!/usr/bin/env python
# coding: utf-8

# The "Corcorde traveling salesman problem" solver is doing much better than other base methods as [XYZT points out in his TSP Solver kernel](https://www.kaggle.com/thexyzt/xyzt-s-visualizations-and-various-tsp-solvers). In this kernel I'd like to explore the Concorde TSP solver mixed with a path flip approach to optimize for the prime portion of this competition as [Matthew Anderson explores in his Flip It kernel](https://www.kaggle.com/matthewa313/flip-it).
# 
# **NOTE:** in order for this to work you'll need to import the "jvkersch/pyconcorde" GitHub repo using the option in the notebook editor called "Packages". This GibHub package is a Python wrapper around the original C code written as the "Concorde TSP Solver". If you want to learn more, check out the [Concorder TSP Solver page](http://www.math.uwaterloo.ca/tsp/concorde.html) on the University of Waterloo website.

# In[ ]:


import numpy as np
import pandas as pd
import os
from concorde.tsp import TSPSolver
#from sympy import primerange

pd.options.mode.chained_assignment = None  # default='warn'

# find cities that are prime numbers
def sieve_of_eratosthenes(n):
    n = int(n)
    primes = [True for i in range(n+1)] # Start assuming all numbers are primes
    primes[0] = False # 0 is not a prime
    primes[1] = False # 1 is not a prime
    for i in range(2,int(np.sqrt(n)) + 1):
        if primes[i]:
            k = 2
            while i*k <= n:
                primes[i*k] = False
                k += 1
    return(primes)

santa_cities = pd.read_csv('../input/cities.csv')
primes_cities = sieve_of_eratosthenes(max(santa_cities.CityId))
santa_cities.head()


# The core code of the Concorde "traveling salesman problem" solver below came from [XYZT's kernel](https://www.kaggle.com/thexyzt/xyzt-s-visualizations-and-various-tsp-solvers). If you want to learn more, check out the [Concorder TSP Solver page](http://www.math.uwaterloo.ca/tsp/concorde.html) on the University of Waterloo website.

# In[ ]:


# calculate total distance of the path
def total_distance(dfcity,path):
    prev_city = path[0]
    total_distance = 0
    step_num = 1
    for city_num in path[1:]:
        next_city = city_num
        total_distance = total_distance +             np.sqrt(pow((dfcity.X[city_num] - dfcity.X[prev_city]),2) + pow((dfcity.Y[city_num] - dfcity.Y[prev_city]),2)) *             (1+ 0.1*((step_num % 10 == 0)*int(not(prime_cities[prev_city]))))
        prev_city = next_city
        step_num = step_num + 1
    return total_distance

# run Concorder TSP Solver to calculate the best path
def concorde_tsp(cities, seed=42):
    solver = TSPSolver.from_data(cities.X, cities.Y, norm="EUC_2D")
    tour_data = solver.solve(time_bound=60.0, verbose=True, random_seed=seed)
    if tour_data.found_tour:
        path = np.append(tour_data.tour,[0])
        return path
    else:
        return None

best_path = concorde_tsp(santa_cities)
print("Total concorde distance is {:,}".format(total_distance(santa_cities,best_path)))


# The core code below came from [Matthew Anderson kernel called Flip It](https://www.kaggle.com/matthewa313/flip-it) that attempts to flip the entire path order to see if it helps the prime penalty lessen.

# In[ ]:


def score_path(tour):
    # length of any given tour with primes calculation
    df = santa_cities.reindex(tour + [0]).reset_index()
    # mark which cities are prime
    df['prime'] = df.CityId.isin(primes_cities).astype(int)
    # calculate the euclidean norm
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()

# Let's take a look at our tour
print("Existing path (0-5):",best_path[0:5])
# And the flipped tour looks like:
path_flip = best_path[::-1]
print("Flipped path (0-5):", path_flip[0:5])
# The scores of our tours are:
print("Score of original tour:", score_path(best_path))
print("Score of flipped tour:", score_path(path_flip))

# If the flipped tour is quicker, change our tour:
if score_path(path_flip) < score_path(best_path):
    print("The total improvement was:", abs(score_path(path_flip) - score_path(best_path)))
    best_path = path_flip 
    print("The better of the original/flipped tour is:", best_path[0:5])


# Generate the output file for submission to the competition.

# In[ ]:


pd.DataFrame({'Path':best_path}).to_csv('santa-path.csv',index = False)

