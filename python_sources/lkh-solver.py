#!/usr/bin/env python
# coding: utf-8

# <h2>LKH 2.0.9</h2>
# 
# LKH is an effective implementation of the Lin-Kernighan heuristic and is considered one of the best algorithms available for the TSP problem. Even though the algorithm is approximate, optimal solutions are produced with an impressively high frequency. It has been implemented in the programming language C and is distributed for academic and non-commercial use.
# 
# In this notebook I will show how to run LKH for this competition in a kernel environment. For more information please refer to [the official page](http://akira.ruc.dk/~keld/research/LKH/) or the [original paper](http://akira.ruc.dk/~keld/research/LKH/LKH-2.0/DOC/LKH_REPORT.pdf). 
# 
# Notes:
# 
# * I'm not considering the prime penalty.
# * Internet must be enabled.
# 
# 
# Acknowledgments:  LKH author, [Keld Helsgaun](https://www.kaggle.com/keldhelsgaun).

# <h2>1. Read files</h2>

# In[ ]:


import os
import pandas as pd
cities = pd.read_csv('../input/cities.csv', index_col=['CityId'], nrows=None)
cities = cities * 1000  # not sure if coords are rounded as concorde
cities.head()


# <h2>2. Build LKH</h2>

# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'wget http://akira.ruc.dk/~keld/research/LKH/LKH-2.0.9.tgz\ntar xvfz LKH-2.0.9.tgz\ncd LKH-2.0.9\nmake')


# <h2>3. Prepare inputs</h2>
# 
# We need to write a TSPLIB file and a parameters file to run LKH. 

# In[ ]:


def write_tsp(nodes, filename, name='traveling-santa-2018-prime-paths'):
    # From https://www.kaggle.com/blacksix/concorde-for-5-hours.
    with open(filename, 'w') as f:
        f.write('NAME : %s\n' % name)
        f.write('COMMENT : %s\n' % name)
        f.write('TYPE : TSP\n')
        f.write('DIMENSION : %d\n' % len(cities))
        f.write('EDGE_WEIGHT_TYPE : EUC_2D\n')
        f.write('NODE_COORD_SECTION\n')
        for row in cities.itertuples():
            f.write('%d %.11f %.11f\n' % (row.Index + 1, row.X, row.Y))
        f.write('EOF\n')

write_tsp(cities, '../working/LKH-2.0.9/cities.tsp')


# For a full list of parameters check the [LKH User Guide](http://akira.ruc.dk/~keld/research/LKH/LKH-2.0/DOC/LKH-2.0_USER_GUIDE.pdf).  I've also created a [discussion here](https://www.kaggle.com/c/traveling-santa-2018-prime-paths/discussion/73694), since there are many options.

# In[ ]:


def write_parameters(parameters, filename='../working/LKH-2.0.9/params.par'):
    with open(filename, 'w') as f:
        for param, value in parameters:
            f.write("{} = {}\n".format(param, value))
    print("Parameters saved as", filename)

parameters = [
    ("PROBLEM_FILE", "cities.tsp"),
    ("OUTPUT_TOUR_FILE", "tsp_solution.csv"),
    ("SEED", 2018),
    ('CANDIDATE_SET_TYPE', 'POPMUSIC'), #'NEAREST-NEIGHBOR', 'ALPHA'),
    ('INITIAL_PERIOD', 10000),
    ('MAX_TRIALS', 1000),
]
write_parameters(parameters)


# <h2>4. Run and write submission</h2>
# 
# I'm using a timeout to kill the process after 5 hours.

# In[ ]:


get_ipython().run_cell_magic('bash', '-e', 'cd ./LKH-2.0.9\ntimeout 18000s ./LKH params.par')


# Read the output file and make a list (tour) of cities.

# In[ ]:


def read_tour(filename):
    tour = []
    for line in open(filename).readlines():
        line = line.replace('\n', '')
        try:
            tour.append(int(line) - 1)
        except ValueError as e:
            pass  # skip if not a city id (int)
    return tour[:-1]

tour = read_tour('../working/LKH-2.0.9/tsp_solution.csv')
print("Tour length", len(tour))


# Finally, we can print the score (considering prime twist) and write the submission file.

# In[ ]:


import numpy as np
import sympy

def score_tour(tour):
    df = cities.reindex(tour + [0]).reset_index()
    primes = list(sympy.primerange(0, len(cities)))
    df['prime'] = df.CityId.isin(primes).astype(int)
    df['dist'] = np.hypot(df.X - df.X.shift(-1), df.Y - df.Y.shift(-1))
    df['penalty'] = df['dist'][9::10] * (1 - df['prime'][9::10]) * 0.1
    return df.dist.sum() + df.penalty.sum()

def write_submission(tour, filename):
    assert set(tour) == set(range(len(tour)))
    pd.DataFrame({'Path': list(tour) + [0]}).to_csv(filename, index=False)

print("Final score", score_tour(tour))
write_submission(tour, 'submission.csv')


# Thanks for reading and merry christmas!
