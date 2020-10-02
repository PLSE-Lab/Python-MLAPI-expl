#!/usr/bin/env python
# coding: utf-8

# There are some implementation of genetic approach in this competition, but they are not very fast. I really wanted to run 10000 epochs in 120 seconds.
# 
# Using numpy for almost all computations.
# 
# Using fast scoring function from https://www.kaggle.com/kernels/scriptcontent/24287559/notebook
# 
# With changes:
# 
#     - if (day_occ[d]<125 || day_occ[d]>300) return max_cost;
#     + if (day_occ[d]<125)                                                       
#     +   r += 100000 * (125 - day_occ[d]);                                      
#     + else if (day_occ[d] > 300)                                               
#     +   r += 100000 * (day_occ[d] - 300);

# In[ ]:


get_ipython().run_cell_magic('writefile', 'score.c', '\n#include <stdio.h>\n#include <stdlib.h>\n#include <string.h>\n#include <math.h>\n\n#define NF 5000\nint cost[NF][101];\nint fs[NF];\n\nint cf[NF][10];\n\nint loaded=0;\n\nfloat acc[301][301];\n\nvoid precompute_acc() {\n    \nfor(int i=125;i<=300;i++) \n    for(int j=125;j<=300;j++)\n      acc[i][j] = (i-125.0)/400.0 * pow(i , 0.5 + fabs(i-j) / 50 );    \n}\n\nvoid read_fam() {\n  FILE *f;\n  char s[1000];\n  int d[101],fid,n;\n  int *c;\n\n  f=fopen("../input/santa-workshop-tour-2019/family_data.csv","r");\n  if (fgets(s,1000,f)==NULL)\n    exit(-1);\n\n  for(int i=0;i<5000;i++) {\n    c = &cf[i][0];\n    if (fscanf(f,"%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",\n               &fid,&c[0],&c[1],&c[2],&c[3],&c[4],&c[5],&c[6],&c[7],&c[8],&c[9],&fs[i])!=12)\n      exit(-1);\n\n    //    printf("%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\\n",\n    //fid,c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],c[9],fs[i]);\n    n = fs[i];\n\n    for(int j=1;j<=100;j++) {\n      if (j==c[0]) cost[i][j]=0;\n      else if (j==c[1]) cost[i][j]=50;\n      else if (j==c[2]) cost[i][j]=50 + 9 * n;\n      else if (j==c[3]) cost[i][j]=100 + 9 * n;\n      else if (j==c[4]) cost[i][j]=200 + 9 * n;\n      else if (j==c[5]) cost[i][j]=200 + 18 * n;\n      else if (j==c[6]) cost[i][j]=300 + 18 * n;\n      else if (j==c[7]) cost[i][j]=300 + 36 * n;\n      else if (j==c[8]) cost[i][j]=400 + 36 * n;\n      else if (j==c[9]) cost[i][j]=500 + 36 * n + 199 * n;\n      else cost[i][j]=500 + 36 * n + 398 * n;\n    }\n  }\n\n}\n\nfloat max_cost=1000000000;\n\nint day_occ[102];\n\nstatic inline int day_occ_ok(int d) {\n  return !(d <125 || d>300);\n}\n\nfloat score(int *pred) {\n  float r=0;\n    \n  if (!loaded) {\n      read_fam();\n      precompute_acc();\n      loaded = 1;\n  }\n\n  // validate day occupancy\n  memset(day_occ,0,101*sizeof(int));\n\n  for(int i=0;i<NF;i++) {\n    day_occ[pred[i]]+=fs[i];\n    r+=cost[i][pred[i]];\n  }\n       \n  day_occ[101]=day_occ[100];\n\n  for (int d=1;d<=100;d++) {\n    if (day_occ[d]<125)                                                       \n      r += 100000 * (125 - day_occ[d]);                                      \n    else if (day_occ[d] > 300)                                               \n      r += 100000 * (day_occ[d] - 300);      \n    r += acc[day_occ[d]][day_occ[d+1]];\n  }\n  return r;\n}  \n\nvoid score_bunch(int *pred, int n, float *dest) {\n    for(int i = 0; i < n; ++i) {\n        dest[i] = score(pred + i * NF);\n    }\n}')


# In[ ]:


get_ipython().system('gcc -O5 -shared -Wl,-soname,score     -o score.so     -fPIC score.c')
get_ipython().system('ls -l score.so')


# In[ ]:


import ctypes
from numpy.ctypeslib import ndpointer

lib = ctypes.CDLL('./score.so')
cost_function = lib.score
cost_function.restype = ctypes.c_float
cost_function.argtypes = [ndpointer(ctypes.c_int)]

_bunch_cost_function = lib.score_bunch
_bunch_cost_function.argtypes = [ndpointer(ctypes.c_int), ctypes.c_int, ndpointer(ctypes.c_float)]
def bunch_cost_function(bunch):
    result = np.zeros(bunch.shape[0], dtype='float32')
    _bunch_cost_function(bunch, bunch.shape[0], result)
    return result


# ## Reading data

# In[ ]:


import numpy as np
import pandas as pd
np.random.seed(666)

fpath = '../input/santa-workshop-tour-2019/family_data.csv'
data = pd.read_csv(fpath, index_col='family_id')

fpath = '/kaggle/input/santa-workshop-tour-2019/sample_submission.csv'
submission = pd.read_csv(fpath, index_col='family_id')
data.head()


# ## Algorithm
# 
# * Choosing parents randomly with equal probability.
# * crossover (for each family choose day from one of parents with equal probability)
# * mutate children (set random day for random family or set day which reduces preference cost for random family)
# * selection (keep best from initial_population + children, remove dublicates)
# 

# In[ ]:


def iteration(population, costs=None, size_of_population=100, n_of_childs=100):
    if costs is None:
        costs = np.array([cost_function(population[i]) for i in range(population.shape[0])])
    assert costs.shape[0] == population.shape[0]
    assert population.shape != (5000, 5000)

    parents0 = population[np.random.choice(population.shape[0], n_of_childs)]
    parents1 = population[np.random.choice(population.shape[0], n_of_childs)]
    
    children = crossover(parents0, parents1)
    children = mutate(children)
    
    return selection(population, costs, children, size_of_population)


# In[ ]:


def crossover(parents0, parents1):
    crossover_mask = np.random.choice([False, True], parents0.shape)
    return np.where(crossover_mask, parents0, parents1)


# In[ ]:


matrix = data[['choice_0', 'choice_1', 'choice_2', 'choice_3', 'choice_4',
       'choice_5', 'choice_6', 'choice_7', 'choice_8', 'choice_9']].to_numpy()

better_choices = []
for family in range(5000):
    days = matrix[family]
    bc = [days for i in range(101)]
    for di in range(len(days)):
        bc[days[di]] = days[:di]
    better_choices.append(bc)

    
def mutate_simple(children, indices):
    families_to_mutate = np.random.choice(5000, indices.shape[0])
    new_days = np.random.randint(1, 101, size=indices.shape[0])
    children[indices, families_to_mutate] = new_days


best_possible_days = np.int32(data.choice_0.values)
    
def mutate_improving(children, indices):
    improvable_families = (children[indices] != best_possible_days)
    for i in range(indices.shape[0]):
        family = np.random.choice(np.nonzero(improvable_families[i])[0])
        children[indices[i]][family] = np.random.choice(better_choices[family][children[indices[i]][family]])
    
    
def mutate(children):
    methods = [
        (mutate_simple, 90),
        (mutate_improving, 100),
    ]
    p = np.random.choice(methods[-1][1], children.shape[0])
    method_booleans = [p < m[1] for m in methods]
    method_indices = [
        np.nonzero(bv)[0]
        for bv in [method_booleans[0]] + [
            ~method_booleans[i-1] & method_booleans[i]
            for i in range(1, len(method_booleans))
        ]
    ]
    for i in range(len(methods)):
        methods[i][0](children, method_indices[i])

    return children


# In[ ]:


def selection(population, costs, children, size_of_population):
#    children_costs = np.array([cost_function(children[i]) for i in range(children.shape[0])])
    children_costs = bunch_cost_function(children)
    
    # throwing away children which are worse than worst of population
    if population.shape[0] >= size_of_population:
        good_children = children_costs < costs.max()
        children_costs = children_costs[good_children]
        children = children[good_children]
    
    new_costs = np.concatenate((costs, children_costs))
    new_population = np.concatenate((population, children))
    
    # sorting by cost
    indices = np.argsort(new_costs)
    new_population = new_population[indices]
    new_costs = new_costs[indices]

    new_population, new_costs = remove_dublicates(new_population, new_costs)
    new_population = new_population[:size_of_population]
    new_costs = new_costs[:size_of_population]
    return new_population, new_costs, children.shape[0]

def remove_dublicates(new_population, new_costs):
    # removing dublicates
    eqsn = np.array([True] * new_population.shape[0])
    
    begin = 0
    d_indices = []
    for i in range(1, new_population.shape[0]):
        if new_costs[i] != new_costs[i-1]:
            d_indices += get_dublicates_indices(new_population, np.arange(begin, i))
            begin = i
    d_indices += get_dublicates_indices(new_population, np.arange(begin, new_population.shape[0]))
    eqsn[d_indices] = False
    
    return new_population[eqsn], new_costs[eqsn]

def get_dublicates_indices(data, indices):
    not_dublicates = [indices[0]]
    for i in indices[1:]:
        if not any([np.array_equal(data[i], data[nd]) for nd in not_dublicates]):
            not_dublicates.append(i)
    return [i for i in indices if i not in not_dublicates]


# ## Run

# In[ ]:


population = np.array([np.int32(submission.assigned_day.values)])
[cost_function(s) for s in population]


# In[ ]:


get_ipython().run_cell_magic('time', '', 'cost = np.array([cost_function(population[i]) for i in range(population.shape[0])])\nprint(0, np.sort(cost)[:3], cost.max())\nfor n in range(10000):\n    population, cost, _ = iteration(population, cost, 100, 100)\n    if n % 100 == 0:\n        print(n, np.sort(cost)[:3], cost.max())\n        \nprint(10000, np.sort(cost)[:3], cost.max())')


# In[ ]:


submission['assigned_day'] = population[0]
submission.to_csv('submission.csv')
submission

