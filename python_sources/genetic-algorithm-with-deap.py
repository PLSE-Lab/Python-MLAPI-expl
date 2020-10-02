#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy
import pandas as pd
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
#from scoop import futures
import hashlib
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Genetic Algorithm with DEAP  
# 
# This kernel introduces genetic algorithms and describes a baseline using genetic algorithms through the DEAP optimization library.
# 
# * Introduction Genetic Algorithm
# * DEAP Lib
# * Solution
# 
# **All parameters used were modified from the original version to maintain competitiveness. The kernel is just a didactic example of how to use lib DEAP.**

# ## Introduction Genetic Algorithm
# 
# Basic Description
# Genetic algorithms are inspired by Darwin's theory about evolution. Solution to a problem solved by genetic algorithms is evolved.
# 
# Algorithm is started with a set of solutions (represented by chromosomes) called population. Solutions from one population are taken and used to form a new population. This is motivated by a hope, that the new population will be better than the old one. Solutions which are selected to form new solutions (offspring) are selected according to their fitness - the more suitable they are the more chances they have to reproduce.
# 
# ![](https://tutorials.retopall.com/wp-content/uploads/2019/03/GeneticAlgorithm-1-1024x374.png)
# 
# Outline of the Basic Genetic Algorithm
# * **[Start]** Generate random population of n chromosomes (suitable solutions for the problem)
# * **[Fitness]** Evaluate the fitness f(x) of each chromosome x in the population
#     * **[New population]** Create a new population by repeating following steps until the new population is complete
#     * **[Selection]** Select two parent chromosomes from a population according to their fitness (the better fitness, the bigger chance to be selected)
#     * **[Crossover]** With a crossover probability cross over the parents to form a new offspring (children). If no crossover was performed, offspring is an exact copy of parents.
#     * **[Mutation]** With a mutation probability mutate new offspring at each locus (position in chromosome).
#     * [Accepting] Place new offspring in a new population
# * **[Replace]** Use new generated population for a further run of algorithm
# * **[Test]** If the end condition is satisfied, stop, and return the best solution in current population
# * **[Loop]** Go to step 2

# ## DEAP - Distributed Evoluationary Algorithms In Python
# 
# ![](https://deap.readthedocs.io/en/master/_images/deap_long.png)
# 
# DEAP is a novel evolutionary computation framework for rapid prototyping and testing of ideas. It seeks to make algorithms explicit and data structures transparent. It works in perfect harmony with parallelisation mechanism such as multiprocessing and SCOOP. The following documentation presents the key concepts and many features to build your own evolutions.
# 
# https://deap.readthedocs.io/en/master/overview.html
# https://github.com/deap/deap
# 

# #### Define constants and support methods

# In[ ]:


# Probem COnstants
N_DAYS        = 100
MAX_OCCUPANCY = 300
MIN_OCCUPANCY = 125
FAMINY_SIZE   = 5000
DAYS          = list(range(N_DAYS,0,-1))


#load dataset
data        = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/family_data.csv', index_col='family_id')
submission  = pd.read_csv('/kaggle/input/santa-workshop-tour-2019/sample_submission.csv', index_col='family_id')


# Load util variables
family_size_dict  = data[['n_people']].to_dict()['n_people']
cols              = [f'choice_{i}' for i in range(10)]
choice_dict       = data[cols].T.to_dict()

# from 100 to 1
family_size_ls  = list(family_size_dict.values())
choice_dict_num = [{vv:i for i, vv in enumerate(di.values())} for di in choice_dict.values()]

# Computer penalities in a list
penalties_dict = {
  n: [
      0,
      50,
      50  + 9 * n,
      100 + 9 * n,
      200 + 9 * n,
      200 + 18 * n,
      300 + 18 * n,
      300 + 36 * n,
      400 + 36 * n,
      500 + 36 * n + 199 * n,
      500 + 36 * n + 398 * n
  ]
  for n in range(max(family_size_dict.values())+1)
} 


# ### Toolbox

# In[ ]:


# Create a Tollbox Optmizer

# The creator is a class factory that can build new classes at run-time. It will be called with first the desired name of the new class, 
# second the base class it will inherit, and in addition any subsequent arguments you want to become attributes of your class. 
# This allows us to build new and complex structures of any type of container from lists to n-ary trees.
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Now we will use our custom classes to create types representing our individuals as well as our whole population.
toolbox = base.Toolbox()


# ### Creating the Population
# 
# The population is random. Each individual is a vector of size k, where k = total families and each value is the day chosen by family d, where d varies between 0 and 100.

# In[ ]:


# Attribute generator
toolbox.register("attr_int",   random.randint, 1, 100)

# Structure initializers
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, FAMINY_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

pop   = toolbox.population(n=1000)

print(pop[0])
print("len: ", len(pop))


# ### The Evaluation Function
# 
# The evaluation function is pretty simple in our example. We just need to count the number of ones in an individual.
# The returned value must be an iterable of a length equal to the number of objectives (weights).

# In[ ]:


# The evaluation function is pretty simple in our example. We just need to count the number of ones in an individual.
def cost_function(prediction, family_size_ls, choice_dict, choice_dict_num, penalties_dict):
    penalty = 0

    # We'll use this to count the number of people scheduled each day
    daily_occupancy = {k:0 for k in DAYS}
    
    # Looping over each family; d is the day, n is size of that family, 
    # and choice is their top choices
    for n, c, c_dict, choice in zip(family_size_ls, prediction, list(choice_dict.values()), choice_dict_num):
        d = int(c)
        daily_occupancy[d] += n

        # Calculate the penalty for not getting top preference
        if d not in choice:
            penalty += penalties_dict[n][-1]
        else:
            penalty += penalties_dict[n][choice[d]]

    # for each date, check total occupancy
    #  (using soft constraints instead of hard constraints)
    k = 0
    for v in daily_occupancy.values():
        if (v > MAX_OCCUPANCY):
            k = k + (v - MAX_OCCUPANCY)
        if (v < MIN_OCCUPANCY):
            k = k + (MIN_OCCUPANCY - v)
    #    if k > 0:
    #        penalty += 100000000 
    penalty += 100000*k

    # Calculate the accounting cost
    # The first day (day 100) is treated special
    accounting_cost = (daily_occupancy[DAYS[0]]-125.0) / 400.0 * daily_occupancy[DAYS[0]]**(0.5)
    # using the max function because the soft constraints might allow occupancy to dip below 125
    accounting_cost = max(0, accounting_cost)
    
    # Loop over the rest of the days, keeping track of previous count
    yesterday_count = daily_occupancy[DAYS[0]]
    for day in DAYS[1:]:
        today_count = daily_occupancy[day]
        diff = abs(today_count - yesterday_count)
        accounting_cost += max(0, (daily_occupancy[day]-125.0) / 400.0 * daily_occupancy[day]**(0.5 + diff / 50.0))
        yesterday_count = today_count

    penalty += accounting_cost

    return (penalty, )


# ### The Genetic Operators
# 
# Within DEAP there are two ways of using operators. We can either simply call a function from the tools module or register it with its arguments in a toolbox, as we have already seen for our initialization methods. The most convenient way, however, is to register them in the toolbox, because this allows us to easily switch between the operators if desired. The toolbox method is also used when working with the algorithms module. See the One Max Problem: Short Version for an example.
# 
# Registering the genetic operators required for the evolution in our One Max problem and their default arguments in the toolbox is done as follows.

# In[ ]:


toolbox.register("evaluate",   cost_function, family_size_ls=family_size_ls, choice_dict=choice_dict, 
                                                 choice_dict_num=choice_dict_num, penalties_dict=penalties_dict)
toolbox.register("mate",       tools.cxUniform, indpb=0.5)
toolbox.register("select",     tools.selTournament, tournsize=10) 
toolbox.register("mutate",     tools.mutShuffleIndexes, indpb=0.5)


# ## Run Evolution...

# In[ ]:


ngen      = 100  # Gerations
npop      = 1000 # Population

hof   = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)

# Statistics
stats.register("avg", numpy.mean, axis=0)
stats.register("std", numpy.std, axis=0)
stats.register("min", numpy.min, axis=0)
stats.register("max", numpy.max, axis=0)


# Evolution
pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=npop, lambda_=npop,
                                          cxpb=0.7,   mutpb=0.3, ngen=ngen, 
                                          stats=stats, halloffame=hof)


# In[ ]:


# Best Solution
best_solution = tools.selBest(pop, 1)[0]
print("")
print("[{}] best_score: {}".format(logbook[-1]['gen'], logbook[-1]['min'][0]))


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# History AVG
plt.figure(figsize=(10,8))
front = np.array([(c['gen'], c['avg'][0]) for c in logbook])
plt.plot(front[:,0][1:-1], front[:,1][1:-1], "-bo", c="b")
plt.axis("tight")
plt.show()


# In[ ]:


# Export Result

submission['assigned_day']=best_solution
print(submission.head())
submission.to_csv('submission_{}.csv'.format(logbook[-1]['min'][0]))  


# ## Conclusion
# 
# It has been shown that the DEAP library can be used to optimize the proposed problem. The optimization speed depends on the parameterization of each step of the genetic algorithm.
# 
# I suggest testing the settings presented in the DEAP documentation https://deap.readthedocs.io/en/master/api/tools.html

# In[ ]:




