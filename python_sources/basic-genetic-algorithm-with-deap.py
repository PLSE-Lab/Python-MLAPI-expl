#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import random
import numpy as np
from deap import base, creator, tools


# In[ ]:


tools.mutShuffleIndexes([0,1,2,0,0,1,2,2,0],0.2)


# In[ ]:


random.seed(30)
min_max_weight = 1.0
generation = 50


# In[ ]:


def eval_func(individual):
    
    def to_int(b):
        return int(b, 3)

    i = to_int(''.join((str(xi) for xi in individual)))
    
    return i-2,


# In[ ]:


creator.create("FitnessMax", base.Fitness, weights=(min_max_weight,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# In[ ]:


tbx = base.Toolbox()


# In[ ]:


INDIVIDUAL_SIZE = 9

tbx.register("attr_int", random.randint, 0, 2)
tbx.register("individual", 
             tools.initRepeat, 
             creator.Individual,
             tbx.attr_int, 
             n=INDIVIDUAL_SIZE)

tbx.register("population", tools.initRepeat, list, tbx.individual)


# In[ ]:


tbx.register("evaluate", eval_func)

tbx.register("mate", tools.cxOnePoint)
tbx.register("mutate", tools.mutFlipBit, indpb=0.01)
tbx.register("select", tools.selBest)


# In[ ]:


def set_fitness(population):
    fitnesses = [ 
        (individual, tbx.evaluate(individual)) 
        for individual in population 
    ]

    for individual, fitness in fitnesses:
        individual.fitness.values = fitness
        
def pull_stats(population, iteration=1):
    fitnesses = [ individual.fitness.values[0] for individual in population ]
    return {
        'i': iteration,
        'mu': np.mean(fitnesses),
        'std': np.std(fitnesses),
        'max': np.max(fitnesses),
        'min': np.min(fitnesses)
    }


# In[ ]:


def sortSecond(val): 
    return val[1]  

# def elite_select(current_population, pop_len ):
#     selected_pop = []
#     ind_fit_list = []
#     for ind in current_population:
#         ind_fit_list.append((ind, eval_func(ind)))
        
#     ind_fit_list.sort(key = sortSecond, reverse = True)
    
#     ind_picked = 0
#     for ind_pair in ind_fit_list:
#         selected_pop.append(ind_pair[0])
#         ind_picked += 1
#         if ind_picked >= pop_len:
#             break
    
#     return selected_pop
        
        


# In[ ]:


## create random population,
population = tbx.population(n=4)

## set fitness,
set_fitness(population)


# In[ ]:


## quick look at the initial population,
population[:5]


# In[ ]:


## globals,
stats = []


# In[ ]:


iteration = 1
while iteration <= generation:
    
    current_population = list(map(tbx.clone, population))
    
    offspring = []
    for _ in range(10):
        i1, i2 = np.random.choice(range(len(population)), size=2, replace=False)

        offspring1, offspring2 =             tbx.mate(population[i1], population[i2])

        offspring.append(tbx.mutate(offspring1)[0])
        offspring.append(tbx.mutate(offspring2)[0])  
    
    for child in offspring:
        current_population.append(child)

    ## reset fitness,
    set_fitness(current_population)

    population[:] = tbx.select(current_population, len(population))
    
    print(population)
    
    ## set fitness on individuals in the population,
    stats.append(pull_stats(population, iteration))
    
    print('generation: ',iteration)
    for ind in population:
        print(ind, ', fitness value: ',eval_func(ind)[0])
    
    iteration += 1


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


# In[ ]:


_ = plt.scatter(range(1, len(stats)+1), [ s['max'] for s in stats ], marker='.')

_ = plt.title('max fitness per iteration')
_ = plt.xlabel('iterations')
_ = plt.ylabel('fitness')

plt.show()


# In[ ]:




