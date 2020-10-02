#!/usr/bin/env python
# coding: utf-8

# # We're going to optimize a very simple problem: trying to create a list of N numbers that equal X when summed together.
# 
# # EX1:  N = 5 ; X = 10 ; one solution is [2, 0, 0 ,4, 2]
# 
# # EX2: N = 5 and X = 200, then these would all be appropriate solutions.
# ## lst = [40,40,40,40,40]
# ## lst = [50,50,50,25,25]
# ## lst = [200,0,0,0,0]
# 

# In[ ]:


from random import randint, random
from operator import add 
from functools import *


# # Ingredients of The Solution
# 
# Each suggested solution for a genetic algorithm is referred to as an individual. In our current problem, each list of N numbers is an individual.
# 
# individual(5,0,100)
# [79, 0, 20, 47, 40]
# 
# individual(5,0,100)
# [64, 1, 25, 84, 87]

# In[ ]:


def individual(length, min, max):
    'Create a member of the population.'
    return [ randint(min,max) for x in range(length) ]


# # The collection of all individuals is referred to as our population.
# 
# population(3,5,0,100)
# 
# [[51, 55, 73, 0, 80],
# 
# [3, 47, 18, 65, 55], 
# 
# [17, 64, 77, 43, 48]]

# In[ ]:


def population(count, length, min, max):
    """
    Create a number of individuals (i.e. a population).

    count: the number of individuals in the population
    length: the number of values per individual
    min: the minimum possible value in an individual's list of values
    max: the maximum possible value in an individual's list of values

    """
    return [ individual(length, min, max) for x in range(count) ]


# # Define the fitness function. 
# 
# For our problem, we want the fitness to be a function of the distance between the sum of an individuals numbers and the target number X.
# 
# We define: fitness = sum(individuals) - target , fitness = 0 is the best solution,  and the higher the worse.

# In[ ]:


def fitness(individual, target):
    """
    Determine the fitness of an individual. Higher is better.

    individual: the individual to evaluate
    target: the target number individuals are aiming for
    """
    sum = reduce(add, individual, 0)
    return abs(target-sum)


# # It's also helpful to create a function that will determine a population's average fitness

# In[ ]:


def grade(pop, target):
    'Find average fitness for a population.'
    summed = reduce(add, (fitness(x, target) for x in pop))
    return summed / (len(pop) * 1.0)


# # Now we just need a way evolve our population; to advance the population from one generation to the next.
# 
# For each generation we'll take a portion of the best performing individuals as judged by our fitness function. These high-performers will be the parents of the next generation.
# 
# # Breed together parents to repopulate the population to its desired size (if you take the top 20 individuals in a population of 100, then you'd need to create 80 new children via breeding).
# 
# ### In our case, breeding is pretty basic: take the first N/2 digits from the father and the last N/2 digits from the mother.
# 
# father = [1,2,3,4,5,6]
# 
# mother = [10,20,30,40,50,60]
# 
# child = father[:3] + mother[3:]
# 
# child: [1,2,3,40,50,60]
# 
# Merge together the parents and children to constitute the next generation's population.
# 
# 
# # Finally we mutate a small random portion of the population. What this means is to have a probability of randomly modifying each individual.
# 
# 

# In[ ]:


def evolve(pop, target, retain=0.2, random_select=0.05, mutate=0.01):
    graded = [ (fitness(x, target), x) for x in pop]
    graded = [ x[1] for x in sorted(graded)]
    retain_length = int(len(graded)*retain)
    parents = graded[:retain_length]
    # randomly add other individuals to
    # promote genetic diversity
    for individual in graded[retain_length:]:
        if random_select > random():
            parents.append(individual)
    # mutate some individuals
    for individual in parents:
        if mutate > random():
            pos_to_mutate = randint(0, len(individual)-1)
            # this mutation is not ideal, because it
            # restricts the range of possible values,
            # but the function is unaware of the min/max
            # values used to create the individuals,
            individual[pos_to_mutate] = randint(
                min(individual), max(individual))
    # crossover parents to create children
    parents_length = len(parents)
    desired_length = len(pop) - parents_length
    children = []
    while len(children) < desired_length:
        male = randint(0, parents_length-1)
        female = randint(0, parents_length-1)
        if male != female:
            male = parents[male]
            female = parents[female]
            half = int(len(male) / 2)
            child = male[:half] + female[half:]
            children.append(child)        
    parents.extend(children)
    return parents


# In[ ]:


# find the best solution
def find_best(pop,target):
    best = None
    val = 10^20  ######### very large number   ,  0 is the best
    for individual in pop:
        r = fitness(individual, target)
        if r < val:
            best = individual
            val = r
    return best, val


# In[ ]:


# Example usage

target = 20
p_count = 1000
i_length = 5
i_min = 0
i_max = 100

# initialize population, estimate average fitness of it
p = population(p_count, i_length, i_min, i_max)
fitness_history = [grade(p, target),]


for i in range(100):
    p = evolve(p, target)
    fitness_history.append(grade(p, target))
    if grade(p, target) < 10e-3:
        print("loop:", i, "average fitness:", grade(p, target))
        break

best, val = find_best(p,target)
print("best solution", best, "fitness value", val, "target", target)   # 

print ("history_fitness")
for datum in fitness_history:
    print (datum)


# In[ ]:





# In[ ]:




