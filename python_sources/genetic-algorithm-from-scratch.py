#!/usr/bin/env python
# coding: utf-8

# In[ ]:


geneSet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ!."
target="AshishPatel"


# In[ ]:


import random
import datetime
def gen_parent(length):
    genes=[]
    while len(genes)<length:
            sampleSize=min(length-len(genes),len(geneSet))
            genes.extend(random.sample(geneSet,sampleSize))
    return ''.join(genes)


# In[ ]:


def get_fitness(guess):
      return sum(1 for expected,actual in zip(target,guess) if expected==actual)


# In[ ]:


def mutate(parent):
    index=random.randrange(0,len(parent))
    childGenes=list(parent)
    newGene,alternate=random.sample(geneSet,2)
    childGenes[index]=alternate if newGene==childGenes[index] else newGene
    return ''.join(childGenes)


# In[ ]:


def display(guess):
    timeDiff=datetime.datetime.now()-startTime
    fitness=get_fitness(guess)
    print("{}\t{}\t{}".format(guess,fitness,timeDiff))


# In[ ]:


random.seed()
startTime=datetime.datetime.now()
bestParent=gen_parent(len(target))
bestFitness=get_fitness(bestParent)
display(bestParent)


# In[ ]:


while True:
    child=mutate(bestParent)
    childFitness=get_fitness(child)
    if bestFitness>=childFitness:
           continue
    display(child)
    if childFitness>=len(bestParent):
           break
    bestFitness=childFitness
    bestParent=child

