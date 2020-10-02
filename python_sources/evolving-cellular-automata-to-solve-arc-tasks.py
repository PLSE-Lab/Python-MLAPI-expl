#!/usr/bin/env python
# coding: utf-8

# # Evolving Cellular Automata to Solve ARC Tasks

# In[ ]:


import numpy as np

import os
import json

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib.animation import ArtistAnimation
from matplotlib import colors

from IPython.display import Image, display

from scipy.ndimage import convolve

from tqdm.notebook import tqdm 

global cmap
global norm

cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


# # Abstract Reasoning Corpus (ARC) Tasks

# "ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence."
# 
# For those unfamiliar with ARC tasks:
# 
# [The Abstraction and Reasoning Corpus (ARC) - GitHub](https://github.com/fchollet/ARC)

# In[ ]:


def plot_task(task):
    trainInput = taskTrain[task][0]['input']
    trainOutput = taskTrain[task][0]['output']
    
    testInput = taskTest[task][0]['input']
    testOutput = taskTest[task][0]['output']
    
    fig, axs = plt.subplots(1,4, figsize=(12,6))
    
    plt.setp(axs, xticks = [], yticks =[], xticklabels=[], yticklabels=[] )
    
    plt.suptitle(taskFileNames[task])
    
    axs[0].imshow(trainInput,cmap=cmap,norm=norm)
    axs[1].imshow(trainOutput,cmap=cmap,norm=norm)
    axs[2].imshow(testInput,cmap=cmap,norm=norm)
    axs[3].imshow(testOutput,cmap=cmap,norm=norm)
    
    axs[0].set_title('Train Input 0')
    axs[1].set_title('Train Output 0')
    axs[2].set_title('Test Input 0')
    axs[3].set_title('Test Output 0')
    
    plt.show()

# load in task files
evalPath = '/kaggle/input/abstraction-and-reasoning-challenge/training/'
taskTrain = list(np.zeros(len(os.listdir(evalPath))))
taskFileNames = list(np.zeros(len(os.listdir(evalPath))))
taskTest = list(np.zeros(len(os.listdir(evalPath))))

for i,file in enumerate(os.listdir(evalPath)):
    with open(evalPath + file, 'r') as f:
        task = json.load(f)
        taskFileNames[i] = file
        taskTrain[i] = []
        taskTest[i] = []
        
        for t in task['train']:
                 taskTrain[i].append(t)
        for t in task['test']:
                taskTest[i].append(t)
        
# plot 5 random tasks as examples
for i in np.random.randint(0,len(taskTrain),5):
    plot_task(i)


# # Cellular Automata

# Background:
# [A cellular automaton consists of a regular grid of cells, each in one of a finite number of states, such as on and off.](https://en.wikipedia.org/wiki/Cellular_automaton)
# 
# Project inspired by:
# [Cellular Automata as a Language for Reasoning](https://www.kaggle.com/arsenynerinovsky/cellular-automata-as-a-language-for-reasoning)

# ## 1D
# 
# [Rule 30](https://en.wikipedia.org/wiki/Rule_30)

# In[ ]:


n = 100
m = 200

# create grid and set center cell in first row to alive as initial condition
grid = np.zeros((n,m))
grid[0,len(grid[0])//2] = 1

# rule 30 kernerls
rules = [[1,0,0],[0,1,1],[0,1,0],[0,0,1]]

fig, ax = plt.subplots(1,1, figsize = (12,6))
plt.setp(ax, xticklabels = [], xticks = [], yticklabels = [], yticks = [])

images = []

for i in range(len(grid)-1):
    for rule in rules:
        
        # apply each rule as a convolution and 
        convRule = convolve(grid[i], rule, mode = 'constant')
        conv = convolve(grid[i], [1,1,1], mode = 'constant')
        
        # compare only non-zero values
        conv[conv == 0] = -1
        
        # rule is true where conv == convRule
        grid[i+1][np.equal(conv,convRule)] = 1
        
        
    images.append([plt.imshow(grid, cmap = 'gray')])

plt.close()


# In[ ]:


# save movie to file
ani = ArtistAnimation(fig, images)
ani.save('Rule30.gif', writer='imagemagick', fps = 8 )


# ![](https://i.imgur.com/OySRZHU.gif)

# ## 2D
# 
# [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life)

# In[ ]:


n = 200
m = 200
grid = np.random.randint(0,2,(n,m))

# kernel to get neighbours, excluding cell being 'looked at'
k = [[1,1,1],
     [1,0,1],
     [1,1,1]]

fig, ax = plt.subplots(1,1, figsize = (12,6))
plt.setp(ax, xticklabels = [], xticks = [], yticklabels = [], yticks = [])
images = []
steps = 100

for step in range(steps):

    # apply convolution to find neighbour count
    conv = convolve(grid, k, mode = 'constant')

    newGrid = np.zeros((n,m))

    # apply GOL rules
    newGrid[conv == 3] = 1
    newGrid[np.logical_and((grid == 1), (conv == 2))] = 1

    grid = newGrid

    images.append([plt.imshow(grid, cmap = 'gray')])
   
plt.close()


# In[ ]:


# save movie to file
ani = ArtistAnimation(fig, images)
ani.save('GOL.gif', writer='imagemagick', fps = 8 )


# ![](https://imgur.com/7ESLxd3.gif)

# # Genetic Algorithm

# Using the same logic in applying rules to the grid of Conway's Game of Life, we can create an array of more specific rulesets that can evolve to transform the input grid into the output grid. 

# ## DNA 
# 
# Each member of the population will be a sequence of encoded rules (genes) to be applied to the task input.
# 
# ## Gene Rule Encoding
# 
# kernel: 3x3 kernel to indicate relative neighbours of interest
# colour rule applies to: the colour of the neighbours we are looking for when we convolve
# old colour: rule only applies to cells that were this colour
# threshold: the minimum number of neighbours required to apply the rule
# 
# | index | 0-8    | 9                      | 10         | 11         |12          |
# |-------|--------|------------------------|------------|------------|------------|
# |       | kernel | colour rule applies to | old colour | new colour |threshold   |
# 
# 

# We will optimize this sequence of rules (DNA) using a genetic algorithm.
# 
# Pseudocode for the solve_tasks():
# 
# initialize population  
# &nbsp;&nbsp;&nbsp;&nbsp; for generation in nGenerations:  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; for member in population:  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Calculate Fitness   
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;-weighted average of correctly estimated cells  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Mating Selection   
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -tournament selection  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Reproduction  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -single point crossover  
# &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -random mutations to DNA  
# 
# (repeat for each step if required)

# In[ ]:


def create_rules(dna):
    
    dna = dna.reshape(len(dna)//13,13)
    
    # randomize order of gene rule application
    np.random.shuffle(dna)
    
    def rules(grid,output):
        
        for gene in dna:
            
            # binary 3x3 kernel indicating state of neighbours
            kernel = np.array([gene[:3],gene[3:6],gene[6:9]])
            
            # colours involved and threshold
            ruleColour = gene[9]
            oldColour = gene[10]
            newColour = gene[11]
            threshold = gene[12]
            
            # get binary grid indicating whether cell is ruleColour or not
            gridRuleColour = (grid == ruleColour).astype(int)
            
            c = convolve(gridRuleColour, kernel, mode='constant')
            
            
            # get boolean matrix of cells that satify rule condition
            rule = np.logical_and(c > threshold, grid == oldColour)
            
            output[rule] = newColour
            
        return output
    return rules 


# # Solving Tasks

# In[ ]:


import ga_utils as ga
import arc_utils as arc


# In[ ]:


def solve_task(task, taskTrain, taskTest, sameShape, nGenes,gens,steps,mutationRate):

    # count number of training examples and tests
    nExamples = len(taskTrain[task])
    nTests = len(taskTest[task])
    
    # empty list to be filled with best rules for each step
    bestSteps = [0]*steps
    
    for step in range(steps):
        
        best = 0
        bestCount = 0
        
        fitness = np.zeros((popSize,nExamples + 1))
        fitMean = np.zeros(gens)
        fitMax = np.zeros(gens)
        

        population = ga.create_population(popSize,nGenes)
        
        for g in tqdm(range(gens)):
            
            for i in range(len(population)):
                # create rules from each dna in the population
                rules = create_rules(population[i])

                # update grid for each training input example and calculate average fitness
                for j in range(nExamples):
                    
                    taskInput = np.array(taskTrain[task][j]['input'])
                    taskOutput = np.array(taskTrain[task][j]['output'])

                    
                    if not sameShape:
                        # "scale" matrix size to match output, n and m are scaling factors
                        n = taskOutput.shape[0]//taskInput.shape[0]
                        m = taskOutput.shape[0]//taskInput.shape[0]
                        taskInput = np.kron(taskInput,np.ones((n,m)))
  
                    
                    grid = taskInput
                                       
                    if step is not 0:
                        # update input with best rules for each steps
                        for s in range(step):
                            grid = arc.update_grid(grid,bestSteps[s])

                    grid = arc.update_grid(grid,rules)
                    
    
                    # calculate fitness after update steps
                    
                    fitness[i,j] = arc.calc_fitness(grid,taskInput,taskOutput)
                

                fitness[i,-1] = fitness[i,:-1].mean()

                # if we find the optimal solution (fitness score of 1)
                if (fitness[i,-1] >= best):
                    
                    best = fitness[i,-1]
                    bestSteps[step] = rules
                    
                    if best == 1:
                        
                        # find three optimal solutions before terminating
                        bestCount = bestCount + 1
                        
                        if bestCount == 1000:
                            fitMean[g] = fitness[:i+1,-1].mean()
                            fitMax[g] = 1

                            print('Optimal Solution Found for Task: ' + str(task))
                            arc.plot_evolve(taskFileNames[task],fitMean,fitMax)
                            arc.plot_solve(task, taskTrain, taskTest, sameShape, bestSteps, step + 1)
                            break
                        

            else:
                # if no break in inner loop
                
                if g is not (gens - 1):
                    sel = ga.selection(population,fitness[:,-1])
                    population = ga.reproduce(population,sel,geneLength,mutationRate)
                fitMean[g] = fitness[:,-1].mean()
                fitMax[g] = fitness[:,-1].max() 
                continue
            break
            
        # after the best rules for each step has been found
        # append function to bestSteps
        

        
        # kill off and replace bottom portion of population
        #population[len(population)//10:] = create_population(len(population)//10*9,nGenes)
        
        
        else:
            bestIndex = np.where(fitness[:,-1] == fitness[:,-1].max())[0][0]
            bestSteps[step] = create_rules(population[bestIndex])
            # if no break in inner loop
            arc.plot_evolve(taskFileNames[task], fitMean,fitMax)
            arc.plot_solve(task, taskTrain, taskTest, sameShape, bestSteps, step + 1)
            continue
        break
            
    # return best rules for each step
    return bestSteps


# In[ ]:


solutions = []

#blue and grey squares, blue and blue corner square, blue and red lines, blue and grey tetris
tasks = ['b60334d2.json','3aa6fb7a.json','a699fb00.json','3618c87e.json']

for i in range(len(tasks)):
    tasks[i] = np.where(np.array(taskFileNames) == tasks[i])[0][0]
    
for task in tasks:
    sameShape, sameColours = arc.check_task(task, taskTrain)
    
    popSize = 5000
    nGenes = 20
    gens = 30
    steps = 1
    geneLength = 12
    mutationRate = 1 

    try:
        print('Solving task: ' + taskFileNames[task])
        bestSteps = solve_task(task, taskTrain, taskTest, sameShape, nGenes, gens, steps, mutationRate)
        solutions.append([task, bestSteps])
    except:
        solutions.append([task,'Failed'])


# ### How about tasks that require multiple steps??

# In[ ]:


#task = ['db3e9e38.json'] #orange and blue triangle
task = ['a65b410d.json'] #b/r/g stairs
task = np.where(np.array(taskFileNames) == task)[0][0]
    
sameShape, sameColours = arc.check_task(task, taskTrain)
    
popSize = 5000
nGenes = 30
gens = 30
steps = 5
geneLength = 12
mutationRate = 2 


print('Solving task: ' + taskFileNames[task])
bestSteps = solve_task(task, taskTrain, taskTest, sameShape, nGenes, gens, steps, mutationRate)
solutions.append([task, bestSteps])

