#!/usr/bin/env python
# coding: utf-8

# (work continuously in progress)
# 
# forked from https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines
# 
# # What is the ARC algorithm in our brains?
# 
# The goal here is to get some insight into how the core knowledge priors, which are supposed to be enough to solve all puzzles, can be combined and chained together to describe in some sort of "cognitive pseudocode" how these puzzles are solved by people.
# 
# In a way, how are the tasks of https://www.kaggle.com/davidbnn92/task-tagging to be chained together to solve the puzzles? Also, how do these task tags relate to the core knowledge priors mentioned in the ARC paper?
# 
# Perhaps this will give some hints on how to build a cognitive architecture that can perform *abstraction* and *reasoning* and solve the ARC puzzles
# 
# 
# # Core Knowledge Priors
# 
# 
# ## Objectness Priors
# 
# * object cohesion
# * object persistence
# * object influence via contact
# 
# ## Goal-Directedness Prior
# 
# ## Numbers and Counting Priors
# 
# * counting objects
# * sorting objects
# * comparing numbers (in ARC relevant quantities are at most ~10)
# * comparing (object) size
# * pattern repetition for N times
# * addition
# * subtraction
# 
# ## Basic Geometry and Topology Priors
# 
# * lines, rectangular shapes (regular shapes are more likely than complex ones)
# * symmetries, rotations, translations
# * shape upscaling or downscaling, elastic distortions
# * containing / being contained / being inside or outside of a perimeter
# * drawing lines, connecting points, orthogonal projections
# * copying / repeating objects
# 

# # Task Tagging Skills
# 
# Here I'm trying to cluster the skills discovered in https://www.kaggle.com/davidbnn92/task-tagging and match them to the core priors. Some where renamed for better matching.
# 
# ## Objectness Prior
# 
# * detect grid
# * detect hor lines
# * detect wall
# * detect closed curves
# * detect background color
# * shape guessing
# * size guessing
# * separate image
# * separate images
# * separate shapes
# * remove noise
# * rectangle guessing
# * extrapolate image from grid
# 
# 
# ## Goal-Directedness Prior
# 
# 
# ## Numbers and Counting Priors
# 
# * algebra
# * take complement
# * count different colors
# * count hor lines
# * count patterns
# * count tiles
# * count ver lines
# * take maximum
# * take minimum
# * take negative
# * divide by n
# * dominant color
# * order numbers
# * measure area
# * measure distance from side
# * measure length
# 
# ## Basic Geometry and Topology Priors
# 
# * adapt image to grid
# * create grid
# 
# * proximity guessing
# 
# * even or odd
# 
# * detect repetition
# * detect symmetry
# * diagonal symmetry
# 
# * detect connectedness
# * image within image
# * detect enclosure
# * out of boundary
# 
# * holes
# * spacing
# 
# * enlarge image
# 
# * diagonals
# 
# * inside out
# 
# 
# ## Others / Associations
# 
# * associate color to bools
# * associate colors to bools
# * associate colors to colors
# * associate colors to colors (color matching)
# * associate colors to images
# * associate colors to numbers
# * associate colors to patterns
# * associate colors to ranks
# * associate images to bools
# * associate images to images
# * associate images to numbers
# * associate images to patterns
# * associate patterns to colors
# * associate patterns to patterns
# 
# 
# ## Others / Comparisons
# 
# * compare image
# * pairwise analogy
# 
# 
# ## Others / Image and Pattern Transformations
# 
# * pattern alignment
# * pattern coloring
# * pattern completion
# * pattern deconstruction
# * pattern differences
# * pattern expansion
# * pattern intersection
# * pattern juxtaposition
# * pattern mimicking
# * pattern modification
# * pattern moving
# * pattern moving (bring patterns close)
# * pattern moving (gravity)
# * pattern reflection
# * pattern repetition
# * pattern repetition (diagonals)
# * pattern repetition (recoloring)
# * pattern repetition (fractals)
# * pattern resizing
# * pattern rotation
# * pattern replacement
# 
# * image expansion
# * image filling
# * image juxtaposition
# * image reflection
# * image repetition
# * image resizing
# * image rotation
# 
# * crop
# 
# 
# ## Others / Drawing
# 
# * draw line from border
# * draw line from point
# * draw pattern from point
# * draw rectangle
# * connect the dots
# 
# 
# ## Others / Coloring
# 
# * color permutation
# * recoloring
# * grid coloring
# * background filling
# * loop filling
# * contouring
# 
# 
# ## Others / Various
# 
# * color guessing
# * color palette
# 
# * concentric
# 
# * projection unto rectangle
# 
# * create image from info
# 
# * direction guessing crop
# * direction guessing
# 
# * ex nihilo
# * find the intruder
# * jigsaw
# * maze
# * portals
# * summarize

# In[ ]:


# import numpy as np
# import pandas as pd

# import os
# import json
# from pathlib import Path

# import matplotlib.pyplot as plt
# from matplotlib import colors
# import numpy as np


# In[ ]:


# from pathlib import Path

# data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
# training_path = data_path / 'training'
# evaluation_path = data_path / 'evaluation'
# test_path = data_path / 'test'

# training_tasks = sorted(os.listdir(training_path))
# evaluation_tasks = sorted(os.listdir(evaluation_path))


# In[ ]:


# def plot_one(ax, i,train_or_test,input_or_output):
#     cmap = colors.ListedColormap(
#         ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
#          '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
#     norm = colors.Normalize(vmin=0, vmax=9)
    
#     input_matrix = task[train_or_test][i][input_or_output]
#     ax.imshow(input_matrix, cmap=cmap, norm=norm)
#     ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
#     ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
#     ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
#     ax.set_xticklabels([])
#     ax.set_yticklabels([])
#     ax.set_title(train_or_test + ' '+input_or_output)
    

# def plot_task(task):
#     """
#     Plots the first train and test pairs of a specified task,
#     using same color scheme as the ARC app
#     """    
#     num_train = len(task['train'])
#     fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
#     for i in range(num_train):     
#         plot_one(axs[0,i],i,'train','input')
#         plot_one(axs[1,i],i,'train','output')        
#     plt.tight_layout()
#     plt.show()        
        
#     num_test = len(task['test'])
#     fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
#     if num_test==1: 
#         plot_one(axs[0],0,'test','input')
#         plot_one(axs[1],0,'test','output')     
#     else:
#         for i in range(num_test):      
#             plot_one(axs[0,i],i,'test','input')
#             plot_one(axs[1,i],i,'test','output')  
#     plt.tight_layout()
#     plt.show() 


# ## training set

# In[ ]:


# for i in range(len(training_tasks)):
#     task_file = str(training_path / training_tasks[i])
    
#     with open(task_file, 'r') as f:
#         task = json.load(f)
        
#     print(i)
#     print(training_tasks[i])
#     plot_task(task)


# ## evaluation set

# In[ ]:


# for i in range(len(evaluation_tasks)):
#     task_file = str(evaluation_path / evaluation_tasks[i])
    
#     with open(task_file, 'r') as f:
#         task = json.load(f)

#     print(i)
#     print(evaluation_tasks[i])
#     plot_task(task)

