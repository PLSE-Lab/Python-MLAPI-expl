#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# Many of us working on the ARC challenge have found that good results on the training and evaluation sets do not necessarily translate to the private test set. It is stated that the private test set contains the same type of problems as the public examples. However the private test set is probably just a little more complicated in some way. Testing our algorithms to check they correctly implement the required generality is hard without access to the test data. So, I got creative and generated some hopefully harder tasks based on the same core knowledge priors as the public ARC data.
# 
# With code taken from https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines, this notebook visualizes the new tasks I have created for testing.

# In[ ]:


import numpy as np
import pandas as pd

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np


# In[ ]:


# From: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines

def plot_one(task, ax, i,train_or_test, input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    
    input_matrix = task[train_or_test][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(train_or_test + ' '+input_or_output)
    

def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3*num_train,3*2))
    for i in range(num_train):     
        plot_one(task, axs[0,i],i,'train','input')
        plot_one(task, axs[1,i],i,'train','output')        
    plt.tight_layout()
    plt.show()        
        
    num_test = len(task['test'])
    fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
    if num_test==1: 
        plot_one(task, axs[0],0,'test','input')
        plot_one(task, axs[1],0,'test','output')     
    else:
        for i in range(num_test):      
            plot_one(task, axs[0,i],i,'test','input')
            plot_one(task, axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show() 


# In[ ]:


extra_tasks_path = Path('/kaggle/input/extra-arc-tasks-for-testing/')

def show_extra_task(task_name):
    task_file = str(extra_tasks_path / (task_name + ".json"))    
    with open(task_file, 'r') as f:
        task = json.load(f)
    plot_task(task)


# ### Task 01
# 
# A simple repeats tasks, but where the number of repeats depends on the count of non-zero values in the input.

# In[ ]:


show_extra_task("new_task_01")


# ### Task 02
# 
# Another repeats tasks. This time the number of repeats is *(count of blue)* horizontally and *(count of red)* vertically.

# In[ ]:


show_extra_task("new_task_02")


# ### Task 03
# 
# Extracting panels in this task is made harder by noise. Once extracted, the most common panel is selected, then flipped left-right to make the output.

# In[ ]:


show_extra_task("new_task_03")


# ### Task 04
# 
# Split the input into 3x3 panels. Apply a logical or to the two panels with the highest non-zero value count, then apply the colour from the panel with the lowest non-zero value count. Note that the first train example is particularly ambiguous and could be solved other ways, so this task relies on the second training example.

# In[ ]:


show_extra_task("new_task_04")


# ### Task 05
# 
# Extract panels in a 3x1 grid. Select the panel that has up-down symmetry and then flip it left-right. A still more complex example might require the solver to check the direction of the symmetry and then flip in the other direction.

# In[ ]:


show_extra_task("new_task_05")

