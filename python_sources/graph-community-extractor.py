#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from pathlib import Path
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors
from collections import Counter
import networkx as nx


# In this kernel, I will be using graph analysis (**netwrokx**) to extract ARC grid's objects as communities. This allows extremely precise and relatively fast object extraction from the tasks. 

# # A tiny recall on graph theory

# ...TODO

# In[ ]:





# # Utils

# ## Arcgrah

# In[ ]:


class ArcGraph():
    def __init__(self, diag = True):
        self.G = None
        self.diag = diag
        
        
        
    def to_graph(self, im):
        G = nx.Graph()
        I,J = im.shape
        for i in range(I):
            for j in range(J):
                if not im[i,j]:
                    continue
                G.add_node((i,j))
                edges = []
                if i >= 1:
                    if im[i,j] == im[i-1,j]:
                        edges.append( ( (i,j), (i-1,j) ) )
                    if j >= 1:
                        if im[i,j] == im[i,j-1]:
                            edges.append( ( (i,j), (i,j-1) ) )
                        if im[i,j] == im[i-1,j-1] and self.diag:
                            edges.append( ( (i,j), (i-1,j-1) ) )
                    if j < J-1:
                        if im[i,j] == im[i,j+1]:
                            edges.append( ( (i,j), (i,j+1) ) )
                        if im[i,j] == im[i-1,j+1] and self.diag:
                            edges.append( ( (i,j), (i-1,j+1) ) )
                
                if i < I-1:
                    if im[i,j] == im[i+1,j]:
                        edges.append( ( (i,j), (i+1,j) ) )
                    if j >= 1:
                        if im[i,j] == im[i+1,j-1] and self.diag:
                            edges.append( ( (i,j), (i+1,j-1) ) )
                    if j < J-1:
                        if im[i,j] == im[i+1,j+1] and self.diag:
                            edges.append( ( (i,j), (i+1,j+1) ) )
                G.add_edges_from(edges)
        self.G = G
        return self.G


# ## Plotting

# In[ ]:


def plot_one(ax, i,train_or_test,input_or_output, task):
#     cmap = colors.ListedColormap(
#         ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
#          '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
#     norm = colors.Normalize(vmin=0, vmax=9)
    
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
        plot_one(axs[0,i],i,'train','input', task=task)
        plot_one(axs[1,i],i,'train','output', task=task)        
    plt.tight_layout()
    plt.show()        
        
#     num_test = len(task['test'])
#     fig, axs = plt.subplots(2, num_test, figsize=(3*num_test,3*2))
#     if num_test==1: 
#         plot_one(axs[0],0,'test','input')
#         plot_one(axs[1],0,'test','output')     
#     else:
#         for i in range(num_test):      
#             plot_one(axs[0,i],i,'test','input')
#             plot_one(axs[1,i],i,'test','output')  
    plt.tight_layout()
    plt.show() 


# # Object detection as graph community analylis

# In[ ]:


cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)


# In[ ]:


data_path = Path('../input/abstraction-and-reasoning-challenge')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'


# In[ ]:


def read_task(task_file, train = True):
    task_file = (training_path if train else evaluation_path)/task_file
    with task_file.open() as f:
        return json.load(f)


# In[ ]:


task = read_task("025d127b.json")
plot_task(task)


# > The task **025d127b** contains  2 objects. If we could extract them, then their colors and positions can be inferred based on additional statistics.

# In[ ]:


task_in = np.array(task["train"][0]["input"], dtype = np.uint8) 


# In[ ]:


arc_graph =  ArcGraph(diag = True) # Make the grah builder
graph = arc_graph.to_graph(task_in) # Convert the image into Networkx graph, 
                                 # two arbitrary cells are linked if they share the same color and are close to each other on the grid


# In[ ]:


nx.draw(graph, pos= nx.spring_layout(graph))


# > By looking at the graph's plot, we could easily identify the 2 objects. Those objects could be extracted using [networkx](https://networkx.github.io/) and its  [k-clique-communities algorithm](https://networkx.github.io/documentation/stable/reference/algorithms/generated/networkx.algorithms.community.kclique.k_clique_communities.html#networkx.algorithms.community.kclique.k_clique_communities). Note that we could also extract those objects as the graph's [connected components](https://networkx.github.io/documentation/stable/reference/algorithms/component.html#connectivity) .
# 
# > Let's recall that the graph consists of the task grid cells. Two arbitrary celles $(i_1,j_1), (i_2,j_2)$ are considered as linked when they are neighbour and does share the same color (which must be different from  the backgroud color).

# In[ ]:


communities = list(nx.community.k_clique_communities(graph,2 ))
len(communities)


# As expected, 2 communties (objects) are found :) ! Let's plot them...

# In[ ]:


plt.figure(figsize=(12,6))
plt.subplot("131")
plt.imshow(task_in, cmap = cmap, norm=norm)
plt.title("Main task")
for k,community in enumerate(communities, 1):
    im = np.zeros(task_in.shape, dtype=int) # A zeros filled image
    # The clique is assigned a rank k
    plt.subplot(f"13{k+1}")
    plt.title(f"Object {k}")
    for i,j in community:
        im[i,j] = k
    plt.imshow(im, cmap=cmap, norm = norm)


# In[ ]:





# # Extract more objects

# In[ ]:


task_files =np.array( list(training_path.glob("*")))
np.random.shuffle(task_files)
NPLOTS = 30
plot_count = 0
for task_file in task_files:
    
    task = read_task(task_file.name)
    task_in = np.array(task["train"][0]["input"], dtype = np.uint8)
    
    graph = arc_graph.to_graph(task_in)
    
    communities = sorted(nx.community.k_clique_communities(graph,2 ), key= lambda x: -len(x))
    if  len(communities) < 3: # Keep only interesting figures
        continue
    
    n = min(7, len(communities)) # Only plot the first objects
    fig, ax = plt.subplots(1,n+1, squeeze=False, figsize=(12,6))
    ax[0,0].imshow(task_in, cmap=cmap, norm=norm)
    ax[0,0].set_title("Main task")
    for k,community in enumerate(communities[:n] , 1):
        im = np.zeros(task_in.shape, dtype=int)# A zeros filled image
        for i,j in community:
            im[i,j] = 1
        ax[0,k].imshow(im, cmap=cmap, norm = norm)
        ax[0,k].set_title(f"Object {k}")
    plt.show()
    plot_count += 1
    if plot_count > NPLOTS:
        break
    print(task_file.name,flush=True)


# In[ ]:




