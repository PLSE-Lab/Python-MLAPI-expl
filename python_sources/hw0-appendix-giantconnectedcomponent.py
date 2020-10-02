#!/usr/bin/env python
# coding: utf-8

# # What is about ? 
# 
# Examples for "giant connected component" phenomena are given here.
# https://en.wikipedia.org/wiki/Giant_component
# 
# It is dicussed in Lecture 2 slides 20 (MSN), 23 (PPI), 33 (Erdos-Renyi)  http://web.stanford.edu/class/cs224w/slides/02-gnp-smallworld.pdf
# 
# Giant connected component phenomena occurs for some random graphs like Erdos-Renyi and real life networks like web.
# It means that most of nodes of graph are connected with each other, and only small part of nodes falls into several much smaller by size connected components.
# 
# Here we will consider the simplest version of graph where each point has one outgoing edge.
# It is NOT Erdos-Renyi.  It is rather a model of the nearest-neigbour graph in degenerate situation where all points on the same distance, so one just randomly choose neigbour.
# 
# 
# Finally the example of real nearest neigbour graph is considered and there is NO giant component phenomena for it.
# 
# 
# Preliminary draft 
# Alexander Chervov July 2020
# 

# In[ ]:


try:
    import igraph # igraph is already preinstalled on kaggle, but not colab  
except:    
    get_ipython().system('pip install python-igraph # Pay attention: not just "pip install igraph" ')
    get_ipython().system('pip install cairocffi # Module required for plots ')
    import igraph # igraph is already preinstalled on kaggle, but not colab  

import numpy as np
import matplotlib.pyplot as plt
import time 

from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree


# # Example small n=45 and plot

# In[ ]:


import igraph
#n = 5242;  m = 14484
#n = 5 ; k = 2  
def create_toy_knn_graph(n,k):
  g = igraph.Graph( directed=True)#  G = snap.TUNGraph.New(n,m) # Allocate memory for UNdirected graph n-nodes, m-edges
  g.add_vertices(n)
  
  for i in range(n): # Add m edges connected at random 
    list_target_nodes = []
    while True: # technical loop - check we are not adding already existing edges 
      v2 = np.random.randint(0,n )
      if i == v2:         continue
      if v2 in list_target_nodes:   continue
      list_target_nodes.append(v2)
      g.add_edge(i,v2)
      if len(list_target_nodes) >= k: 
        break
  return g

n = 45 ; k = 1 
g =  create_toy_knn_graph(n,k)
h = g.degree_distribution(bin_width=1, mode = "out", )
#print(h)

print(" Random graph is created by the rule - each node has one random out-going edge to other node. As you can see from the plots typically the graph falls into one connected component or one big and 1-2 small" )

r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
list_components_sizes = [ len(t) for t in r  ]
print("Sizes of connected components")
print(list_components_sizes)
            
igraph.plot(g, bbox = (600,500))


# # Example statistics for large n = 1000, 10 000, 100 000
# 
# 
# Here are examples (each run generates different number but phenomena does not change) : 
# 
# n_nodes =  1000 Connected component sizes:
# [355, 79, 90, 359, 94, 10, 9, 4]
# 
# n_nodes =  10000 Connected component sizes:
# [9800, 197, 3]
# 
# n_nodes =  100000 Connected component sizes:
# [98714, 1246, 40]
# 
# we can see that always largest component size **is much much large than the second size**, especially for larger graphs
# 
# 

# In[ ]:



k = 1
for n in [1e1, 1e3,1e4,1e5]:
    n = int(n)
    t0 = time.time()
    g =  create_toy_knn_graph(n,k)
    r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
    print("n_nodes = ", n, "seconds passed", time.time()-t0,  "Connected component sizes:" )
    list_components_sizes = [ len(t) for t in r  ]
    print(list_components_sizes)
    print()


# 

# # Actual 1-NN graphs do NOT have "giant component" phenomena
# 
# Generate some data cloud - say  Gaussian d-dimensional sample,
# construct 1-Nearest neigbour graph
# (https://en.wikipedia.org/wiki/Nearest_neighbor_graph )
# 
# We will see NO giant component phenomenta, i.e. graph has many connected components, and the largest size is not big comparing with the others neigbour sizes, and quite small comparing with total node number
# 
# For example  for the case above we have seen largest component size  9800 for 10 000 nodes, but in simulation below we will see 288 for 10 000 nodes,
# 9800 vs 288 is strikind difference.
# 
# Actually number of connected components for nearest neigbour graphs grows approximately linearly with sample size, moreover coefficient of linear dependence is theoretically known
# (from 1997 ) for dimension 2. There is also theoretical proposal for higher dimensions https://cstheory.stackexchange.com/a/47039/2408 , but it is not yet fit with simulation results (June 2020).
# **
# 

# In[ ]:


dim = 100
        
t0 = time.time()    
c = 0        
for n in [1e4]:
    n = int(n)    
    X = np.random.randn(n, dim)
    print("Dimension",dim," n ", n)
    nbrs = NearestNeighbors(n_neighbors=2  ).fit(X) # 'ball_tree'
    distances, indices = nbrs.kneighbors(X)
    g = igraph.Graph( directed = True )
    g.add_vertices(range(n))
    g.add_edges(indices )
    r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
    list_components_sizes = [ len(t) for t in r  ]
    print("10 largest components sizes")
    print(np.sort(list_components_sizes)[::-1][:10])    
    print("Maximum size of the connected component")
    print(np.max(list_components_sizes)    )
    print(time.time() - t0 , "seconds passed" )
    plt.hist(list_components_sizes)
    plt.title("Histogram of components sizes")
    plt.show()


# # Plotting graph example with small number of points - 100 for NN graph, it is striking different from the model above

# In[ ]:


dim = 2
        
t0 = time.time()    
c = 0        
for n in [1e2]:
    n = int(n)    
    X = np.random.randn(n, dim)
    print("Dimension",dim," n ", n)
    nbrs = NearestNeighbors(n_neighbors=2  ).fit(X) # 'ball_tree'
    distances, indices = nbrs.kneighbors(X)
    g = igraph.Graph( directed = True )
    g.add_vertices(range(n))
    g.add_edges(indices )
    r = g.clusters(mode='WEAK') # Returns list of lists like [ [1,2],[3,4]] - means [1,2] - first connected comp., [3,4] - second , here 1,2,3,4 - nodes ids
    list_components_sizes = [ len(t) for t in r  ]
    print("10 largest components sizes")
    print(np.sort(list_components_sizes)[::-1][:10])    
    print("Maximum size of the connected component")
    print(np.max(list_components_sizes)    )
    print(time.time() - t0 , "seconds passed" )
    plt.hist(list_components_sizes)
    plt.title("Histogram of components sizes")
    plt.show()
    
    
igraph.plot(g)    


# In[ ]:




