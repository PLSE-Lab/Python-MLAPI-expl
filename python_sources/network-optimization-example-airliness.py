#!/usr/bin/env python
# coding: utf-8

# In[15]:


# import the libraries & dataset
import pandas as pd 
data = pd.read_excel('../input/dataset_airline_optimization_problem.xlsx') 
data.head()


# ## Formulate the Network

# In[16]:


import networkx as nx
FG = nx.from_pandas_edgelist(data, source='origin', target='dest',edge_attr=True,)
FG.nodes()
FG.edges()
nx.draw_networkx(FG, with_labels=True,node_size=600, node_color='gray') 
nx.algorithms.degree_centrality(FG) 
nx.density(FG) 
nx.average_shortest_path_length(FG)
nx.average_degree_connectivity(FG) 


# ## Shortest path between JFK to DFW

# In[17]:


# Let us find the dijkstra path from JAX to DFW.
dijpath = nx.dijkstra_path(FG, source='JAX', target='DFW')
print('Dijkstra path: ', dijpath)

# Let us try to find the dijkstra path weighted by airtime (approximate case)
shortpath = nx.dijkstra_path(FG, source='JAX', target='DFW', weight='air_time')
print('dijkstra path weighted by airtime:', shortpath)

# nx.draw(FG)
# Let us find all the paths available
for path in nx.all_simple_paths(FG, source='JAX', target='DFW'):
  #print(path)
  pass
# You can read more in-depth on how dijkstra works from this resource - https://courses.csail.mit.edu/6.006/fall11/lectures/lecture16.pdf
#Note:  this code is inspired by www.analyticsvidhya.com /blog/2018/04/introduction-to-graph-theory-network-analysis-python

