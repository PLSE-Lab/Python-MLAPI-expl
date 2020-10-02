#!/usr/bin/env python
# coding: utf-8

# # What is about
# 
# Noteboook shows some examples of modularity calculation for different graphs.
# Hope  it might help to give some intution what modularity is, as well, exercise oneself with networkx and igraph packages.
# 
# 
# It does NOT correspond to HW1 task, just complement. 
# 
# 
# Draft version "under construction", June 2020 AC.
# 
# 

# # Modularity with networkx 

# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# ## Simplest graph - just two nodes with one edge

# In[ ]:


G = nx.Graph()
G.add_nodes_from([0, 1])
G.add_edge(0, 1)
# G.add_node(1)
fig = plt.figure(figsize= (8,2))
plt.subplot(121)
pos = {0: [0,0],1:[1,0]}
nx.draw(G, pos,  node_color= [0,1 ])
m1 = nx.algorithms.community.modularity(G, [{0}, {1}])
str1 = 'Partition into two subgroups \n'+ 'Modularity {}'.format(m1)
plt.title(str1)
plt.subplot(122)
pos = {0: [0,0],1:[1,0]}
nx.draw(G, pos,  node_color= [0,0 ])
m1 = nx.algorithms.community.modularity(G, [{0,1}])
str1 = 'Partition - all nodes in one group \n'+ 'Modularity {}'.format(m1)
plt.title(str1)

plt.show()


# ## Simplest graph - three  nodes in line 

# In[ ]:


G = nx.Graph()
# G.add_node(1)
G.add_nodes_from([0, 1,2])
#G.add_edge(0, 1)
G.add_edges_from([(0, 1), (1, 2)])
fig = plt.figure(figsize= (12,3))
plt.subplot(131)
pos = {0: [0,0],1:[1,0],2:[2,0]}
nx.draw(G, pos,  node_color= [0,0,0 ])
m1 = nx.algorithms.community.modularity(G, [{0,1,2}])
str1 = 'Partion: \n all nodes in one group \n'+ 'Modularity {}'.format(m1)
plt.title(str1)

plt.subplot(132)
pos = {0: [0,0],1:[1,0],2:[2,0]}
nx.draw(G, pos,  node_color= [0,1,2 ])
m1 = nx.algorithms.community.modularity(G, [{0},{1},{2}])
str1 = 'Partition: \n all nodes in different groups \n'+ 'Modularity {}'.format(m1)
plt.title(str1)

plt.subplot(133)
pos = {0: [0,0],1:[1,0],2:[2,0]}
nx.draw(G, pos,  node_color= [0,0,1 ])
m1 = nx.algorithms.community.modularity(G, [{0,1},{2}])
str1 = 'Partion: \n Group1 - [0,1], Group2 - [2]  \n'+ 'Modularity {}'.format(m1)
plt.title(str1)

plt.show()


# ## Simplest graph - Triangle

# In[ ]:


G = nx.Graph()
# G.add_node(1)
G.add_nodes_from([0, 1,2])
#G.add_edge(0, 1)
G.add_edges_from([(0, 1), (1, 2), (0, 2)])
fig = plt.figure(figsize= (12,3))
plt.subplot(131)
nx.draw(G,   node_color= [0,0,0 ])
m1 = nx.algorithms.community.modularity(G, [{0,1,2}])
str1 = 'Partion: \n all nodes in one group \n'+ 'Modularity ' + str(np.round(m1,3) )
plt.title(str1)

plt.subplot(132)
nx.draw(G,   node_color= [0,1,2 ])
m1 = nx.algorithms.community.modularity(G, [{0},{1},{2}])
str1 = 'Partion: \n all nodes in different groups \n'+ 'Modularity ' + str(np.round(m1,3) )
plt.title(str1)

plt.subplot(133)
nx.draw(G,   node_color= [0,0,2 ])
m1 = nx.algorithms.community.modularity(G, [{0,1},{2}])
str1 = 'Partion: \n  Group1 - [0,1], Group2 - [2] \n'+ 'Modularity ' + str(np.round(m1,3) )
plt.title(str1)

plt.show()


# ## Complete graph on 5 nodes
# 
# Explore different partitions and observe that modularity behaves as intuitevely expected:
# 
# 1) Largest modularity is zero - all nodes in the same group, all other partitions will lead to negative modularity - that is expected, because complete graph is the most connected graph, so splitting it to subgroups, should decrease any reasonable measure of grouping perforamce 
# 
# 2) the smallest value of modularity is when all nodes in different groups - 
# again it is expected by similar reason as above. 
# Actual value = 1/nodes_count - holds for complete graphs. 
# 
# 3) The bigger partition differs from - all in one group the smaller modularity would be 
# 

# In[ ]:


G = nx.complete_graph(5)

fig = plt.figure(figsize= (18,3))
plt.subplot(151)
nx.draw(G,   node_color= np.zeros( G.number_of_nodes() )    )
m1 = nx.algorithms.community.modularity(G, [range( G.number_of_nodes() )])
str1 = 'Partion: \n all nodes in one group \n'+ 'Modularity ' + str(np.round(m1,3) )
plt.title(str1)


plt.subplot(152)
nx.draw(G,   node_color= range( G.number_of_nodes() ) )
m1 = nx.algorithms.community.modularity(G, [ {t} for t in  range( G.number_of_nodes()) ] )
str1 = 'Partion: \n all nodes in different groups \n'+ 'Modularity ' + str(np.round(m1,3)) + '=1/node_number' 
plt.title(str1)

plt.subplot(153)
v = np.zeros( G.number_of_nodes() )
v[0] = 1
nx.draw(G,   node_color= v  )
m1 = nx.algorithms.community.modularity(G, [{0}] + [  range(1, G.number_of_nodes() ) ] )
str1 = 'Partion: \n Group1 = [0], Group2 = [1,2,3,4] \n'+ 'Modularity ' + str(np.round(m1,3)) 
plt.title(str1)

plt.subplot(154)
v = np.zeros( G.number_of_nodes() )
v[0] = 1; v[1] = 1
nx.draw(G,   node_color= v  )
m1 = nx.algorithms.community.modularity(G, [{0,1}] + [  range(2, G.number_of_nodes() ) ] )
str1 = 'Partion: \n Group1 = [0,1], Group2 = [2,3,4] \n'+ 'Modularity ' + str(np.round(m1,3)) 
plt.title(str1)

plt.subplot(155)
v = np.zeros( G.number_of_nodes() )
v[0] = 1; v[1] = 2
nx.draw(G,   node_color= v  )
m1 = nx.algorithms.community.modularity(G, [{0},{1}] + [  range(2, G.number_of_nodes() ) ] )
str1 = 'Partion: \n Group1 = [0,1], Group2 = [2,3,4] \n'+ 'Modularity ' + str(np.round(m1,3)) 
plt.title(str1)

plt.show()


# # Modularity with "igraph"
# 
# Package "igraph" has built-in functions to calculate modularity, and clustering algorithms like Louvain, Leiden, optimal_clustering (with respect to modularity - works for small graphs only), and many many other graph clustering algorithms
# 
# https://igraph.org/python/doc/igraph.Graph-class.html#modularity  - modularity function
# https://igraph.org/python/doc/igraph.Graph-class.html#community_optimal_modularity - optimal clustering 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel - Louvain clustering 
# 
# 
# 
# Plots of igraph are not with matplotlib backend, so it is not clear how to create subplots, add titles, so on - which can be easy done by matplotlib. 
# So it is not clear how to make pictures like above with igraph 

# In[ ]:


kaggle_env = 1
if kaggle_env != 1: # On kaggle igraph is pre-installed 
    get_ipython().system('pip install python-igraph # Pay attention: not just "pip install igraph" ')
    get_ipython().system('pip install cairocffi # Module required for plots ')
import igraph # On kaggle it is pre-installed 
import numpy as np


# In[ ]:


g = igraph.Graph(directed=False)
g.add_vertices(3)
g.add_edges([[0,1],[1,2],[2,0]] )
print('Modularity for all nodes in different groups partition:', g.modularity([0,1,2]) ) # [0,1,2] - membership list - i.e. defines group number for i-th node. For example [0,0,0] - all nodes in Group0, [0,1,2] - all nodes in different groups.
    # It is different from the networkx interface , where we define groups like: [ {0},{1},{2}] - each group is sublist 
    # nx.algorithms.community.modularity(G, [{0},{1},{2}])
print('It is 1/3 - general pattern  for any complete graph - 1/nodes_count')

visual_style = {}
visual_style["vertex_color"] = ['red', 'blue', 'green']
visual_style["vertex_label"] = range(g.vcount()) 

igraph.plot(g, **visual_style,  bbox=(100,100))


# ## Louvain clustering and optimal clustering by igraph
# 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_optimal_modularity
# 
# https://stackoverflow.com/questions/24514297/igraph-community-optimal-modularity-takes-no-weight-argument?rq=1
# 
# Clearly on some simple examples Louvain clustering algorithm would produce optimal clustering (i.e. with the top possible modularity). 
# Package igraph has  method to find optimal clustering, but it would be slow in general and applicable only in graph with not big number of nodes 
# 
# https://stackoverflow.com/a/60930979/625396
# Example to use Louvain in 3 packages - networkx, igraph, bct
# 

# In[ ]:


g = igraph.Graph()
g.add_vertices(16)
nodes = np.array([0,1,4,5])
for k in [0,2,8,10]:#,2,4,6,8]:
  for i in nodes+k:
    for j in nodes+k:
      if i<=j: continue 
      g.add_edge(i, j)
g.add_edge(1, 2)
g.add_edge(4, 8)
g.add_edge(13, 14)
g.add_edge(7, 11)

########################################################################
# Cluster by Louvain algorithm 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel
########################################################################
louvain_partition = g.community_multilevel()# weights=graph.es['weight'], return_levels=False)
modularity1 = g.modularity(louvain_partition)#, weights=graph.es['weight'])
print("The modularity for igraph-Louvain partition is {}".format(modularity1))
#print();
print('Partition info:')
print(louvain_partition)

########################################################################
# Cluster by optimal algorithm (applicable only for small graphs <100 nodes), it would be very slow otherwise 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_optimal_modularity
########################################################################
print();
v = g.community_optimal_modularity() # weights= gra.es["weight"]) 
modularity1 = g.modularity(v)#, weights=graph.es['weight'])
print("The modularity for igraph-optimal partition is {}".format(modularity1))
#print();
print('Partition info:')
print(v) 

########################################################################
# Plot graph 
########################################################################
layout = g.layout_grid( ) # reingold_tilford(root=[2])
visual_style = {} 
dict_colors = {0:'Aqua', 1:'Aqua', 4:'Aqua', 5:'Aqua',2:'Aquamarine', 3:'Aquamarine', 6:'Aquamarine', 7:'Aquamarine',
               8:'Crimson', 9:'Crimson', 12:'Crimson', 13:'Crimson',10:'Goldenrod', 11:'Goldenrod', 14:'Goldenrod', 15:'Goldenrod',
               } # https://en.wikipedia.org/wiki/X11_color_names - colors by names supported by igraph 
visual_style["vertex_color"] = [dict_colors[k]  for k in range(g.vcount() )]
visual_style["vertex_label"] = range(g.vcount()) 
igraph.plot(g, layout = layout, **visual_style, bbox = (200,200) )


# ## Example of triangle+self loop

# In[ ]:


g = igraph.Graph(directed=False)
g.add_vertices(3)
g.add_edges([[0,0],[0,1],[1,2],[2,0]] )

louvain_partition = g.community_multilevel()# weights=graph.es['weight'], return_levels=False)
modularity1 = g.modularity(louvain_partition)#, weights=graph.es['weight'])
print("The modularity for igraph-Louvain partition is {}".format(modularity1))
#print();
print('Partition info:')
print(louvain_partition)

modularity1 = g.modularity([0,1,2])#, weights=graph.es['weight'])
print("The modularity for all nodes in different groups partition is {}".format(modularity1))

modularity1 = g.modularity([0,0,0])#, weights=graph.es['weight'])
print("The modularity for all nodes in one groups partition is {}".format(modularity1))

modularity1 = g.modularity([0,1,1])#, weights=graph.es['weight'])
print("The modularity for all 0,1,1 partition is {}".format(modularity1))


igraph.plot(g,bbox = [200,100])


# # Coding modularity by hands
# 
# Explicit example of calculation by hands and coding by hands modularity can be found in notebook:
# https://www.kaggle.com/alexandervc/hw1-part3-modularity-and-louvain-by-ac#3.1-Modularity-gain-when-an-isolated-node-moves-into-a-community-[4-points]
# 
# Subsections "Calculations by hands" and "Code to calculate modularity"
# 
# 

# In[ ]:




