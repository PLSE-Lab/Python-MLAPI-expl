#!/usr/bin/env python
# coding: utf-8

# # Read in group-to-group graph data
# 
# Graphing is pretty straight-forward, but for large graphs can become very code-intensive if you want to make an accessible representation of your data. Frequently this is done through choosing an appropriate layout, changing colors and lightening edges. 
# 
# We'll walk through some basic examples of this here.

# In[ ]:


# First, import the important packages
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

# Next read in the edges and create a graph
df = pd.read_csv('../input/group-edges.csv')
g = nx.from_pandas_edgelist(df, 
                            source='group1', 
                            target='group2', 
                            edge_attr='weight')

print('The member graph has {} nodes and {} edges.'.format(len(g.nodes),
                                                          len(g.edges)))


# # Default graph

# In[ ]:


pos = nx.spring_layout(g)
nx.draw_networkx(g, pos)


# # Plot different layouts

# In[ ]:


# Circular Layout
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.circular_layout(g)
nx.draw_networkx_nodes(g, pos, node_size=10, 
                       node_color='xkcd:muted blue')
nx.draw_networkx_edges(g, pos, alpha=0.05)

ax.axis('off')
plt.show()


# In[ ]:


# Random Layout
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.random_layout(g)
nx.draw_networkx_nodes(g, pos, node_size=50,
                      node_color='xkcd:muted green')
nx.draw_networkx_edges(g, pos, alpha=0.05)

ax.axis('off')
plt.show()


# In[ ]:


# Spring Layout
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.spring_layout(g, k=2)
nx.draw_networkx_nodes(g, pos, node_size=50,
                      node_color='xkcd:muted purple')
nx.draw_networkx_edges(g, pos, alpha=0.03)

ax.axis('off')
plt.show()


# # Varying colors, transparency and edge widths
# 
# To create node- or edge-variable plots, you need to create a list of values that match up to the nodes being plotted. A simple way to ensure you are getting data in the appropriate order is to use a list comprehension, pulling from either `g.nodes` or `g.edges`. Flagging `data=True` in either of these methods will return a dictionary of attributes in addition to the node or edge index.

# In[ ]:


# Weight nodes by degree and edge size by width 
fig, ax = plt.subplots(1,1, figsize=(8,8), dpi=150)

pos = nx.spring_layout(g, k=2)
node_sizes = [g.degree[u] for u in g.nodes]
nx.draw_networkx_nodes(g, pos, node_size=node_sizes,
                       node_color='xkcd:muted purple')

edge_widths = [d['weight'] for u,v,d in g.edges(data=True)]
nx.draw_networkx_edges(g, pos, width=edge_widths, alpha=0.03)

ax.axis('off')
plt.show()


# In[ ]:




