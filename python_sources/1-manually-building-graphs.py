#!/usr/bin/env python
# coding: utf-8

# In this notebook, we're going to build a very simple graph manually using `networkx`. This is not the most efficient way to import graphs into `networkx`, but it's a decent teaching exercise (and won't take long)!
# 
# First, we initialize the Graph object and add some nodes and edges to it.

# In[ ]:


import networkx as nx

g = nx.Graph()
g


# In[ ]:


# Add nodes
g.add_node('A')
g.add_node('B')
g.add_node('C')
g.add_node('D')

print(g.nodes)


# In[ ]:


# Add edges
g.add_edge(u='A', v='B')
g.add_edge('A', 'D')
g.add_edge('B', 'D')
g.add_edge('B', 'C')
g.add_edge('C', 'D')

# Or, you could use:
# g.add_edges_from([('A','B'), ('A', 'D'), ...])

print(g.edges)


# Next, we want to some edge weights. These can be added individually -- e.g. `g['A']['B']['weight'] = 1` -- or they can be added all at once, using a dictionary of `(u,v): value` pairs. 

# In[ ]:


# Add edge weights
edge_weights = {('A', 'B'): 1, ('A', 'D'): 2, 
                ('B', 'D'): 3, ('B', 'C'): 4 , 
                ('C', 'D'): 5} 
nx.set_edge_attributes(g, edge_weights, 'weight')

print(g.edges(data=True))


# #### Accessing data via dictionary-style indexing
# 
# Graph data is stored in NetworkX in a "dict-of-dict-of-dict" format. Different methods, such as `g.nodes` and `g.degree` return special views of the data, but it can also be accessed via more traditional dictionary key indexing.

# In[ ]:


g['A']


# In[ ]:


g['A']['B']


# In[ ]:


g['A']['B']['weight']


# In[ ]:


# You can also use dict/list-comps
[g[u][v]['weight'] for u,v in g.edges]


# #### Basic plotting
# At it's simplest, graphs require a "position" dictionary which describes where to put each node on the canvas. These can be generated via several layout commands, including `nx.circular_layout`, `nx.random_layout` and `nx.spring_layout`. 
# 
# You can plot the entire graph (nodes, edges, labels) with the `nx.draw_networkx` function, or you can split each class of objects into their own function for more precise control.

# In[ ]:


# Simplest possible graph
pos = nx.circular_layout(g)
nx.draw_networkx(g, pos)


# In[ ]:


# Graph plotting edges separately and by weight
pos = nx.random_layout(g)
nx.draw_networkx_nodes(g, pos)

edge_widths = [d['weight'] for u,v,d in g.edges(data=True)]
nx.draw_networkx_edges(g, pos, width=edge_widths)


# #### Create a new graph from an edge list

# In[ ]:


edge_list = [(x,y) for x in 'ABCDEFKL' for y in 'DAHLOUVW']
g2 = nx.from_edgelist(edge_list)

print('Nodes: ', g2.nodes)
print('Edges: ', g2.edges)


# In[ ]:


pos = nx.circular_layout(g2)
nx.draw_networkx(g2, pos)

