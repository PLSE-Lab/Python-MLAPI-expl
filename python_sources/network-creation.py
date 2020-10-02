#!/usr/bin/env python
# coding: utf-8

# # Import

# In[ ]:


import networkx as nx
import pandas as pd


# # Path definition

# In[ ]:


pathFile = "../input/reddit-ut3-ut1/comments_students.csv"


# # Input data loading

# In[ ]:


df = pd.read_csv(pathFile)


# In[ ]:


df.head()


# # Empty directed network creation

# In[ ]:


g= nx.DiGraph()


# # Add nodes

# We first start by the creation on "link" nodes (those in the link_id column) and we add an attribute to type this node

# In[ ]:


g.add_nodes_from(df.link_id, type="link")


# We first start by the creation on "comment" nodes (those in the name column) and we add an attribute to type this node

# In[ ]:


g.add_nodes_from(df.name, type="comment")


# # Add edges

# The idea here is to consider every pairs (name, parent_id) in the dataframe and create one edge per pair in the dataframe. This is done using the `add_edges_from` function and specifying `df[["name","parent_id"]].values` as the list of edges (source, target) to add. We additionnally create an attribute `link_type` and instantiate it as `parent` for every edges.

# In[ ]:


g.add_edges_from(df[["name","parent_id"]].values, link_type="parent")


# Note that if either a source or a target does not exist in the network, a node will be created. This can happen if a comment is a reply to a comment that is older than May2015. In this case, this "on-the-fly" created node do not have the attribute `type`.

# # Write the graph to use it later
# It can be convenient to save the network once created. A appropriate format is GML. Thus, using the function `nx.read_gml`function you could reload the network.

# In[ ]:


nx.write_gml(g, "graph.gml")


# # Conclusion
# This short notebook shows you how to create a network from raw data. What I showed you is not the only way to create network in the sense that it may exists some other network models that better suits your needs. Here, nodes are either links or comments and there is a link between nodes if there is a parenting relationship. A other way to model your data could be to define a user-user undirected network where a node is a user and there exists a link between two users if one has answered a comment from the other. In this way, you could extract communities of users and perhaps calculte some centrality metrics. 
# In conclusions, depending on the structural features you want to design, there exists an appropriate network you can exploit! Thus, in the project, you will likely build and analyze several networks!
