#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, you will learn how to create your first graph with **networkx** and then jump to an example to see how graph theory can help you solve a real-life problem 

# # Creating a network

# A **graph** (or a network) consists of a set of nodes and a set of edges. Edges connect one node to anothers. We are dealing with a **simple graph in this notebook**, one edge has exactly two endpoints and two points has at most one adjacent edge. An edge can have directions or not. An edge can have a **weight** (a positive real number) associated with it. 
# 
# A real-life example of a graph is the map system. Houses are nodes, route from one house to another is the edge, and weight is distance. Sometimes, all roads are two-way, we call this an **undirected** graph. If the road is one-way, we call it a **directed** graph.
# 
# We are going to create a visual of a undirected weighted graph (with edge weights).

# In[ ]:


import networkx as nx # The graph library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import Image #For image
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


G = nx.Graph() #Initialize

#Creating set of nodes
nodes = ["A", "B", "C", "D", "E","F"]
for node in nodes:
    G.add_node(node)

#Creating a set of edges
edges = {"A": [("B",7), ("C",9), ("F",14)], #edges AB, AC, AF with weight 7, 9, 14 respectively
         "B": [("A",7),("C",10), ("D",15)],
         "C": [("A",9), ("B",10), ("D",11), ("F",2)],
         "D": [("B",15), ("C",11), ("E",6)],
         "E": [("D",6), ("F",9)],
         "F": [("A",14), ("C",2), ("E",9)]
        }

#Add edges with direction to graph G
for node in edges.keys():
    neighbors = edges[node]
    for neighbor_node in neighbors:
        G.add_edge(node, neighbor_node[0],weight=neighbor_node[1]) #Source node, node, weight


# It's hard to put weight on edges using networkx, so I will just draw it with out label. 

# In[ ]:


#Graph edge with edge weight label is hard!
#Let do it with out weight label
nx.draw(G, with_labels=True, node_color="#1cf0c7", 
        node_size=1500, alpha=0.7, font_weight="bold", pos=nx.circular_layout(G)) #Adding pos make it look cleaner


# In[ ]:


#Displays edges and edge weights of graph G
print("All the edges of G:", G.edges)
print("Weight on edge FE:", G.edges[('F', 'E')])
print("Neighbor node of F:", ['F']) 


# # Dijkstra's algorithm (shortest path)
# 
# pronounce: Diekstra
# 
# **Dijkstra's algorithm** (or Dijkstra's Shortest Path First algorithm, SPF algorithm) is an algorithm for finding the shortest paths between nodes in a graph. A description of the algorithm can be found [here](https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/).
# 
# Below is a gif that sum of the process of finding the shortest path from node 1 to node 5.  (Source: wiki image)

# <img src="https://upload.wikimedia.org/wikipedia/commons/5/57/Dijkstra_Animation.gif">

# Observe that, there are many path from $1$ to $5$, but the shortest one is $1->3->6->5$ since the total distance is $9+2+9=20$
# 
# Similarly, we will use the graph created above and find the shortest path from A to E. It is a little bit hard to see, but A is equivalent to node 1, B is 2, C is 3, D is 4, and E is 5. Use your imagination to verify that these two graph are the same!

# In[ ]:


nx.draw(G, with_labels=True, node_color="#1cf0c7", 
        node_size=1500, alpha=0.7, font_weight="bold", pos=nx.circular_layout(G)) #Adding pos make it look cleaner


# Below are two set of codes which find the shortest path with weighted version (or all edge weight is 1) and with unweighted version

# In[ ]:


#with no weight
print("Is there a path from A to E in graph G: ", nx.has_path(G, "A", "E"))
print("Shortest path from A to E not considering the weight: ",nx.shortest_path(G, "A", "E")) 
print("Number of edges to get there (distance): ",nx.shortest_path_length(G, "A", "E"))

#with weight
print("************************************************************")
print("Dijkstra path from A to E: ",nx.dijkstra_path(G, "A", "E"))
print("The distance of that path",nx.dijkstra_path_length(G, "A", "E"))


# ## Centrality
# 
# Centrality describes the influence/importance of nodes in the graph. There are three metrics 
# 
# * **Degree-centrality**: The number of edges attached to a node divide by the maximum edges a node can have
# * **Closeness-centrality**: The reciprocal of the sum of the distances to all other nodes in the network 
# * **Betweenness-centrality**: The number of shortest paths between all node pairs the node lies on divided by the maximum number of shortest-paths any one node in the network lies on.
# * **Eigenvalue-centrality**: An iterative algorithm that assigns relative influence to a node based on the number and importance of connected nodes. It can be very computationally expensive to compute for large networks. Google's PageRank algorithm is a variation of eigenvalue-centrality.
# 
# All of these metrics basically tell how important a node is. 

# In[ ]:


ls ../input/facebook/facebook_network.png


# In[ ]:


Image(filename = "../input/facebook/facebook_network.png")


# Let's create a same graph but with no weight

# In[ ]:


G = nx.Graph() #Initialize

#Creating Nodes
nodes = ["A", "B", "C", "D", "E","F"]
for node in nodes:
    G.add_node(node)
    
#Adding edges
edges = {"A": [("B"), ("C"), ("F")],
         "B": [("A"),("C"), ("D")],
         "C": [("A"), ("B"), ("D"), ("F")],
         "D": [("B"), ("C"), ("E")],
         "E": [("D"), ("F")],
         "F": [("A"), ("C"), ("E")]
        }

for node in edges.keys():
    neighbors = edges[node]
    for neighbor_node in neighbors:
        G.add_edge(node, neighbor_node) #Source node, node, weight
        
nx.draw(G, with_labels=True, node_color="#1cf0c7", 
        node_size=1500, alpha=0.7, font_weight="bold", pos=nx.circular_layout(G))


# Calculate the centrality of each nodes.

# In[ ]:


degrees = nx.degree_centrality(G)
closeness = nx.closeness_centrality(G)
betweeness = nx.betweenness_centrality(G)
eigs = nx.eigenvector_centrality(G)

centrality = pd.DataFrame([degrees, closeness, betweeness, eigs]).transpose()
centrality.columns = ["degrees", "closeness", "betweeness", "eigs"]
centrality = centrality.sort_values(by='eigs', ascending=False)
centrality


# # Application
# 
# Imagine that we want to design an online bookstore that has a recommendation feature for the user. There are two main keys: users and items. We have to choose which key is the node and which is the edge of our graph. If we choose the user as the node, we are using **User-Based Collaborative Filtering**. If we choose items as the nodes, we are using **Item-Based Collaborative Filtering**
# 
# **User base:** When recommending items to a user whether they be books, music, movies, restaurants or other consumer products one is typically trying to find the preferences of other users with similar tastes who can provide useful suggestions for the user in question. With this, examining the relationships amongst users and their previous preferences can help identify which users are most similar to each other
# 
# **Item base:** Alternatively, one can examine the relationships between the items themselves.

# # Example: Book Recommendation
# 
# We are going to create the a book recommendation using items base. The weight is already provided from the data

# In[ ]:


ls ../input/edge-data/books_data.edgelist


# In[ ]:


#This data set contains node(book), its neighbors (similar book), and similarity score (weight)
df = pd.read_csv('../input/edge-data/books_data.edgelist', names=['source', 'target', 'weight'], delimiter=' ')
df.head()


# Note that the weight here represent the similarity of any two books

# In[ ]:


#Load second data which has more detail
meta = pd.read_csv('../input/edge-data/books_meta.txt', sep='\t')
meta.head()


# Note that **Clustering Coefficient** is s a measure of the degree to **which nodes in a graph tend to cluster together**. Evidence suggests that in most real-world networks, and in particular social networks, nodes tend to create tightly knit groups characterised by a relatively high density of ties; this likelihood tends to be greater than the average probability of a tie randomly established between two nodes (Holland and Leinhardt, 1971 Watts and Strogatz, 1998).
# 
# The (Local) **clustering coefficient** formula for a directed graph is:
# $$C_{i}={\frac  {|\{e_{{jk}}:v_{j},v_{k}\in N_{i},e_{{jk}}\in E\}|}{k_{i}(k_{i}-1)}}$$
# where
# * $e_{jk}$ is edge between nodes $v_j$ and $v_k$
# * $N_{i}$ is the neighbor of node $v_i$
# * $k_i$ is the number neighbor nodes of $v_i$. Note that there are $k_{i}(k_{i}-1)$  links that could exist among the vertices within the neighbourhood $N_i$
# For a undirected graph, just multiply the formula by 2. More about it [here](https://en.wikipedia.org/wiki/Clustering_coefficient).

# In[ ]:


#Type your preference book here
GOT = meta[meta.Title.str.contains('Harry Potter and the Order of the Phoenix')]
GOT


# The code bellow will 
# 
# 1) Identify the node correspond to the book
# 
# 2) Find its neighbors and sort by weight. 
# 
# 3) Display the top ten books

# In[ ]:


rec_dict = {}
id_name_dict = dict(zip(meta.ASIN, meta.Title))
for row in GOT.index:
    book_id = GOT.ASIN[row]
    book_name = id_name_dict[book_id]
    most_similar = df[(df.source==book_id)
                      | (df.target==book_id)
                     ].sort_values(by='weight', ascending=False).head(10)
    most_similar['source_name'] = most_similar['source'].map(id_name_dict)
    most_similar['target_name'] = most_similar['target'].map(id_name_dict)
    recommendations = []
    for row in most_similar.index:
        if most_similar.source[row] == book_id:
            recommendations.append((most_similar.target_name[row], most_similar.weight[row]))
        else:
            recommendations.append((most_similar.source_name[row], most_similar.weight[row]))
    rec_dict[book_name] = recommendations
    print("Recommendations for:", book_name)
    for r in recommendations:
        print(r)
    print('\n')


# ### References
# All about drawing graph [here](https://qxf2.com/blog/drawing-weighted-graphs-with-networkx/)
# 
# How weight between two books is calculated: [Association Rules Generation from Frequent Itemsets](http://rasbt.github.io/mlxtend/user_guide/frequent_patterns/association_rules/#example-1-generating-association-rules-from-frequent-itemsets)
# 
# There are also alternative approaches to [recommendations systems](https://www.researchgate.net/publication/256458336_Basic_Approaches_in_Recommendation_Systems)
