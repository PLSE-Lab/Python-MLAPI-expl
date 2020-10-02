#!/usr/bin/env python
# coding: utf-8

# # StackOverFlow Tag Network Visualization and Analysis Outline : 
# 
# * Making the network
# * Network Visualization
# * Finding Cliques 
# * Visualizing Maximal Clique 
# * Visualizing SubGraph of Programming Languages
# * Degree Distribution 
# * Node metadata distributions(group and nodesize)
# * Centrality Measures (degree and betweenness centrality)
# 

# # Build Network

# In[ ]:


import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


nodes = pd.read_csv('../input/stack_network_nodes.csv')
edges = pd.read_csv('../input/stack_network_links.csv')


# In[ ]:


nodes.head()


# In[ ]:


edges.head()


# A graph consists of nodes and edges.  In this case the nodes are the tags that appear in a developer's profile in stackoverflow ["Developer Stories"](https://stackoverflow.com/users/story/join). If two tags appear on the same profile there's a tag between them.  To build a graph in networkx first we define an empty graph. Then we add the nodes and the edges to the graph from the dataframes ```nodes``` and ```edges``` by iterating over the rows. 
# 
# > The dataset include only a subset of tags used on Developer Stories, tags that were used by at least 0.5% of users and were correlated with another tag with a correlation coefficient above 0.1. This means that very sparsely used tags and tags that are not used with other tags were filtered out. 
# 
# For more details see here : https://www.kaggle.com/stackoverflow/stack-overflow-tag-network/discussion/41307
# 
# Nodes and edges can have metadata associated with them. 
# 
# **Node metadata : **
# *  ```nodesize```  :  Proportional to how many developers have that tag in their developer story profile.
# *  ```group``` : which group that node belongs to (calculated via a cluster walktrap).
# 
# **Edge metadata : **
# *  ```value``` :  proportional to how correlated those two tags are (correlation coefficient * 100). 
# 
# 

# In[ ]:


G = nx.Graph()


# In[ ]:


for index, row in nodes.iterrows():
    G.add_node(row["name"],group = row["group"], nodesize = row["nodesize"] )


# In[ ]:


for index, row in edges.iterrows():
    G.add_edge(row["source"], row["target"], weight = row["value"])


# After building the network the first thing to check is the number of nodes and edges it consists of.  ```nx.info(G)``` also shows us the average degree. Degree of a node in an undirected graph shows the number of nodes it's connected to. We can also consider indegrees(number of edges incoming) and outdegrees(number of edges outgoing) of nodes. Average degree is the average of all node's degree. Stackoverflow tags have an average degree of 4.26 which indicates that on average tags are connected to four other tags. 115 nodes mean there are 115 unique tags which are connected among themselves with 245 edges.

# In[ ]:


print(nx.info(G))


# # Connectivity 
# 
# A network is connected if there is a path between every pair of vertices. But this tag network is not connected, which means there are some isolated nodes or isolated subgraphs. A connected component is the maximal connected subgraph of a graph. In the tag network we have 6 unique connected components. We can also extract the largest connected component of a graph.

# In[ ]:


nx.is_connected(G)


# In[ ]:


nx.number_connected_components(G)


# In[ ]:


maximum_connected_component = max(nx.connected_component_subgraphs(G), key=len)


# # Network Visualization
# 
# Note : I've used the code in this kernel : https://www.kaggle.com/jncharon/python-network-graph for the network visualization by encapsulating it in a function with minimal changes for convenience. I've changed the syntax for the list comprehensions for assigning node color and also changed the parameters for the spring layout a little bit. We can see the different connected components in the graph, often consisting of 2-3 edges. E.g (excel, excel-vba) and (testing, selenium) which probably refers to the business analysists and Quality assurance developers.

# In[ ]:


print(nx.__version__)


# In[ ]:


def draw_graph(G,size):
    nodes = G.nodes()
    color_map = {1:'#f09494', 2:'#eebcbc', 3:'#72bbd0', 4:'#91f0a1', 5:'#629fff', 6:'#bcc2f2',  
             7:'#eebcbc', 8:'#f1f0c0', 9:'#d2ffe7', 10:'#caf3a6', 11:'#ffdf55', 12:'#ef77aa', 
             13:'#d6dcff', 14:'#d2f5f0'}
    node_color= [color_map[d['group']] for n,d in G.nodes(data=True)]
    node_size = [d['nodesize']*10 for n,d in G.nodes(data=True)]
    pos = nx.drawing.spring_layout(G,k=0.70,iterations=60)
    plt.figure(figsize=size)
    nx.draw_networkx(G,pos=pos,node_color=node_color,node_size=node_size,edge_color='#FFDEA2',edge_width=1)
    plt.show()


# In[ ]:


draw_graph(G,size=(25,25))


# # Cliques 
# 
# In general we consider cliques as groups of people who are closely connected to each other but not connected to people outside the group. In network theory a clique is defined as a maximal complete subgraph of a graph where each node is connected to all the other nodes. The word 'maximal' means that if we add another node to the clique the clique will cease to be a clique. ```nx.find_cliques``` finds all the cliques in a network. We can also extract all the cliques from the tag network. 

# In[ ]:


cliques = list(nx.find_cliques(G))


# In[ ]:


clique_number = len(list(cliques))
print(clique_number)


# In[ ]:


for clique in cliques:
    print(clique)


# # Language Specific Ego Network And Cliques 
# 
# For each programming language there's a tag in the network. E.g 'python' will refer to the python language. So we can check the cliques that contains that node. We can also visualize the ego network for a node. Ego network for a node is the subgraph containing that node and all its neighbors with a specifed depth range. 
# 
# For example, we can check the ego network for python with radius 2, which means that we get the subgraph containing python and all it's direct neighbors which are 1 edge away from python  and also the nodes which are 2 hop away from python. 
# 
# Ego networks can be used for checking shortest paths or generally conducting analysis of who is connected to whom, but cliques are helpful because it shows us the data in a more granular way. 

# In[ ]:


print(nx.ego_graph(G,'python',radius=2).nodes())


# Python participates in 4 different cliques,  one for web development with django and flask, one for open source development presumably which is connected to linux. One for machine learning where it's adjacent to R. I think the fourth one is for porting python and C/C++ back and forth.

# In[ ]:


nx.algorithms.clique.cliques_containing_node(G,"python")


# In[ ]:


nx.algorithms.clique.cliques_containing_node(G,"c++")


# In[ ]:


nx.algorithms.clique.cliques_containing_node(G,"php")


# # Visualize Maximal Clique
# 
# It's possible that visualizing the largest cliques will let us see some pattern in the data. After finding all the cliques here we sort them by the length(number of nodes in that clique) and draw the cliques with the maximum length. ```G.subgraph``` allows us to extract a subgraph from the graph by passing a list of nodes. We have 3 cliques of size 7 which are the biggest, however I've only taken the unique nodes in a set while extracting the subgraphs, so we can see two different clusters containing javascript and .net related tags.

# In[ ]:


sorted_cliques = sorted(list(nx.find_cliques(G)),key=len)


# In[ ]:


max_clique_nodes = set()

for nodelist in sorted_cliques[-4:-1]:
    for node in nodelist:
        max_clique_nodes.add(node)


# In[ ]:


max_clique = G.subgraph(max_clique_nodes)


# In[ ]:


print(nx.info(max_clique))


# In[ ]:


draw_graph(max_clique,size=(10,10))


# # Visualizing Programming Language Network
# 
# Since it's possible to draw the subgraph of a graph, a subgraph containing the nodes for the programming languages only can also be visualized.  In the visualization its possible to see the different clusters for each programming language and familiar patterns like android with java or embedded systems with C and C++.

# In[ ]:


major_languages = ['c','c++','c#','java','python','ruby','scala','haskell','javascript','sql']


# In[ ]:


p_language_nodes = []
for language in major_languages:
    neighbors = G.neighbors(language)
    p_language_nodes.extend(neighbors)


# In[ ]:


programming_language_graph = G.subgraph(set(p_language_nodes))


# In[ ]:


draw_graph(programming_language_graph,size=(20,20))


# # Degree Distribution 
# 
# For checking the degree distribution of the graph plotting the list containing degrees for each node works. In the tag network clearly most tags hae only 1 or 2 neighbors while some tags are linked to more than 10-12 tags.

# In[ ]:


plt.hist([node[1] for node in list(G.degree())])
plt.title("Stack Overflow Tag Degree Distribution")


# # Group Distribution 
# 
# For this network since the group numbers are provided it's possible to check how many tags fall into each group. Group 6 and 1 has the highest number of nodes falling into them while group 10-12 probably refers to the isolated nodes.

# In[ ]:


nodes['group'].plot(kind='hist')


# # Node size distribution 
# 
# Again, since the node size metadata is provided it's possible to check the node size distribution.

# In[ ]:


nodes['nodesize'].plot(kind="hist")


# # Centrality Measures
# 
# Centrality measures helps us to idenfity the most important nodes or vertices in a graph. Different centrality measures like degree centrality, betweenness centrality, eigenvector centrality are used to measure influence of nodes in a network.
# 
# 
# * **Degree Centrality : ** Degree centrality of a node is the fraction of the nodes it's connected to. Intuitively, the greater the degree that node can be more powerful. For example we can think that a twitter celebrity with 1 m follower is more influential than a regular user with 100 followers.
# 
# * **Betweenness Centrality :** Betweenness centrality is a measure of centrality in a graph based on the idea of shortest path. Betwenness centrality of node A is fraction of shortest paths that passes through node A. Nodes with high betweeness centrality works as the 'power broker' or the 'bridges' between different isolated parts of a network.
# 
# Here we idenfity the top 10 nodes according to both of the centraliy measures, but they overlap a lot , presumably because it's just a co-occurance network of tags and undirected. In a human social network often the people with higher betweenenss centrality are more interesting.

# In[ ]:


degree_centrality = nx.degree_centrality(G)


# In[ ]:



top_10_nodes_by_degree_centrality = sorted(degree_centrality.items(),key=lambda x:x[1],reverse=True)[0:10]

top_10_nodes_by_degree_centrality


# In[ ]:


betweenness_centrality = nx.betweenness_centrality(G)


# In[ ]:


top_10_nodes_by_betweenness_centrality = sorted(betweenness_centrality.items(),key=lambda x:x[1],reverse=True)[0:10]


# In[ ]:


top_10_nodes_by_betweenness_centrality


# # More Resources : 
# 
# * Datacamp Network Analysis Course Part 1 and 2.
# * https://github.com/ericmjl/Network-Analysis-Made-Simple
# 

# In[ ]:




