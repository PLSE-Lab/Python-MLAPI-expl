#!/usr/bin/env python
# coding: utf-8

# # Social Computing - Summer 2019
# 
# # Exercise 2 - Centrality
# Centrality is a key concept in social network analysis. It measures the importance or influence of a certain node/edge in a network. The interpretation of importance or influence, however, depends on the type of centrality and the application for which it is measured. Different types of centrality were discussed in the lecture: degree centrality, closeness centrality, betweenness centrality and eigenvector centrality.<br>
# In this exercise, you are going to implement different centrality algorithms using the igraph library which you already know from last exercise. You can find its manual [here](https://igraph.org/python/doc/igraph-module.html) and a tutorial [here](https://igraph.org/python/doc/tutorial/tutorial.html).

# ## Introduction Problem: The Krackhardt Kite Graph
# We will use the Krackhardt Kite for the first exercise. As you know from exercise 1, the Krackhardt Kite is a simple connected, unweighted and undirected graph. [This figure](https://en.wikipedia.org/wiki/Krackhardt_kite_graph#/media/File:Krackhard_kite.PNG) illustrates the Krackhardt Kite.
# 
# **Calculate the degree centrality of the Krackhardt Kite graph - just a list of ten values, one for each node. You can use the pre-defined function of the igraph library.**
# 
# **Optional:** Look at the graph and the list with the degree centrality values. Can you identify which node has which degree centrality?<br>
# **Optional:** Calculate the closeness and betweeness centrality as well. What information do they give us?

# In[ ]:


import igraph as ig

# Importing the graph (connected, unweighted, undirected social network)
krackhardt_kite = ig.Graph.Famous('Krackhardt_Kite')

# Formatting the graph
visual_style = {}
visual_style['vertex_size'] = 20
visual_style['bbox'] = (300, 300)
visual_style['margin'] = 20

# TODO: Calculate the Kite's degree centrality
degree_centrality = krackhardt_kite.degree()
print(degree_centrality)
# TODO: Plot the graph
ig.plot(krackhardt_kite)


# ## Problem 2.1: Degree Centrality
# In the following three problems, you are working with an anonymized real-life social network from [1] represented in the file _UniversityNetwork.graphml_. It represents the faculty of a university, consisting of individuals (vertices) and their directed and weighted connections (edges). The nodes have attributes (which faculty the person is affiliated with), but we will neglect that information. The edges' weights are a measure of friendship between the persons.
# 
# **Your task in this exercise is to read in the graph and to calculate the degree centrality of all the nodes in it. Plot the graph as well.** You are **not allowed** to use the pre-defined function `degree()` but have to implement your own. The output should be a list of integers - nodes with a centrality of 0 do not need to be listed, but can be. 
# 
# [1] T. Nepusz et al: _Fuzzy communities and the concept of bridgeness in complex networks._ Physical Review E 77:016107, 2008.
# 
# **Notes:**
# * Degree centrality of a graph node is the number of edges (incoming and outgoing) of that node.
# * The functions `Read_GraphML()` and `are_connected()` might help you with the task.

# In[ ]:



# Calculates degree centrality for a graph g
def degree_centrality(g):
    myList = [None] * 81
       
    nodes = g.vs
    for i in range(len(nodes)):
        for j in range(i+1,len(nodes)):
            if g.are_connected(nodes[i],nodes[j]):
                if myList[i] != None:
                    myList[i] += 1
                else:
                    myList[i] = 1
                    
                if myList[j] != None:
                    myList[j] += 1
                else:
                    myList[j] = 1
                
                
    
    print(myList)   
    

# TODO: Import the graph

g = ig.Graph.Read_GraphML("../input/UniversityNetwork.graphml")


# Formatting the graph
visual_style = {}
visual_style['vertex_size'] = 10
visual_style['vertex_label'] = g.vs['id']
visual_style['bbox'] = (700, 700)
visual_style['margin'] = 50

# TODO: Calculate the degree centrality
degree_centrality(g)
# TODO: Plot the graph
ig.plot(g)


# ## Problem 2.2: Closeness Centrality
# 
# Now we want to take a closer look at the closeness centrality for the given network. It measures how close a node is to other nodes in the graph. This is calculated via the sum of distances from that node to all the other nodes in the graph.
# 
# **Write a Python program that computes the closeness centrality for each node for the given social network.** The output should be a list where each item contains the value of the closeness centrality of a node. You are **not allowed** to use the pre-defined function `closeness()` , but you can use it as an inspiration.
# 
# **Notes:**
# * The formula for the closeness centrality can be found in the lecture or exercise slides.
# * Calculating the shortest paths is a common problem, maybe there is a pre-defined function for that?
# * The edges of the graph have weights which you need to take into account for shortest paths calculation.
# * You can print the node ID list with: `print(g.vs['id'])`
# * You can print the edge list with: `print(g)`

# In[ ]:


# Calculates the closeness centrality for a graph g
def closeness_centrality(g):
    # TODO: Calculate shortest paths list for each node
    
    # TODO: Calculate closeness centrality for each node
    
# TODO: Calculate closeness centrality


# ## Problem 2.3: Betweenness Centrality
# 
# Betweenness centrality also measures centrality based on shortest paths. For every pair of vertices in a graph, there exists a shortest path between the vertices such that either the number of edges that the path passes through (for undirected graphs) or the edges' sum of the weights (for directed graphs) is minimized.<br>
# Vertices with high betweenness may have considerable influence within a network by virtue of their control over information passing between others.
# 
# **Calculate the betweenness centrality with the help of the pre-defined function in the igraph library. Interpret the resulting values based on two exemplary nodes.** To do that, pick two nodes and explain how their betweenness centrality links to the graph structure. Name the two nodes that you discussed (and their betweenness centrality). Do not write more than 5 sentences.

# In[ ]:


# TODO: Calculate the betweenness centrality (using the pre-defined function is fine)
btwn = g.betweenness()
print(btwn)


# **TODO: Write your discussion here!**
