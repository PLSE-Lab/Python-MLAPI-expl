#!/usr/bin/env python
# coding: utf-8

# # HW1 Part 3. Overview. Modularity and Louvain algorithm. 
# 
# Part 3 of HW1 is devoted to the concpet of modularity and Louvain algorithm.
# 
# The present notebook contains four parts, the first is a reminder and explanation of subtle points in the material, the next three parts are  devoted to the solution of the HW1 Part 3:
# 
# The preliminary part contains: 
# 
# 1) Reminder on  modularity and Louvain
# 
# 2) Working out explicit examples of calculation of modularity and Louvain by hands. 
# 
# 3) Explain some sublte unexpected technical points in calculations,
# which are related self-edges, and some unexpected multiplication by 2 in weights calculation. Pay attention you understand these detail, before you go to exercises. 
# 
# 4) Examples how to calculate the modularity and Louvain by the igraph package (and partly by networkx)
# 
# The "part 3.1" contains: solution of the first exercise - what happethe explanation of the  formula for the modularity change when one node is moved from one group to another. 
# 
# The "part 3.2" contains: solution of the second exercises-set - devoted to working out part of the Louvain on 16 node graph.
# 
# The "part 3.3" contains: solution of the third exercises-set - devoted to working out part of the Louvain on 128 node graph.
# 
# 
# 
# 
# 
# 
# 
# Notebook by Alexander Chervov June 2020
# 
# PS 
# 
# Notebook is quite big , and so it is better seen on colab or other  ide with "Table of Contents" option - for easy navigation between sections
# 

# In[ ]:


# In what follows we will use igraph package. 
# Not networkx because networkx cannot plot self-loops

# To install:
not_kaggle = 0
if not_kaggle: # igraph is already preinstalled on kaggle, but not colab  
  get_ipython().system('pip install python-igraph # Pay attention: not just "pip install igraph" ')
  get_ipython().system('pip install cairocffi # Module required for plots ')
import igraph # On kaggle it is pre-installed , do not need to pip install 
import numpy as np


# # Preliminary part. Modularity and Louvain algorithm - reminder and subtle points

# ## Modularity - reminder

# ### Modularity - brief ideas. All what you need to know, if do not want  going into technical  details 
# 
# (1)
# Modularity - is a number for a graph and a clustering of its nodes, it says how "good" is your clustering. It takes values from -1 to 1 , where 1 means super-good clusters found, 0 - clustering is not better than just random, high  negative number would correspond to something like anti-clusters.
# Modularity in the range 0.3-0.7 means good clusters found - that numbers people want to see in practice.
# 
# For clustering data-clouds, not graphs, one typically uses "silhouette coefficient" to measure clustering quality, so modularity is its kind of "analogue" but graph clustering. 
# 
# 
# (2) 
# The intuition behind its caculation is the following - clusters are "good" 
# if there are many links inside clusters and not so many between clusters. 
# 
# (3)
# Many graph packages have built-in functions to calculate modularity.
# For example: 
# 
# networkx: 
# 
# modularity = nx.algorithms.community.modularity(graph_instance, [{0,2,3}, {1}, {4,5}]) # Clustering is given by listing all members of clusters
# 
# igraph:
# 
# modularity = graph_instance.modularity([0,1,0,0,2,2]) # Clustering is given by vector showing to what cluster i-th node belongs to.  That is different from networkx 
# 
# 
# (4) There are algorithms which aim to produce "good" clustering, Louvain algo (2008) is one of them discussed here in details. More recent is Leiden (2018). Actually there many of them, but these two most popular.
# 
# 
# PS
# 
# What follows below is quite long, but a kind of tecnical details,
# which might not be necessary to understand for practical purposes.
# In lectures, one technical detail is not given very clearly - 
# during calculation with self-loops one should multiply its weight by 2.
# That is somewhat unexpected. 
# Please do pay attention on it, otherwise you will spent time on same bugs as we did. It is important to understand these details if one is going to do the homeworks. 
# 
# 
# 
# 

# ### Modularity - conceptual view on the definition
# 
# That is explained quite well in lectures. Please look there. In brief - comparing the real number of edges in each cluster with "expected" number of edges, where "expected" means from very very simple reasons. 
# 

# ### Modularity - mathematical definition. Subtleties with self-loops.
# 
# Modularity for a pair: a graph and a clustering of its nodes into commnuties (groups) $C_i$ is defined by the formula:
# $$
# Q = \frac{1}{2m} \sum_{vw} \left[ A_{vw} - \frac{k_v k_w}{2m} \right]  \delta(C_u,C_v)           
# $$
# 
# where  summation is going over all pairs of nodes (u,v). $\delta(C_u,C_v)$ 
# is 1 when u,v are in the same community and 0 otherwise. $A_{uv}$ adjacency matrix (see, however, subtleties below). $k_u$ is sum of weights of edges attached to $u$, i.e. $k_u = \sum_v A_{uv}$ (for unweighted graph it is just a degree of $u$  (see, however, subtleties below) ).  $2m = \sum_{u,v} A_{uv}$. 
# 
# The definition should be seen clear. But... actually , there are certain traps hidden. They are not conceptual, but just technical.
# Let us clarify these things: 
# 
# **Pay attention(!) there are several subtleties in definition**:
# 
# (1) If self-loops exist in the grapth they contribute  with the weight multiplied by 2 to  the adjacency matrix, i.e. $A_{uu} = 2*weight ~ of  ~ self-loop ~ from ~ u ~ to ~ u $. That multiplication by 2 is something unexpected. Actually it is just only matter of convention, but it is standard convention.
# 
# (2) Respectively $k_u = \sum_v A_{uv}$ that means weight of any self-loop is doubled here, becuase it is doubled in $A_{uu}$. For example, if graph is just one node with a self-loop, then $k_u=2$ , but not $1$ , as one would naively expect. Moreover packages like igraph and networkx would say that degree in that case equal to $2$, but snap will give $1$. 
# 
# (3) Adjacency matrix $A_{uv}$ for undirecrted graphs is constructed as symmetric, i.e. both $A_{vu}$ and $A_{uv}$ equal to the same $"weight~of~edge~{uv}"$, i.e. $A_{vu}= A_{uv}= weight~of~edge~{uv}$. (In present study we always consider only undirected graphs.)
# 
# (4) Summation above contains pairs (u,v) and (v,u) and (u,u) - all of them.
# For undirected network $A_{uv} = A_{vu}$, and so equal terms are summed-up twice. 
# 
# 

# ### Example of modularity calculation for the simplest graph
# 
# Let us consider the simplest undirected graph (see plot in the cell below) - just the two nodes: red and green connected with an edge and one additional self-loop edge from the red node to itsef. 
# And calculate modularity for it by hands (see cell below) and compare with built-in functions in igraph and networkx 

# #### Create Figure for graph

# In[ ]:


import igraph
g = igraph.Graph(directed=False)
g.add_vertices(2)
g.add_edges([[0,0],[0,1]] ) # ,[1,2],[2,0]
modularity1 = g.modularity([0,1])#, weights=graph.es['weight'])
print("Built-in igraph: The modularity =  {} for the partition - put  all nodes in different groups".format(modularity1))
print('Calculation "by hands" explained in details below')
visual_style = {}
visual_style["margin"] = 80
visual_style["vertex_color"] = ['red', 'green']
igraph.plot(g,**visual_style, bbox = [200,100])


# #### Calculations by hands
# 
# Let us consider the simplest undirected graph (see plot in the cell above or below) - just the two nodes: red and green connected with an edge and one additional self-loop edge from the red node to itsef. The graph is "unweighted" which means all weights of edges equal to 1.
# 
# Let us calculate the modularity for it. 
# 
# First let us consider the adjacency matrix:
# **Pay attention: here is 2  in position (1,1)** (not 1 as one would naively expect!), that is because our rule - to double weight of each self-loop edge. 
# Pay attention that standard adjacency matrix, both in igraph and networkx would have 1 at position (1,1). 
# 
# $$
# A = \begin{pmatrix}
# 0 & 1 \\
# 1 & 2
# \end{pmatrix}.
# $$
# 
# Respectively normalization factor $2m = \sum_{ij} A_{ij} = 4$,
# 
# The weighted degrees $k_{green} = A_{00}+A_{10} = 1 , k_{red}= A_{01}+A_{11}  = 2+1 = 3$. 
# 
# The modularity is the sum of 4 summands (all pairs u,v), however two of them dissappear because of the term $\delta(C_u,C_v)$  - the only two terms survive:
# 
# u=green, v=green:  $ 1/2m( A_{00} - k_{green}k_{green}/2m) = 1/4 (0 - 1/4) = -1/16   $
# 
# u=red, v=green - does not contribute, since u,v in different groups
# 
# u=green, v=red: - does not contribute, since u,v in different groups
# 
# u=green, v=green:  $ 1/2m( A_{11} - k_{green}k_{green}/2m) = 1/4 (2 - 3*3/4) =  1/4(-1/4) = -1/16  $
# 
# Summing up we get: $ -1/16-1/16 = -1/8 = -0.125 $ 
# The result exactly as igraph modularity function gives. Later we check with the networkx - gives the same. 
# 

# In[ ]:


print("Built-in igraph: The modularity =  {} for the partition - put  all nodes in different groups".format(modularity1))
visual_style = {}; visual_style["margin"] = 80;  visual_style["vertex_color"] = ['red', 'green']
igraph.plot(g,**visual_style, bbox = [200,100])


# #### check by networkx built-in - the same result -0.125

# In[ ]:


import networkx as nx
G = nx.Graph()
G.add_nodes_from([0, 1])
G.add_edges_from([(0, 1), (1,1) ] ) 
m1 = nx.algorithms.community.modularity(G, [{0}, {1}])
print("Built-in networkx: The modularity =  {} for the partition - put  all nodes in different groups".format(m1))


# ### Propeties of modularity 
# 
# Proposition 1: modularity is in [-1,1] (if weights are positive). Moreover for undirected, unweighted graph it is [-1/2 ,1 ]. 
# 
# Proposition 2: If all nodes are in the same group, then  modularity = 0 
# 
# Proposition 3: If there is lonely node (i.e. not connected to any other) then joining it to any group will not change modularity 

# ### Code to calculate modularity 
# 
# For the sake of better understanding let us present a code how to caclculate modularity (for undirected, but possibly weighted graphs). And check that our code gives the same results as built-in igraph function.
# 

# In[ ]:


# Below is simple code to calculate modularity, it is not optimized - just for better understanding of the definition
# One may also look at networkx Python implementation - 
# nx.algorithms.community.modularity in file:  /usr/local/lib/python3.6/dist-packages/networkx/algorithms/community/quality.py

def modularity_byhands(g,communities_membership_vector ):
  '''
  Modularity of graph is caculated and returned
  # Input:
  # g - undirected graph (igraph is supported, but easy to modify for other)
  # communities_membership_vector -  for each node the id of its community is given
  
  # Pay attention on multiplier = 2 for self-loop edges contribution to adjacency matrix 
  '''

  g.es['weight'] = [1,1]
  n_nodes = g.vcount()

  # Prepare symmetric  adjacency matrix,
  # Pay attention - diagonal elements A_{uu} = 2* "weight of  self-loop edge from u to u" (if such exists, otherwise zero), 
  # That is somewhat unexpected, typically one do not 
  A_matrix = np.zeros( (n_nodes,n_nodes) )
  for i in range(n_nodes): 
    for j in range(n_nodes): 
      e = g.get_eid(g.vs[i], g.vs[j], directed=False, error=False)
      if e != -1: # Check  is there an edge 
        multiplier = 1
        if i==j: multiplier = 2
        if g.is_weighted():
          A_matrix[i][j] = g.es['weight'][e]*multiplier # Multiply by 2, in case of self-loop edges 
        else:
          A_matrix[i][j] = 1*multiplier
      else:
        A_matrix[i][j] = 0

  vect_k_u = np.sum(A_matrix, axis = 1 ) # k_u - weighted degree of all nodes , denoted as k_u in lectures 
  normalization_2m = np.sum(A_matrix) # normalization factor denoted as 2m in lectures

  modularity = 0
  for i in range(n_nodes): 
    for j in range(n_nodes): 
      if communities_membership_vector[i] == communities_membership_vector[j]:  
        modularity += ( A_matrix[i][j] - vect_k_u[i]*vect_k_u[j]/ normalization_2m ) / normalization_2m

  return modularity

g = igraph.Graph(directed=False)
g.add_vertices(2)
g.add_edges([[0,0],[0,1]] ) # ,[1,2],[2,0]


modularity1 = g.modularity([0,1])#, weights=graph.es['weight'])
print("Builtin igraph: The modularity =  {} for the partition - put  all nodes in different groups".format(modularity1))
modularity2 = modularity_byhands(g,[0,1]) #communities_membership_vector )
print("Our function: The modularity =  {} for the partition - put  all nodes in different groups".format(modularity1))

visual_style = {}
visual_style["margin"] = 80
visual_style["vertex_color"] = ['red', 'green']
igraph.plot(g,**visual_style, bbox = [200,100])


# ### Compare our function for modularity with built-in in igraph on random graphs - results coincide

# In[ ]:


for i in range(10):
  n = 100
  m = 200
  g = igraph.Graph.Erdos_Renyi(n,  m=m, directed=False, loops=False) # Generates a graph based on the Erdos-Renyi model.
  # https://igraph.org/python/doc/igraph.Graph-class.html

  communities_membership_vector = range(g.vcount()) # all nodes in separate groups 
  modularity1 = g.modularity(communities_membership_vector)#, weights=graph.es['weight'])
  print("Builtin igraph: The modularity =  {} for the partition - put nodes all in different groups".format(modularity1))
  modularity2 = modularity_byhands(g,communities_membership_vector) #communities_membership_vector )
  print("Our function: The modularity =  {} for the partition - put nodes all in different groups".format(modularity1))
  if (modularity1 - modularity2) < 1e-14:
    print('Okay. Our function gives same result as built-in')
  else:
    print('NOT OKAY. Out function gives different then built-in')
    raise ValueError()


# ## Louvain algorithm - reminder

# ### Louvain algo - in brief - all what you need to know, if do not want to go into technical details
# 
# (1) Louvain algorithm is an algorithm to cluster nodes of graphs into some groups. It was proposed in 2008 and become extremely popular. It can process quite large graphs with reasonable speed. In 2018 new algo - Leiden has been proposed and seems to be better both in quility and speed. 
# 
# (2) Actually algorithm is very simple.
# Brief idea  - that one starts with each node in seperate group, than join nodes such that modularity increases , later collapse all nodes in obtained groups to get a new graph with recalculated weights. Than repeat process many times. 
# It is discussed in details later. 
# 
# (3) 
# Many packages have its implementation.
# 
# louvain_partition = g.community_multilevel() # igraph 
# 
# 
# PS
# 
# What follows below is quite long, but a kind of technical details, which might not be necessary to understand for practical purposes. In lectures, one technical detail is not given very clearly - during calculation at "phase 2" for weight of new self-loops one should multiply weights of cross edges by 2. That is somewhat unexpected. Please do pay attention on it, otherwise you will spent time on same bugs as we did. It is important to understand these details if one is going to do the homeworks. 
# 

# ### Louvain algorithm - description 
# 
# **Phase 1 (Modularity Optimization):** Start with each node in its own community.
# For each node "i", go over all the neighbors "j" of "i". Calculate the change in modularity when "i" is moved from its present community to j`s community. Find the neighbor jm for which this process causes the greatest increase in modularity, and assign "i" to jm`s community (break ties arbitrarily). If there is no positive increase in modularity during this process, keep "i" in
# its original community.
# 
# Repeat the above process for each node (going over nodes multiple times if required) until there is no further maximization possible (that is, each node remains in its community). This
# is the end of Phase 1. 
# 
# **Phase 2 (Community Aggregation)**: Once Phase 1 is done, we contract the original
# graph G to a new graph H. Each community found in G after Phase 1 becomes a node in H. The weights of edges in between 2 nodes in H are given by the sums of the weights between the respective 2 communities in G. The weights of edges within a community in G sum up to form a self-edge of the same weight in H. 
# **Pay attention** during that sum up weights of edges between distinct nodes are **multiplied by 2**, but weights of self-edges come as they are (i.e. with factor 1). (See detailed example below). This is the end of Phase 2. Phase 1 and Phase 2 together make up a single pass of the algorithm.
# 
# 
# **Repeat Phase 1 and Phase 2** again on H and keep proceeding this way until no further improvement is seen (you will reach a point where each node in the graph stays in its original community to maximize modularity). The final modularity value is a heuristic for the maximum modularity of the graph.
# 
# -----------------------------------------------------
# 
# https://en.wikipedia.org/wiki/Louvain_modularity
# 
# Original paper 
# https://arxiv.org/abs/0803.0476 Fast unfolding of communities in large networks
# Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre
# (it is cited 11K times - increadibly great citation number):
# 
# 

# ### Illustration for  the second phase (aggregation) of Louvain algo - pay attention on doubling of weights of non-self-loops.
# 
# Let us look in details on the example presented in original article (Figure 1) (and also the lectures) for the second phase (aggregation) of the Louvain algo. 
# Pay attentation on the detail - the weights of non-self-edges are multiplied by 2 when aggregated to super-node. It might not be very clear from expositions above. 
# 
# For the example given below we will see: 
# 
# 26 = 14 + 4 + 2*4 Pay attention that weight 4 for edge between green-blue is multiplied by 2'
# 
# 24 = 16 + 2 + 2*3 Pay attention that weight 3 for edge between red-aqua is multiplied by 2
# 
# 3 = 1 + 1 + 1 , when calculating weights of cross node edges - just sum up weight - no multiplication by 2 
# 

# In[ ]:


# Code below just plots the graph from the example and print the comments

g = igraph.Graph()
g.add_vertices(6)
g.add_edges([[0,1],[1,2],[2,3], [3,0], [0,2] , [0,0],[1,1],[2,2], [3,3]  ])
g.add_edges([  [4,4 ]  , [4,5 ], [5,5 ] ]) 

g.es['weight'] = [4,1,3,1,1,14,4,2,16,26,3,24]
g.es['label'] = g.es['weight']

#layout = g2.layout_circle()
layout = igraph.Layout([ (0, 0) , (1, 0), (1, 1), (0, 1), (1.5, 0.5), (2.5, 0.5) ])
visual_style = {}
visual_style["vertex_color"] = ['green' , 'blue','aqua',  'red', 'green' , 'aqua',]
#visual_style["vertex_label"] = [0,1,2, 3]
visual_style["edge_label_size"] = 30 # [2,2,2]
visual_style["margin"] = 80
print('Example on phase two (aggregation) of the Louvain algorithm'); 
print('Remark: how to read plot - weights 14,4,2,16 are weights of self-loops') ;print();
print('Assume that right graph  is obtained by phase two (aggregation) of the first graph ')
print('I.e. Green+Blue -> Green; Red+Aqua -> Aqua')
print('The weights 26,3,24 are obtained by following rule:')
print('26 = 14 + 4 + 2*4 Pay attention that weight 4 for edge between green-blue is multiplied by 2')
print('24 = 16 + 2 + 2*3 Pay attention that weight 3 for edge between red-aqua is multiplied by 2')
print('3 = 1 + 1 + 1 , when calculating weights of cross node edges - just sum up weight - no multiplication by 2 ')

print(); print();
print('Figure 2')
igraph.plot(g, layout = layout, **visual_style,  bbox = (800,300))


# # 3.1 Modularity gain when an isolated node moves into a community [4 points]
# 
# There exists a simple and fast  way  to calculate modularity change for the case when one node moves from a group to another group. It is key feature of the Louvain algorirhm which makes it fast. 
# The goal of the exerices is to derive that formula. 
# It is very easy if we first slightly rewrite  the formula for modularity. 
# Thus we first rewrite  and then discuss the exercise itself.
# 
# Remark: That exercise is somewhat mathematical technicality, one might omit it if one is looking more for the practical side. The sense of what going on should be clear without explicit formulas - modularity is defined by the huge sum over all nodes, but when one changes the group of only one node, you do not need to recalculate the huge sum - look only on those terms which are related to the group and the node  and recaculation would be much faster, moreover actually pretty nice formula exists, but you might omit it.
# 

# ## Rewriting formula for modularity. New form makes easy to see what happens when nodes change their groups.  
# 
# It is useful to make simple rewriting the modularity formula   (it is borrowed from https://en.wikipedia.org/wiki/Modularity_(networks)#Modularity Formula 4). 
# From that rewritting one immediately gets the solution to the exercise below, i.e. it gives clear look on what happens when one ( or more) nodes change their group. 
# 
# $$
# Modularity  = \frac{1}{(2m)}\sum_{vw} \left[ A_{vw} - \frac{k_v k_w}{(2m)} \right] \delta(c_{v}, c_{w})     = $$ 
# $$  =\sum_{i=1}^{c} (e_{ii} - a_{i}^2) 
# $$
# 
# Where $e_{ii} = \sum_{u \in C_i, v \in C_i} $ , $ a_{i} = \sum_{v \in C_i} \frac{k_v}{2m} $, $C_i$ - the groups/clusters, $u,v$ - nodes. So the equality above is just straight forward (use $ \sum_{v , w } k_v k_w = (\sum_{v } k_v)^2  $). 
# Despite the rewriting above is quite simple, it is extremely useful.
# 
# First let us observe that each term has a clear sense in the formula above: 
# 
# $ e_{ii} $ -  summation of  weights of "inner" edges in group $C_i$ (i.e. those edges having both ends in the same group $C_i$) 
# 
# $a_{i}$  - summation of weights of "total" edges of group $C_i$ (i.e. those having at least one end in $C_i$).   
# 
# Thus, changing notation $ e_{ii} \to \frac{ \Sigma_{i,inner}}{2m} , a_{i} \to \frac{Sigma_{i,total}}{2m}  $ one rewrites: 
# 
# $$
# Modularity  =\sum_{i=1}^{c} (\frac{ \Sigma_{i,inner}}{2m} - (\frac{ \Sigma_{i,total}}{2m})^2) 
# $$
# 
# Such notations are used in the exercise below and in the original paper on the Louvain algorithm. 
# 
# Second: Consider the situation when there is lonely node $i$, which is joining some group $C$. How the modularity would change ? The formula consists of summation over groups, thus  we need to consider only the group C and group of solely node $i$ - the other terms in the sum would remain completely the same before join and after, and so they cancel each other ! That is the key point !
# Thus we would have: 
# 
# $$
# \Delta Modularity  =  (\frac{\Sigma_{inner  ~ C  ~ and  ~  i }}{2m} - (\frac{\Sigma_{ total ~  C  ~  and  ~  i}}{2m})^2) -  
# [ (\frac{ \Sigma_{inner~ C }}{2m} - (\frac{ \Sigma_{total ~ C }}{2m})^2)  +  (\frac{ \Sigma_{inner  ~  i }}{2m} - (\frac{ \Sigma_{total  ~  i } }{2m} )^2 ) ] 
# $$
# 
# The first term comes from the modularity of the joined C and $i$ partition; the other terms from modularity of disjoint C and $i$ partition; all the other terms cancel each other since groups have not changed.
# Now notice that  $(\Sigma_{inner  ~  i } = 0$ because group $i$ consits of node which assumed not to have a self-loop, so it means there is no edge from $i$ to $i$, so it is zero. The other thing to notice is that:
# $ \Sigma_{inner  ~ C  ~ and  ~  i } = \Sigma_{inner  ~ C} +     2*k_{i,in} $ where $k_{i,in}$ is sum weights of edges coming from node $i$ to group $C$, here factor $2$ comes from trivial reason - in definition of modularity there is summation $\sum_{u, v}$ that means we have both pairs  ${u, v}$
# and ${v, u}$ in that summation, so each edge is summed up twice. 
# The third thing to notice is that:
# $ \Sigma_{total  ~ C  ~ and  ~  i } = \Sigma_{total  ~ C} +     k_{i} $ , where
# $k_i$ is the total sum of weight of edges attached to $i$, that is true more or less by defition $\Sigma_{total} = \sum_u k_u $ - so if one adds new node to group one adds one term to sum. 
# 
# After that remarks we arrive to: 
# $$ \Delta Modularity = \bigg[ \frac{\Sigma_{inner~ C} + 2k_{i,in}}{2m} - \bigg(\frac{\Sigma_{total~ C} + k_i}{2m}\bigg)^2 \bigg]-\bigg[\frac{\Sigma_{inner~ C}}{2m} - \bigg(\frac{\Sigma_{total~ C}}{2m}\bigg)^2-\bigg(\frac{k_i}{2m}\bigg)^2\bigg] 
# $$
# 
# Which is exactly the solution to exercise given below.
# 

# ## Exercise 

# Task here  is to prove  formula below. It is immediate consequence of the consideration above. The Figure 2 mentioned here is plotted in the cell below.
# 
# 
# 
# 
# Consider a node i that is in a community all by itself. Let C represent an existing community
# in the graph. Node i feels lonely and decides to move into the community C, we will inspect the
# change in modularity when this happens.
# 
# This situation can be modeled by a graph (Figure 2 - see below) with C being represented by a single node. C
# has a self-edge of weight $\Sigma_{in}$. There is an edge between i and C of weight $k_{i;in}/2$ (to stay consistent
# with the notation of the paper). The total degree of C is $\Sigma_{tot}$ and the degree of i is $k_i$. As always,
# $2m = \sum A_{ij}$ is the sum of all entries in the adjacency matrix. To begin with, C and i are in separate
# communities (colored green and red respectively). **Prove that** the modularity gain seen when i
# merges with C (i.e., the change in modularity after they merge into one community) is given by
# 
# $$ \Delta Q = \bigg[ \frac{\Sigma_{in} + 2k_{i,in}}{2m} - \bigg(\frac{\Sigma_{tot} + k_i}{2m}\bigg)^2 \bigg]-\bigg[\frac{\Sigma_{in}}{2m} - \bigg(\frac{\Sigma_{tot}}{2m}\bigg)^2-\bigg(\frac{k_i}{2m}\bigg)^2\bigg] 
# $$
# 
# Hint: Using the community aggregation step of the Louvain method may make computation easier.
# In practice, this result is used while running the Louvain algorithm (along with a similar related
# result) to make incremental modularity computations much faster.

# ## Figure 2 for exercise above

# In[ ]:


g2 = igraph.Graph()
g2.add_vertices(3)
g2.add_edges([[0,1],[1,2],[2,0], [2,2] ])
g2.es['label'] = ['k_i - (k_i,in/2)   ',
  'Sigma_tot-Sigma_in-(k_i,in/2)','k_i,in/2','   Sigma_in'] # It is misleading to put so much unnecessary info on that graph 

g2.es['label'] = ['',
  '','k_i,in/2','   Sigma_in']

#layout = g2.layout_circle()
layout = igraph.Layout([ (2, 0) , (3, 2), (0, 0), ])
visual_style = {}
visual_style["vertex_color"] = ['red','gray','green']
visual_style["vertex_label"] = ['i','all others','']
visual_style["edge_label_size"] = 10 # [2,2,2]
visual_style["margin"] = 80
#g2.name = 'Node i - red, is joining green group'
print('Node i - red, is joining the green group')
igraph.plot(g2, layout = layout, **visual_style,  bbox = (600,300))


# # 3.2 Louvain algorithm on a 16 node network
# 
# Consider the graph "g" with 16 nodes plotted in the cell below. 
# Current section contains several exercises-questions on the Louvain aglrithm for that graph. 
# 
# Notations are the following.
# The first phase of modularity optimization detects each clique as a single community (giving 4 communities in all).
# After the community aggregation phase, the new network H will have 4 nodes.
# The question will concern that graph H and its descendet.
# 
# 

# ## Figure with 16 nodes graph 
# 
# Create graph with 16 nodes from the task HW1. 
# (Use igraph, since we need self-loops at latter step, and networkx **canNOT** plot self-loops by default). 

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

#print(g)
layout = g.layout_grid( ) # reingold_tilford(root=[2])
visual_style = {}
visual_style["vertex_color"] = ['green' for v in g.vs]
#visual_style["vertex_label"] = range(g.vcount()) 

modularity1 = g.modularity(range(g.vcount() ))# ,  weights=h.es['weight'])
print('Modularity of intial graph, where all nodes in separate groups', modularity1)


igraph.plot(g, layout = layout, **visual_style, bbox = (200,200) )


# ## Figure of the graph "H" (obtained by aggregation step from G)

# In[ ]:


h = igraph.Graph()
h.add_vertices(4)
h.add_edges([[0,1], [1,2],[2,3],[3,0], [0,0], [1,1], [2,2], [3,3]])
igraph.plot(h, bbox = (400,200))


# ## What is the weight of any edge between two distinct nodes in H? [1 point]
# 
# Answer: since there is only 1 edges between groups, new weight on edges between groups will be 1. 

# ## What is the weight of any self-edge in H? [2 point]
# 
# Answer: since there are 6 edges between nodes in each group and since **we must double weights between distinct nodes** - the new weight for self-loops will be 12
# 

# ## What is the modularity of H (with each node in its own community)? [2 points]

# In[ ]:


h = igraph.Graph()
h.add_vertices(4)
h.add_edges([[0,1], [1,2],[2,3],[3,0], [0,0], [1,1], [2,2], [3,3]])
h.es['weight'] = [ 1,1,1,1, 12,12,12,12]
h.es['label'] = h.es['weight']#weights
modularity1 = h.modularity([0,1,2,3],  weights=h.es['weight'])
print('modularity', modularity1)
visual_style= {} ; visual_style["margin"] = 40
igraph.plot(h,**visual_style,  bbox = (400,200))


# ### Modularity of H by networkx. (Each node in its own group). Get the same result as by igraph.

# In[ ]:


import networkx as nx
G = nx.Graph()
G.add_nodes_from([0, 1,2,3])
G.add_edges_from([[0, 1],[1,2],[2,3],[3,0]])
G.add_edge(0, 0, weight = 12)
G.add_edge(1, 1, weight = 12)
G.add_edge(2, 2, weight = 12)
G.add_edge(3, 3, weight = 12)
m1 = nx.algorithms.community.modularity(G, [{0},{1},{2},{3}])
m1 


# ## Figure of graph "J"

# In[ ]:


# Modularity by igraph
j_graph = igraph.Graph()
j_graph.add_vertices(2)
j_graph.add_edges([[0,0], [0,1],[1,1]]) # ,[3,0], [0,0], [1,1], [2,2], [3,3]])
visual_style= {} ; visual_style["margin"] = 40
igraph.plot(j_graph,**visual_style,  bbox = (400,200))


# ## Contract H further - two adjacent nodes in first community, the other two in the second - call result graph "J"
# Spoiler alert: In this network, this is the maximum modularity and the algorithm will terminate here. However, assume that we wanted to contract the graph further into a two node network (call it J) by grouping two adjacent nodes in H into one community and the other two adjacent nodes into another community, and then aggregating (following the same rules of the community aggregation phase).

# ## What is the weight of any edge between two distinct nodes in J? [1 point]
# 
# Answer: 2  - since two nodes go from one group to another 

# ## What is the weight of any self-edge in J? [2 point]
# 
# Answer: 26  = 12 + 12 + 2*1 - sum of weights in the group, where self-edges comes with factor 1, but weight of edge between distinct nodes is multiplied by 2.

# ## What is the modularity of J (with each node in its own community)? [2 points]
# 
# As expected, the modularity of J is less than the modularity of H.

# In[ ]:


# Modularity by igraph
j_graph = igraph.Graph()
j_graph.add_vertices(2)
j_graph.add_edges([[0,0], [0,1],[1,1]]) # ,[3,0], [0,0], [1,1], [2,2], [3,3]])
j_graph.es['weight'] = [ 26,2,26]# ,1, 6,6,6,6]
j_graph.es['label'] = j_graph.es['weight']#weights
modularity1 = j_graph.modularity([0,1],  weights=j_graph.es['weight'])
print(modularity1)
visual_style= {} ; visual_style["margin"] = 40
igraph.plot(j_graph,**visual_style,  bbox = (400,200))


# In[ ]:


# Modularity by networkx
import networkx as nx
G = nx.Graph()
G.add_nodes_from([0, 1])
G.add_edges_from([[0,0],[0, 1],[1,1]])
G.add_edge(0, 0, weight = 26)
G.add_edge(1, 1, weight = 26)
G.add_edge(0, 1, weight = 2)
m1 = nx.algorithms.community.modularity(G, [{0},{1}])
m1


# ## Check that builtin Louvain (and optimal partitioning) will generate partition as it is expected - 4 groups. 

# In[ ]:


import time 

########################################################################
# Cluster by Louvain algorithm 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel
########################################################################
t0 = time.time()
louvain_partition = g.community_multilevel()# weights=graph.es['weight'], return_levels=False)
modularity1 = g.modularity(louvain_partition)#, weights=graph.es['weight'])
print('Finished', time.time()-t0, 'seconds passed')
print("The modularity for igraph-Louvain partition is {}".format(modularity1))
print('Pay attention this modularity is different from modularity of collapsed graph Jbig, we only check partitions are as expected')
#print();
print('Partition info:')
print(louvain_partition)

########################################################################
# Cluster by optimal algorithm (applicable only for small graphs <100 nodes), it would be very slow otherwise 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_optimal_modularity
########################################################################
print();
t0 = time.time()
v = g.community_optimal_modularity() # weights= gra.es["weight"]) 
modularity1 = g.modularity(v)#, weights=graph.es['weight'])
print('Finished', time.time()-t0, 'seconds passed')
print("The modularity for igraph-optimal partition is {}".format(modularity1))
print('Pay attention this modularity is different from modularity of collapsed graph Jbig, we only check partitions are as expected')
#print();
print('Partition info:')
print(v) 


# # 3.3 Louvain algorithm on a 128 node network
# 
# Now consider a larger version of the same network, with 32 cliques of 4 nodes each (arranged in a ring as earlier); call this network Gbig. Again, assume all the edges have same weight value 1,  and there exists exactly one edge between any two adjacent cliques. The first phase of modularity
# optimization, as expected, detects each clique as a single community. After aggregation, this forms a new network Hbig with 32 nodes.
# 
# 

# In[ ]:


g = igraph.Graph()
g.add_vertices(128)

nodes = np.array([0,1,4,5])
vertical_start = 0
for vertical_start in range(16):# [0,16,32,48,64,80,96,112]:
  vertical_start *= 8
  for k in [0,2]:#,2,4,6,8]:
    for i in nodes+k+vertical_start:
      for j in nodes+k+vertical_start:
        if i<=j: continue 
        g.add_edge(i, j)

for x_shift in [0,3]:
  for y_shift in list( range(4,117,8)):
    g.add_edge(x_shift + y_shift,x_shift + y_shift+8 )
g.add_edge(1,2)
g.add_edge(125,126)

#print(g)
layout = g.layout_grid(width = 4 )# ()     ) # reingold_tilford(root=[2])
visual_style = {}
visual_style["vertex_color"] = ['green' for v in g.vs]
#visual_style["vertex_label"] = range(g.vcount()) 
visual_style["vertex_size"] = 5

modularity1 = g.modularity(range(g.vcount() ))# ,  weights=h.es['weight'])
print('Modularity of intial graph, where all nodes in separate groups', modularity1)

igraph.plot(g, layout = layout, **visual_style, bbox = (500,800) )


# ## What is the weight of any edge between two distinct nodes in Hbig? [1 point]
# 
# Answer: 1 , since only 1 node between corresponding cliques 

# ## What is the weight of any self-edge in Hbig? [2 point]
# 
# Answer: 12, since there are 6 edges, but they counted twice, since they are between distinct nodes, only self-loops are counted with factor 2 

# ## What is the modularity of Hbig (with each node in its own community)? [2 points]

# In[ ]:


h = igraph.Graph()
n_nodes = 32
h.add_vertices(n_nodes)
for i in range(n_nodes-1):
  h.add_edge(i, i+1)
h.add_edge(n_nodes-1, 0)
for i in range(n_nodes):
  h.add_edge(i, i)
h.es['weight'] = list(np.ones(n_nodes)) +list(12*np.ones(n_nodes)) 
h.es['label'] = h.es['weight']

#layout = g.layout_grid(width = 2 )# ()     ) # reingold_tilford(root=[2])
layout = h.layout_circle()
visual_style = {}
visual_style["vertex_color"] = ['green' for v in h.vs]
visual_style["vertex_size"] = 5

modularity1 = h.modularity(range(h.vcount() )  ,  weights=h.es['weight'])
print('Modularity of intial graph, where all nodes in separate groups', modularity1)

igraph.plot(h, layout = layout, **visual_style, bbox = (800,500) )


# ## Next step of Louvain collapse to 16 node graph
# 
# After what we saw in the earlier example, we would expect the algorithm to terminate here.  However (spoiler alert again), that doesn't happen and the algorithm proceeds. The next phase of modularity optimization groups Hbig into 16 communities with two adjacent nodes from Hbig in  each community. Call the resultant graph (after community aggregation) Jbig.

# ## What is the weight of any edge between two distinct nodes in Jbig? [1 point]
# 
# Answer: 1 , since only 1 edge between groups

# ## What is the weight of any self-edge in Jbig? [2 point]
# 
# Answer: 26 = 2*2+12+12, edges between nodes comes with factor 2, self-loops with factor 1 

# ## What is the modularity of Jbig (with each node in its own community)? [2 points]
# 
# This particular grouping of communities corresponds to the maximum modularity in this network
# (and not the one with one clique in each community). The community grouping that maximizes the
# modularity here corresponds to one that would not be considered intuitive based on the structure
# of the graph.

# In[ ]:


jbig = igraph.Graph()
n_nodes = 16
jbig.add_vertices(n_nodes)
for i in range(n_nodes-1):
  jbig.add_edge(i, i+1)
jbig.add_edge(n_nodes-1, 0)
for i in range(n_nodes):
  jbig.add_edge(i, i)
jbig.es['weight'] = list(np.ones(n_nodes)) +list(26*np.ones(n_nodes)) 
jbig.es['label'] = jbig.es['weight']


#layout = g.layout_grid(width = 2 )# ()     ) # reingold_tilford(root=[2])
layout = jbig.layout_circle()
visual_style = {}
visual_style["vertex_color"] = ['green' for v in jbig.vs]
visual_style["vertex_size"] = 5

modularity1 = jbig.modularity(range(jbig.vcount() )  ,  weights=jbig.es['weight'])
print('Modularity of intial graph, where all nodes in separate groups', modularity1)

igraph.plot(jbig, layout = layout, **visual_style, bbox = (600,300) )


# ## Check by built-in Louvain (and optimal partitioning) that they generate same partition

# In[ ]:


import time 

########################################################################
# Cluster by Louvain algorithm 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel
########################################################################
t0 = time.time()
louvain_partition = g.community_multilevel()# weights=graph.es['weight'], return_levels=False)
modularity1 = g.modularity(louvain_partition)#, weights=graph.es['weight'])
print('Finished', time.time()-t0, 'seconds passed')
print("The modularity for igraph-Louvain partition is {}".format(modularity1))
print('Pay attention this modularity is different from modularity of collapsed graph Jbig, we only check partitions are as expected')
#print();
print('Partition info:')
print(louvain_partition)

########################################################################
# Cluster by optimal algorithm (applicable only for small graphs <100 nodes), it would be very slow otherwise 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_optimal_modularity
########################################################################
print();
t0 = time.time()
v = g.community_optimal_modularity() # weights= gra.es["weight"]) 
modularity1 = g.modularity(v)#, weights=graph.es['weight'])
print('Finished', time.time()-t0, 'seconds passed')
print("The modularity for igraph-optimal partition is {}".format(modularity1))
print('Pay attention this modularity is different from modularity of collapsed graph Jbig, we only check partitions are as expected')
#print();
print('Partition info:')
print(v) 


# ## What just happened? [1 point]
# 
# Explain (in a few lines) why you think the algorithm behaved the way it did for the larger network (you don't need to prove anything rigorously, a rough argument will do). In other words, what might have caused modularity to be maximized with an unintuitive community grouping for the
# larger network?
# 
# 
# Answer. Well, not very clear, but less us try to say something.  First it is clear, we get nodes with high weight for self-loops, and low weight on cross-node edges - that means that such nodes are good enough clusters by themselves, but it is clear are they good enough to stop algo or not. 
# However what seems to be reasoable - is modularity depends  on the size of the circle we get. If we get big enough circle Louvain would try to shrink in further if circle is small Louvain stops.
# 
# There is experiment below with 256 1024 ... similar graphs, for 256 we stop at the same level as 128. So we see number of final clusters grows that somewhat supports our claim. Well, it is better to do another experiment but, may be next time.... 
# 
# 

# ### Experiment - construct even bigger graphs with 256, 512, 1024 ... nodes, by the same pattern as "g128" - ring of 4-cliques. Apply Louvain and look how many groups  we get
# 
# Results: 
# 256 - 16 groups the same as for 128  (modularity 0.901785714285714 )
# 
# 512 - 32 groups (modularity  0.9330357142857145)
# 
# 1024 - 32 groups (modularity 0.9508928571428574 )
# 
# 2048 - 64 groups (modularity 0.9665178571428559 )
# 
# 
# 

# In[ ]:


g = igraph.Graph()
n_nodes = 512
g.add_vertices(n_nodes)

nodes = np.array([0,1,4,5])
vertical_start = 0
for vertical_start in range(int(n_nodes/8)):# [0,16,32,48,64,80,96,112]:
  vertical_start *= 8
  for k in [0,2]:#,2,4,6,8]:
    for i in nodes+k+vertical_start:
      for j in nodes+k+vertical_start:
        if i<=j: continue 
        g.add_edge(i, j)

for x_shift in [0,3]:
  for y_shift in list( range(4,n_nodes-11,8)):
    g.add_edge(x_shift + y_shift,x_shift + y_shift+8 )
g.add_edge(1,2)
g.add_edge(n_nodes-3,n_nodes-2)

#print(g)
layout = g.layout_grid(width = 4 )# ()     ) # reingold_tilford(root=[2])
visual_style = {}
visual_style["vertex_color"] = ['green' for v in g.vs]
visual_style["vertex_label"] = range(g.vcount()) 
visual_style["vertex_size"] = 5

modularity1 = g.modularity(range(g.vcount() ))# ,  weights=h.es['weight'])
print('Modularity of intial graph, where all nodes in separate groups', modularity1)

import time 

########################################################################
# Cluster by Louvain algorithm 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel
########################################################################
t0 = time.time()
louvain_partition = g.community_multilevel()# weights=graph.es['weight'], return_levels=False)
modularity1 = g.modularity(louvain_partition)#, weights=graph.es['weight'])
print('Finished', time.time()-t0, 'seconds passed')
print("The modularity for igraph-Louvain partition is {}".format(modularity1))
print('Pay attention this modularity is different from modularity of collapsed graph Jbig, we only check partitions are as expected')
#print();
print('Partition info:')
print(louvain_partition)

igraph.plot(g, layout = layout, **visual_style, bbox = (500,1800) )


# In[ ]:


import time 

########################################################################
# Cluster by Louvain algorithm 
# https://igraph.org/python/doc/igraph.Graph-class.html#community_multilevel
########################################################################
t0 = time.time()
louvain_partition = g.community_multilevel()# weights=graph.es['weight'], return_levels=False)
modularity1 = g.modularity(louvain_partition)#, weights=graph.es['weight'])
print('Finished', time.time()-t0, 'seconds passed')
print("The modularity for igraph-Louvain partition is {}".format(modularity1))
print('Pay attention this modularity is different from modularity of collapsed graph Jbig, we only check partitions are as expected')
#print();
print('Partition info:')
print(louvain_partition)


# In[ ]:





# In[ ]:




