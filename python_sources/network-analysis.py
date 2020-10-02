#!/usr/bin/env python
# coding: utf-8

# # Network Analysis: Important stations of public transportation net
# 
# **What?** <br>
# In this kernel, we will find out which stations could turn out to be some kind of "bootlenecks". Think of ticket terminal breakdowns, overcrowded escalators or train cancellation at a station. Knowing the bottlenecks of a subway or any other transportation network could help to better maintain frequently travelled stations by knowing at which edge a failure would harm most.  <p>
# 
# **Graph-theory** <br>
# The graph-theoretical concept beyond this considerations is called "betweenness centrality". It is the ratio between the "number of shortest paths trough a node" and "all possible shortest paths" in a graph. <p>
# 
# **Interpretation of betweenness centrality** <br>
# The measure captures the attractiveness of travel paths between each pair of nodes within a network. It reveals frequently travelled hubs by telling us to what extent a station is part of central interesections.  The higher the betweenness centrality of a node, the more of a "bootleneck" this station is, since many attractive travel opportunities are leading trough this node.  <p>

# ### 1. Import packages, load and prepare data

# In[1]:


#import relevant packages
import matplotlib.pyplot as plt
import numpy as np 
import networkx as nx #package for creating, manipulating, and studying networks
import pandas as pd 


# In[40]:


#load data
stations = pd.read_csv('../input/Vienna subway.csv', sep=";")
print(stations.sample(5))
print("The dataset contains", stations.shape[0], "rows and", stations.shape[1], "columns.")

#define nodes of bikesharing network
network = nx.from_pandas_edgelist(stations, 'Start', 'Stop')

#get amount of stations and paths
print('The subway in vienna consists of', len(network.nodes), 'different stations (nodes) and', 
      len(network.edges), 'connections (edges) between the stations.')


# ### 3. Find out top 5 possible bottlenecks 

# In[41]:


#compute betweenness centrality of stations
bw_centrality = nx.betweenness_centrality(network)

#convert items of bw_ccentrality dictionary into dataframe
bw_centrality = pd.DataFrame(list(bw_centrality.items()))

#rename columns of data frame
bw_centrality.columns = ["station", "betweenness_centrality"]

#show head rows of data frame
bw_centrality.head()

#get top 5 bottleneck stations
bw_centrality.sort_values(by="betweenness_centrality", ascending=False).iloc[:5,:]


# ### 4. Plausibility check and intepretation

# Go and assess yourself whether Karlsplatz, Schwedenplatz, Praterstern, Stephansplatz and Laengenfeldgasse should be considered as a bottleneck of subway transportation in vienna and expecially taken care of!
# ![](http://homepage.univie.ac.at/horst.prillinger/ubahn/m/largemap.png)
# 
