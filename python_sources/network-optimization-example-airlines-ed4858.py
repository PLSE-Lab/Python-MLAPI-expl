#!/usr/bin/env python
# coding: utf-8

# # An example of Network Optimization (Airlines)

# ## Import the *libraries* and *dataset*

# In[ ]:


# import the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

# load the dataset
data = pd.read_csv('../input/airlines_network_optimization.csv') # download the csv file in your local directory and play with it.


# ## Preprocessing the dataset/ Feature Engineering

# ### Creat two columns for data name 'std' ( Scheduled time of departure ) & 'sta' (Scheduled time of arrival)
# 
# ** converting sched_dep_time to 'std' - Scheduled time of departure**
# * To convert *'sched_dep_time'* to* 'std'* column take data from coloum *'sched_dep_time'* data and read as  string datatype.
# * Use *'str.rplace'* to replace first two digits of the data with ' '. Here ' ' means replace nothing.
# * Add a colon *':'* after that.
# * Use* 'str.extract'* to extract last two digits.
# * Use* 'expand=False' *which means a pattern with one group will return a Series.
# * Add :00 after that.
# * After converting it will look like somthing like this :  1915 will become 19:15:00
# 
# **converting sched_arr_time to 'sta' - Scheduled time of arrival**
# * It is as similar as the Scheduled time of departure process.
# 
# ### Creat two columns for data name 'atd'(actual time of departure) & 'ata'(actual time of arrival)
#  
# ** converting dep_time to 'atd' - Actual time of departure**
# * To convert *'dep_time'* to *'atd*' column take data from coloum *'dep_time*' data as int64 datatype and read as  string datatype.
# *  Use *'fillna(0)' *which means replace all Na elements with zero.
# * Use* 'str.rplace'* to replace first two digits of the data with ' '. Here ' ' means replace nothing.
# * Add a colon *':'* after that.
# * Use* 'str.extract'* to extract last two digits.
# * Use *'expand=False'* which means a pattern with one group will return a Series.
# * Add :00 after that.
# * After converting it will look like somthing like this :  1907 will become 19:07:00
# 
# **converting arr_time to 'ata' - Actual time of arrival**
# * It is as similar as the Actual time of departure process.
# 

# In[ ]:


# data.shape 
# converting sched_dep_time to 'std' - Scheduled time of departure
data['std'] = data.sched_dep_time.astype(str).str.replace('(\d{2}$)', '') + ':' + data.sched_dep_time.astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
# converting sched_arr_time to 'sta' - Scheduled time of arrival
data['sta'] = data.sched_arr_time.astype(str).str.replace('(\d{2}$)', '') + ':' + data.sched_arr_time.astype(str).str.extract('(\d{2}$)', expand=False) + ':00'

# converting dep_time to 'atd' - Actual time of departure
data['atd'] = data.dep_time.fillna(0).astype(np.int64).astype(str).str.replace('(\d{2}$)', '') + ':' + data.dep_time.fillna(0).astype(np.int64).astype(str).str.extract('(\d{2}$)', expand=False) + ':00'
# converting arr_time to 'ata' - Actual time of arrival
data['ata'] = data.arr_time.fillna(0).astype(np.int64).astype(str).str.replace('(\d{2}$)', '') + ':' + data.arr_time.fillna(0).astype(np.int64).astype(str).str.extract('(\d{2}$)', expand=False) + ':00'


# ### Creat 'date' column & remove unecessary columns
# * To creat date take data from the column of* 'year, month , day'* from each and every row and convert it into datetime.
# * Use *'data.drop'* to remove  the existing column of *'year, month, day'*.

# In[ ]:


data['date'] = pd.to_datetime(data[['year', 'month', 'day']])
# finally we drop the columns we don't need
data = data.drop(columns = ['year', 'month', 'day'])


# ## Formulate the Network which will be optimized

# ### Creat edgelist
# * *NetworkX* is a Python package for the creation, manipulation, and study of the structure, dynamics, and functions of complex networks.
# * Return a graph from a list of edges where source is* 'origin'* and target is *'dest'* of data containing edgelist.
# * Use  *'edge_attr=True'* which means all of the remaining columns will be added.

# In[ ]:


import networkx as nx
FG = nx.from_pandas_edgelist(data, source='origin', target='dest', edge_attr=True,)
# detail documentation of networkx https://networkx.github.io/documentation/networkx-1.7/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html


# ### Draw the graph using networkx
# * Use *'nx.draw_networkx'* which means draw the graph G using Matplotlib.
# * Use* 'with_labels=True'* which  means there will be labels of all nodes.
# * Use *'node_size=600'* which means size of the node is 600.
# * Use *'node_color='y'*' which means color of the node is y=yellow.

# In[ ]:


FG.nodes()
FG.edges()
nx.draw_networkx(FG, with_labels=True,node_size=600, node_color='y') # Quick view of the Graph. As expected we see 3 very busy airports


# * use* 'nx.algorithms.degree_centrality(FG)'* to compute the degree centrality for nodes of FG and returns a dictionary of nodes with degree centrality as the value.
# * Use *'nx.density(FG)'* to compute the average edge density of the FG graph. The density is 0 for a graph without edges and 1 for a complete graph. 
# * Use* 'nx.average_shortest_path_length(FG)'* to return the average shortest path length for all paths in the FG graph.
# * Use *'nx.average_degree_connectivity(FG)'* to compute the average degree connectivity of FG graph.

# In[ ]:


nx.algorithms.degree_centrality(FG) # Notice the 3 airports from which all of our 100 rows of data originates
nx.density(FG) # Average edge density of the Graphs
nx.average_shortest_path_length(FG) # Average shortest path length for ALL paths in the Graph
nx.average_degree_connectivity(FG) # For a node of degree k - What is the average of its neighbours' degree?


# ## Shortest path between JFK to DFW

# * Find and print all the possible paths available from 'JAX' to 'DFW.
# * *Dijkstra's algorithm* is an algorithm for finding the shortest paths between nodes in a graph.
# * Use *'nx.dijkstra_path'* to returns the shortest path from source='JAX' to target='DFW' in the FG graph.
# * Calculate dijkstra path weighted by airtime frome 'JAX' to 'DFW'.
# 

# In[ ]:


# Let us find all the paths available
for path in nx.all_simple_paths(FG, source='JAX', target='DFW'):
 print(path)
# Let us find the dijkstra path from JAX to DFW.
# You can read more in-depth on how dijkstra works from this resource - https://courses.csail.mit.edu/6.006/fall11/lectures/lecture16.pdf
dijpath = nx.dijkstra_path(FG, source='JAX', target='DFW')
dijpath
# Let us try to find the dijkstra path weighted by airtime (approximate case)
shortpath = nx.dijkstra_path(FG, source='JAX', target='DFW', weight='air_time')
shortpath


# Note:  this code is inspired by www.analyticsvidhya.com /blog/2018/04/introduction-to-graph-theory-network-analysis-python-codes
#   /?utm_source=feedburner&utm_medium=email&utm_campaign=Feed%3A+AnalyticsVidhya+%28Analytics+Vidhya%29
# 
