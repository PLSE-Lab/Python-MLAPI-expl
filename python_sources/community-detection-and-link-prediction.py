#!/usr/bin/env python
# coding: utf-8

# # Link Prediction Using Travian Networks

# In[83]:


# Setting packages import
import os
import pandas as pd
import networkx as nx
import community
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 1- Setup

# In[84]:


# Setting paths
messages_directory_path = "../input/messages-network-csv/Messages-Network-CSV/"
messages_directory = os.listdir(messages_directory_path)
trades_directory_path = "../input/trades-network-csv/Trades-Network-CSV/"
trades_directory = os.listdir(trades_directory_path)
raids_directory_path = "../input/attacks-network-csv/Attacks-Network-CSV/"
raids_directory = os.listdir(raids_directory_path)


# ## 2- Read different networks stats

# In[85]:


# Messages
all_edges = []
all_nodes = []

edges = []
nodes = []
# Read each network snapshot for 30 days
for file_name in messages_directory:
    network_df = pd.read_csv(messages_directory_path + file_name)
    network_df.columns = ['timestamp', 'sourceid', 'destinationid']
    message_graph = nx.from_pandas_dataframe(network_df, 'sourceid', 'destinationid')
    # Get number of nodes and edges for each network snapshot
    edges.append(message_graph.number_of_edges())
    nodes.append(message_graph.number_of_nodes())
all_edges.append(edges)
all_nodes.append(nodes)


# In[86]:


# Trades
edges = []
nodes = []
# Read each network snapshot for 30 days
for file_name in trades_directory:
    network_df = pd.read_csv(trades_directory_path + file_name)
    network_df.columns = ['timestamp', 'sourceid', 'destinationid']
    trade_graph = nx.from_pandas_dataframe(network_df, 'sourceid', 'destinationid')
    # Get number of nodes and edges for each network snapshot
    edges.append(trade_graph.number_of_edges())
    nodes.append(trade_graph.number_of_nodes())
all_edges.append(edges)
all_nodes.append(nodes)


# In[87]:


# Attacks
edges = []
nodes = []
# Read each network snapshot for 30 days
for file_name in raids_directory:
    network_df = pd.read_csv(raids_directory_path + file_name)
    network_df.columns = ['timestamp', 'sourceid', 'destinationid']
    raids_graph = nx.from_pandas_dataframe(network_df, 'sourceid', 'destinationid')
    # Get number of nodes and edges for each network snapshot
    edges.append(raids_graph.number_of_edges())
    nodes.append(raids_graph.number_of_nodes())
all_edges.append(edges)
all_nodes.append(nodes)


# ## 3- Stats comparison and visualization

# ### 3 (a) - User engagement

# In[88]:


plt.plot(all_nodes[0])
plt.plot(all_nodes[1])
plt.plot(all_nodes[2])
plt.xlabel('Days')
plt.ylabel('Count')
plt.legend(('Messages', 'Trades', 'Attacks'), loc = 'upper right')


# ### 3 (b) - Frequency of connections

# In[89]:


plt.plot(all_edges[0])
plt.plot(all_edges[1])
plt.plot(all_edges[2])
plt.xlabel('Days')
plt.ylabel('Count')
plt.legend(('Messages', 'Trades', 'Attacks'), loc = 'upper right')


# ## 4- Unsupervised Link Prediction

# In[92]:


""" There are plenty of community detection and link prediction algorithms 
implemented in networkx. For more information please visit:
https://bit.ly/2JN89D8
https://bit.ly/2FzOiEZ
""" 
adamic_adar_predictions = nx.adamic_adar_index(message_graph)
preferential_attachment_predictions = nx.preferential_attachment(message_graph)

# Uncomment in case you want to see prediction results for every node pair
#for u, v, p in adamic_adar_predictions:
#    print('(%d, %d) -> %d' % (u, v, p))


# ## 5- Community Detection

# In[93]:


#first compute the best partition
partition = community.best_partition(message_graph)

#drawing
size = float(len(set(partition.values())))
pos = nx.spring_layout(message_graph)
count = 0.
for com in set(partition.values()) :
    count = count + 1.
    list_nodes = [nodes for nodes in partition.keys()
                                if partition[nodes] == com]
    nx.draw_networkx_nodes(message_graph, pos, list_nodes, node_size = 20,
                                node_color = str(count / size))

nx.draw_networkx_edges(message_graph, pos, alpha=0.5)
plt.show()

