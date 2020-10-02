#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import networkx as nx


# ### Find number of neighbours & Node Degree

# In[ ]:


# find neibhours for a given node


# In[ ]:


fb_network = pd.read_csv("../input/fb_network_sub.csv")


# In[ ]:


G = nx.from_pandas_edgelist(fb_network, source="source_node", target="destination_node")


# In[ ]:


# how many friends does user 1 has
len(list(G.neighbors(741)))


# In[ ]:


# same as 
val1 = fb_network.loc[fb_network["source_node"] == 741,"destination_node"].values
val2 = fb_network.loc[fb_network["destination_node"] == 741, "source_node"].values


# In[ ]:


len(set(list(np.append(val1,val2))))


# ### Common network analysis 

# <b> Find nodes with influence </b>

# In[ ]:


## Node centrality
fb_deg_cen = nx.degree_centrality(G)


# In[ ]:


type(fb_deg_cen)


# In[ ]:


fb_deg_cen[741]


# In[ ]:


# same as 
len(list(G.neighbors(741)))/(len(list(G.nodes())) - 1)


# In[ ]:


fb_deg_cen = pd.DataFrame([fb_deg_cen]).T


# In[ ]:


fb_deg_cen["user_id"] = fb_deg_cen.index
fb_deg_cen = fb_deg_cen.rename(columns={0:"deg_cen"})


# In[ ]:


fb_deg_cen.head()


# In[ ]:


fb_deg_cen[fb_deg_cen["user_id"] == 741]


# In[ ]:


fb_deg_cen.sort_values(by="deg_cen", ascending=False).head()


# In[ ]:


len(list(G.neighbors(1492489)))/(len(list(G.nodes())) - 1)


# <b> Shortest path algorithm </b>

# In[ ]:


## find few unconnected people in the network
count = 0
for e in nx.non_edges(G):
    print(e)
    count = count + 1
    if count > 10:
        break


# In[ ]:


nx.shortest_path(G, source = 786432, target = 1048579)


# ### Link Prediction (or Recommendation)

# <b> Common Neighbors Method </b>

# In[ ]:


## common neighbours
## find common neighbours for a paticular user say user no 4850
len(list(G.neighbors(4850)))


# In[ ]:


# find list of key users the user 4850 is not currently connected to
key_users = list(set(fb_network["source_node"].values))
unconnected_users = [x for x in key_users if x not in list(G.neighbors(4850))]


# In[ ]:


len(unconnected_users)


# In[ ]:


# find common neighbours
common_neighbors = []
for user_id in unconnected_users:
    common_neighbors.append(len(list(nx.common_neighbors(G, 4850, user_id))))


# In[ ]:


common_neighbors_df = pd.DataFrame({"user_id": 4850, 
                                    "unconnctd_user_id": unconnected_users, "common_neighbors": common_neighbors})


# In[ ]:


common_neighbors_df.sort_values(by = "common_neighbors", ascending=False).head()


# In[ ]:


n1 = set(G.neighbors(4850))
n2 = set(G.neighbors(1492489))


# In[ ]:


len(n1.intersection(n2))


# <b> Using Jaccard Similarity </b>

# In[ ]:


list(nx.jaccard_coefficient(G, [(4850,1492489)]))


# In[ ]:


len(n1.intersection(n2))/len(n1.union(n2))


# In[ ]:


jac_sim = [list(nx.jaccard_coefficient(G, [(4850,user_id)]))[0] for user_id in unconnected_users]


# In[ ]:


jac_sim_df =  pd.DataFrame.from_records(jac_sim,columns=["user_id","unconnctd_user_id","jac_sim"])


# In[ ]:


jac_sim_df.sort_values(by="jac_sim", ascending=False).head()


# In[ ]:


n1 = set(G.neighbors(4850))
n2 = set(G.neighbors(245648))


# In[ ]:


print("intersection -" ,len(n1.intersection(n2)))
print("union -" , len(n1.union(n2)))


# In[ ]:


40/1026


# <b> Resource Allocation Index </b>

# In[ ]:


list(nx.resource_allocation_index(G,[(4850,1492489)]))


# In[ ]:


res_alloc_sim = [list(nx.resource_allocation_index(G, [(4850,user_id)]))[0] for user_id in unconnected_users]


# In[ ]:


res_alloc_sim_df =  pd.DataFrame.from_records(res_alloc_sim,columns=["user_id","unconnctd_user_id","res_alloc_sim"])


# In[ ]:


res_alloc_sim_df.sort_values(by="res_alloc_sim", ascending=False).head()

