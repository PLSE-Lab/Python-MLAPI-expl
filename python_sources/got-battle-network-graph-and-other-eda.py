#!/usr/bin/env python
# coding: utf-8

# # Game of Thrones - Network Graph Portion of EDA
# ### I had pasted in most of my notebook--- then, I swiped back and lost it, so here's just
# ### the network graph portion

# ### Westeros has a complex social structure, and many people there have a
# ### murderous bent. Let's see where loyalties lie in battle. 

# In[ ]:


import pandas as pd
import glob
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


files = glob.glob('../input/*.csv')


# In[ ]:


# Note--- sometimes the order changes on different runs of glob
files


# In[ ]:


# Read the files in
for x in files:
    if x.find('battles') != -1:
        bt_df = pd.read_csv(x)
    elif x.find('-predictions') != -1:
        cp_df = pd.read_csv(x)
    else:
        cd_df = pd.read_csv(x)


# In[ ]:


# Check the shape
bt_df.shape


# In[ ]:


# Graph analysis. 
# Get these back into single king and ally columns
aka1 = bt_df[['attacker_king', 'attacker_1']]
aka2 = bt_df[['attacker_king', 'attacker_2']]
aka3 = bt_df[['attacker_king', 'attacker_3']]
aka4 = bt_df[['attacker_king', 'attacker_4']]
dkd1 = bt_df[['defender_king', 'defender_1']]
dkd2 = bt_df[['defender_king', 'defender_2']]
dkd3 = bt_df[['defender_king', 'defender_3']]
dkd4 = bt_df[['defender_king', 'defender_4']]


# In[ ]:


# Make a list of the df's made in previous cell
mat_list = [aka1, aka2, aka3, aka4, dkd1, dkd2, dkd3, dkd4]


# In[ ]:


# Change the column names
for x in mat_list:
    x.columns = ['king', 'ally']


# In[ ]:


# Concat the frames
bk = pd.concat(mat_list,axis=0)


# In[ ]:


# Drop the null allies, since they don't represent an alliance
bk.dropna(inplace=True)


# In[ ]:


# Check the shape to make sure it worked
bk.shape


# In[ ]:


# Make a crosstab
mx_df = pd.crosstab(bk.king, bk.ally)


# In[ ]:


# Make a kings x allies copy
kings_x_allies = mx_df.copy()


# In[ ]:


# ... and an allies x kings copy, transposed.
allies_x_kings = mx_df.T.copy()


# In[ ]:


print(kings_x_allies.shape)
print(allies_x_kings.shape)


# In[ ]:


# Make the adjacency matrix
ally_adj = allies_x_kings.dot(kings_x_allies).copy()


# In[ ]:


ally_adj.shape


# In[ ]:


G = nx.from_numpy_matrix(ally_adj.values)
G = nx.relabel_nodes(G, dict(enumerate(ally_adj.columns)))


# In[ ]:


plt.figure(figsize=(40,40))
pos=nx.spring_layout(G, iterations=500, scale=5, k=.3)
nx.draw_networkx_labels(G,pos,fontsize=8)
font = {'fontname'   : 'Helvetica',
            'color'      : 'k',
            'fontweight' : 'bold',
            'fontsize'   : 32}
plt.title("Game of Thrones - Kings' Allies", font)


nx.draw_networkx_nodes(G,pos,node_color='b',alpha=0.4)
nx.draw_networkx_edges(G,pos,alpha=0.4,node_size=0,width=1,edge_color='k')
plt.axis('off')
plt.show()


# In[ ]:


# Determine centrality for an ordered list of who best has their king's back during a battle
centrality=nx.eigenvector_centrality(G)


# In[ ]:


loyalty_list = []
for node in centrality:
    loyalty_list.append((node,centrality[node]))
    
sorted_loyalty_list = loyalty_list.sort(key=lambda x: x[1])


# In[ ]:


# Interesting. The Kingslayer is at the top of this list.
sorted(loyalty_list, key=lambda x: x[1], reverse=True)


# In[ ]:


# Now, let's switch and see what the king look like
king_adj = kings_x_allies.dot(allies_x_kings).copy()


# In[ ]:


king_adj.shape


# In[ ]:


K = nx.from_numpy_matrix(king_adj.values)
K = nx.relabel_nodes(K, dict(enumerate(king_adj.columns)))


# In[ ]:


plt.figure(figsize=(40,40))
pos=nx.spring_layout(K, iterations=500, scale=5, k=.3)
nx.draw_networkx_labels(K,pos,fontsize=8)
font = {'fontname'   : 'Helvetica',
            'color'      : 'k',
            'fontweight' : 'bold',
            'fontsize'   : 32}
plt.title("Game of Thrones - Allies' Kings", font)


nx.draw_networkx_nodes(K,pos,node_color='b',alpha=0.4)
nx.draw_networkx_edges(K,pos,alpha=0.4,node_size=0,width=1,edge_color='k')
plt.axis('off')
plt.show()


# In[ ]:


centrality=nx.eigenvector_centrality(K)


# In[ ]:


king_list = []
for node in centrality:
    king_list.append((node,centrality[node]))
    
sorted_king_list = king_list.sort(key=lambda x: x[1])


# In[ ]:


# And now for the sorted list of kings
sorted(king_list, key=lambda x: x[1], reverse=True)


# In[ ]:


# More EDA has been done on the battle file... more to come!
# TODO: 
#    - Look for sourcing in the books and book companion guides to flesh out this dataset with
#      info on Targaryen battles
#    - Use that information and differences in these graphs with subsequent Targaryen graphs
#      to try to predict which houses will support Daenaerys Targaryen when she returns to 
#      Westeros, and if there will be a split in support for House Baratheon


# In[ ]:




