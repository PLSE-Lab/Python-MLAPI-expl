#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import networkx as nx
from networkx.algorithms import bipartite
from subprocess import check_output
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv("../input/steam-200k.csv", header=None, index_col=None, names=['UserID', 'Game', 'Action', 'Hours', 'Other'])
df.head()


# In[3]:


dplay = df.loc[df['Action'] == 'play']
dplay.head()


# In[4]:


dplay.isnull().sum()


# In[5]:


dplay.info()


# In[6]:


B = nx.from_pandas_edgelist(df=dplay, create_using=nx.DiGraph(), source='UserID', target='Game')
len(B.edges)


# In[7]:


G = nx.algorithms.bipartite.projected_graph(B=B, nodes=dplay.UserID, multigraph=True)
len(G.edges)


# In[8]:


B.degree('Dota 2')


# In[9]:


dbuy = df.loc[df['Action'] == 'purchase']
dbuy.head()


# In[10]:


dbuy.info()


# In[11]:


B = nx.from_pandas_edgelist(df=dbuy, source='UserID', target='Game')
print(nx.info(B))


# In[12]:


players = set(dbuy.UserID)
len(players)


# In[13]:


G = bipartite.projected_graph(B=B, nodes=players, multigraph=False)
print(nx.info(G))


# In[14]:


degrees = list()
for deg in list(G.degree):
    degrees.append(deg[1])

hist = nx.degree_histogram(G)
sns.set_style("darkgrid")
plt.plot(hist)


# In[15]:


from collections import Counter
import collections
import scipy as sp
from scipy import stats
import matplotlib.pyplot as plt

in_degrees = degrees
in_values = sorted(set(in_degrees))
in_hist = [in_degrees.count(x) for x in in_values]
plt.figure()
plt.grid(True)
plt.plot(in_values, in_hist, 'ro-') # in-degree
plt.legend(['In-degree', 'Out-degree'])
plt.xlabel('Degree')
plt.ylabel('Number of nodes')
plt.xlim([0, 2*10**2])
plt.show()


# In[16]:


degree_sequence = sorted([d for n, d in G.degree()], reverse=True)  # degree sequence
# print "Degree sequence", degree_sequence
degreeCount = collections.Counter(degree_sequence)

sns.set_style('ticks')

deg, cnt = zip(*degreeCount.items())

fig, ax = plt.subplots()
plt.bar(deg, cnt, width=0.80, color='b')

plt.title("Degree Histogram")
plt.ylabel("Count")
plt.xlabel("Degree")
ax.set_xticks([d + 0.4 for d in deg])
ax.set_xticklabels(deg)

# draw graph in inset
plt.axes([0.4, 0.4, 0.5, 0.5])
# Gcc = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
# pos = nx.spring_layout(G)
plt.axis('off')
# nx.draw_networkx_nodes(G, pos, node_size=20)
# nx.draw_networkx_edges(G, pos, alpha=0.4)

plt.show()


# In[ ]:


matrix = nx.to_numpy_matrix(G)
matrix = matrix.astype('short')
np.savetxt('adj.csv', matrix, delimiter=',', fmt='%1d')


# In[ ]:




