#!/usr/bin/env python
# coding: utf-8

# 

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn

df = pd.read_csv('../input/results_by_booth_2015 - english - v3.csv', encoding='iso-8859-1')
votes=df.select_dtypes(include=[np.number])
votes=votes.drop(votes.columns[range(0,6)],axis=1)
party_titles=df.select_dtypes(include=[np.number]).columns.tolist()
party_titles=party_titles[6:]

votes=(votes[(votes.sum(axis=1)>0)])

N=votes.shape[0]
M=votes.shape[1]


# In[ ]:


from sklearn.decomposition import PCA as sklearnPCA
import matplotlib.pyplot as plt
pca=sklearnPCA(n_components=2)

relative_votes=votes.div(votes.sum(axis=1), axis=0)

X=pca.fit_transform(relative_votes)
plt.plot(X[:,0],X[:,1],'.',markersize=1)
plt.show()


# In[ ]:


from sklearn.cluster import KMeans
K=4;
kmeanModel = KMeans(n_clusters=K, max_iter=100).fit(relative_votes)
kmeanModel.fit(relative_votes)
for k in range(0,K):
    plt.plot(X[kmeanModel.labels_==k,0],X[kmeanModel.labels_==k,1],'.',markersize=1)

cluster=[1]*K
for k in range(0,K):
    cluster[k]=tuple(zip(kmeanModel.cluster_centers_[k],party_titles))
    cluster[k]=sorted(cluster[k],reverse=True)
    for i in range(0,5): # showing top 5 parties in each cluster
        print(cluster[k][i][1]+" - "+str(round(100*cluster[k][i][0])/100), end=', ')
    print()
    


# In[ ]:


import networkx as nx
from community import best_partition # --> http://perso.crans.org/aynaud/communities/

C=np.corrcoef(relative_votes,rowvar=0)
A=1*(C>0)
G=nx.Graph(A)
G=nx.relabel_nodes(G,dict(zip(G.nodes(),relative_votes.columns.values)))
communities=best_partition(G)

community_colors={0:0,1:0.5,2:1}
node_coloring=[community_colors[communities[node]] for node in G.nodes()]

nx.pos=nx.fruchterman_reingold_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1.0, center=None)

nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=12)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)


# In[ ]:


threshold=0.02
party_is_major=((votes.sum(axis=0)/sum(votes.sum(axis=0)))>threshold)
major_parties=relative_votes.columns.values[party_is_major==True]
major_party_votes=relative_votes[major_parties]

C=np.corrcoef(major_party_votes,rowvar=0)
A=1*(C>0)
G=nx.Graph(A)
G=nx.relabel_nodes(G,dict(zip(G.nodes(),major_parties)))
communities=best_partition(G)

community_colors={0:0,1:0.2,2:0.5,3:0.7,4:0.9}
node_coloring=[community_colors[communities[node]] for node in G.nodes()]

nx.pos=nx.fruchterman_reingold_layout(G, dim=2, k=None, pos=None, fixed=None, iterations=5000, weight='weight', scale=1.0, center=None)

nx.draw_networkx(G, cmap=plt.get_cmap('jet'), with_labels=True, node_color=node_coloring,font_size=30)
fig = plt.gcf()
fig.set_size_inches(18.5, 10.5)


# In[ ]:


# help (nx.draw_networkx)
help (nx.layout)

