#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn import datasets


# In[ ]:


iris = datasets.load_iris()


# In[ ]:


X_iris = iris.data
y_iris = iris.target


# In[ ]:


from sklearn.preprocessing import scale
X_scaled = pd.DataFrame(scale(X_iris))


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt

Ks = [2,3,4,5,6,7,8,9]
ssw=[]
for k in Ks:
    kmeans=KMeans(n_clusters=int(k))
    kmeans.fit(X_scaled)
    sil_score=silhouette_score(X_scaled,kmeans.labels_)
    print("silhouette score:",sil_score,"number of clusters are:", int(k))
    ssw.append(kmeans.inertia_)
plt.plot(Ks,ssw)


# 2.	Find optimal K

# In[ ]:


k = 3
kmeans = KMeans(n_clusters=k)
kmeans.fit(X_scaled)


# In[ ]:


labels1 = kmeans.labels_
X_scaled["cluster"]=labels1


# In[ ]:


for i in range(k):
    ds = X_scaled[X_scaled["cluster"]==i].as_matrix()
    plt.plot(ds[:,0],ds[:,1],'o')

plt.show()


# In[ ]:


kmeans.inertia_


# when k = 2 the inertia > than when k = 3 so optimal number of clusters = 3 as the purpose of k means clustering is to to rduce the inertia.
# 
# From the elbow curve maximum number of clusters = 3

# 3.	Perform Hierarchical clustering using different linkage methods

# SINGLE

# In[ ]:


from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering


# In[ ]:


for n_clusters in range(2,10):
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',linkage='single')
    cluster_labels = cluster_model.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled,cluster_labels,metric='euclidean')
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is:", silhouette_avg)


# In[ ]:


for n_clusters in range(2,10):
    cluster_model = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean',linkage='ward')
    cluster_labels = cluster_model.fit_predict(X_scaled)
    silhouette_avg = silhouette_score(X_scaled,cluster_labels,metric='euclidean')
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is:", silhouette_avg)


# 3 is the optimum number of clusters for hierarchical clustering as well

# In[ ]:


s = 3
hclust = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='single')
hclust.fit(X_scaled)


# In[ ]:


hclust1 = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='complete')
hclust1.fit(X_scaled)


# In[ ]:


hclust2 = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='average')
hclust2.fit(X_scaled)


# In[ ]:


hclust3 = AgglomerativeClustering(n_clusters=s, affinity='euclidean',linkage='ward')
hclust3.fit(X_scaled)


# In[ ]:


labels = hclust.fit_predict(X_scaled)
X_scaled["cluster"]=labels


# In[ ]:


for i in range(s):
    hc = X_scaled[X_scaled["cluster"]==i].as_matrix()
    plt.plot(hc[:,0],hc[:,1],'o')
plt.show()


# In[ ]:


# SINGLE


# In[ ]:


Z = linkage(X_scaled, 'single')
plt.figure(figsize=(10, 10))
plt.title('Agglomerative Hierarchical Clustering Dendogram')
plt.xlabel('Cluster points')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )
plt.tight_layout()


# In[ ]:


# COMPLETE

Z1 = linkage(X_scaled, 'complete')
plt.figure(figsize=(10, 10))
plt.title('Agglomerative Hierarchical Clustering Dendogram')
plt.xlabel('Cluster points')
plt.ylabel('Distance')
dendrogram(Z1, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )
plt.tight_layout()


# In[ ]:


# AVERAGE

Z2 = linkage(X_scaled, 'average')
plt.figure(figsize=(10, 10))
plt.title('Agglomerative Hierarchical Clustering Dendogram')
plt.xlabel('Cluster points')
plt.ylabel('Distance')
dendrogram(Z2, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )
plt.tight_layout()


# In[ ]:


# WARD

Z33 = linkage(X_scaled, 'ward')
plt.figure(figsize=(10, 10))
plt.title('Agglomerative Hierarchical Clustering Dendogram')
plt.xlabel('Cluster points')
plt.ylabel('Distance')
dendrogram(Z, leaf_rotation=90.,color_threshold = 40, leaf_font_size=8. )
plt.tight_layout()


# The complete and average distance methods seem to be good as the clusters are compact and are separated from one another by maximal distance.

# In[ ]:




