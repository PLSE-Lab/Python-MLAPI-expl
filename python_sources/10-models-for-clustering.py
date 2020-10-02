#!/usr/bin/env python
# coding: utf-8

# # Introduction
#  Involve 10 Models Clustering
#  
# <br>
# <br>
# <font color = 'blue'>
# <b>Content: </b>
# 
# 1. [Prepare Problems]
#     * [Load Libraries](#2)
#     * [Load Dataset](#3)    
# 1. [Models]
#     * [K-Means](#4)
#     * [Affinity Propagation](#5)
#     * [BIRCH](#6)
#     * [DBSCAN](#7)
#     * [Mini Batch K-Means](#8)
#     * [Mean Shift](#9)
#     * [OPTICS](#10)
#     * [Spectral Clustering](#11)
#     * [Gaussian Mixture Model](#12)
#     * [Agglomerative Clustering](#13)
# 1. [References](#14)

# <a id = "2"></a><br>
# ## Load Libraries

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy import unique
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics
#                                 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname,filename))


# <a id = "3"></a><br>
# ## Load Dataset

# In[ ]:


data = pd.read_csv('/kaggle/input/mall-customers/Mall_Customers.csv', index_col=0)
data.head()


# In[ ]:


data.drop(['Genre'], axis=1, inplace=True)
data.drop(['Age'], axis=1, inplace=True)

data.head()


# ### Taking full fraction of data
# It shuffles the data

# In[ ]:


data = data.sample(frac=1)


# In[ ]:


data.head()


# <a id = "4"></a><br>
# ## 1 - K-Means 

# In[ ]:


k_means = KMeans(n_clusters=2)
k_means.fit(data)


# ### Labels

# In[ ]:


k_means.labels_


# In[ ]:


np.unique(k_means.labels_)


# In[ ]:


centers = k_means.cluster_centers_

centers


# In[ ]:


plt.figure(figsize=(10, 8))

plt.scatter(data['Annual Income (k$)'], 
            data['Spending Score (1-100)'], 
            c=k_means.labels_, s=100)

plt.scatter(centers[:,0], centers[:,1], color='blue', marker='s', s=200) 

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('K-Means with 2 clusters')

plt.show()


# * A measure of how similar a point is to other points in its own cluster and how different it is from points in other clusters.

# In[ ]:


from sklearn.metrics import silhouette_score

score = silhouette_score (data, k_means.labels_)

print("Score = ", score)


# In[ ]:


wscc = []
for i in range(1,15): 
    kmeans = KMeans(n_clusters=i, init="k-means++",random_state=0)
    kmeans.fit(data)
    wscc.append(kmeans.inertia_)  

plt.plot(range(1,15),wscc,marker="*",c="black")
plt.title("Elbow plot for optimal number of clusters")


# ### KMeans clustering with 5 clusters

# In[ ]:


k_means = KMeans(n_clusters=5)
k_means.fit(data)


# In[ ]:


np.unique(k_means.labels_)


# In[ ]:


centers = k_means.cluster_centers_

centers


# ### Displaying Data in 5 cluster form 
# with 5 centroids

# In[ ]:


plt.figure(figsize=(10, 8))

plt.scatter(data['Annual Income (k$)'], 
            data['Spending Score (1-100)'], 
            c=k_means.labels_, s=100)

plt.scatter(centers[:,0], centers[:,1], color='blue', marker='s', s=200) 

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('5 Cluster K-Means')

plt.show()


# Silhouette Score: This is a better measure to decide the number of clusters to be formulated from the data. 

# In[ ]:


score = metrics.silhouette_score(data, k_means.labels_)

print("Score = ", score)


# This function returns the Silhouette Coefficient for each sample.
# 
# The best value is 1 and the worst value is -1. Values near 0 indicate overlapping clusters.

# In[ ]:


score1 = metrics.silhouette_samples(data, k_means.labels_, metric='euclidean')
print("Score = ", score1)


# <a id = "5"></a><br>
# ## 2 - Affinity Propagation
# Affinity Propagation involves finding a set of exemplars that best summarize the data.

# In[ ]:


model_aff = AffinityPropagation(damping=0.9)
model_aff.fit(data)
#
yhat_aff = model_aff.predict(data)
clusters_aff = unique(yhat_aff)
print("Clusters of Affinity Prop.",clusters_aff)
labels_aff = model_aff.labels_
centroids_aff = model_aff.cluster_centers_


# In[ ]:


plt.figure(figsize=(10, 8))

plt.scatter(data['Annual Income (k$)'], 
            data['Spending Score (1-100)'], 
            c=labels_aff, s=100)

plt.scatter(centroids_aff[:,0], centroids_aff[:,1], color='red', marker='*', s=200) 

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Affinity Propagation')
plt.grid()
plt.show()


# In[ ]:


score_aff = metrics.silhouette_score(data,labels_aff)

print("Score of Affinity Propagation = ", score_aff)


# <a id = "6"></a><br>
# ## 3 - BIRCH
# BIRCH Clustering (BIRCH is short for Balanced Iterative Reducing and Clustering using
# Hierarchies) involves constructing a tree structure from which cluster centroids are extracted.

# In[ ]:


model_br = Birch(threshold=0.01, n_clusters=5)
model_br.fit(data)
#
yhat_br = model_br.predict(data)
clusters_br = unique(yhat_br)
print("Clusters of Birch",clusters_br)
labels_br = model_br.labels_


# In[ ]:


score_br = metrics.silhouette_score(data,labels_br)

print("Score of Birch = ", score_br)


# <a id = "7"></a><br>
# ## 4- DBSCAN

#  * DBSCAN Clustering (where DBSCAN is short for Density-Based Spatial Clustering of Applications with Noise) involves finding high-density areas in the domain and expanding those areas of the feature space around them as clusters.
#  * For this data, could not get a good result.

# In[ ]:


# dbscan clustering
from numpy import unique
from numpy import where
data_X = data.iloc[:,[0,1]].values


# In[ ]:


# define the model
model = DBSCAN(eps=0.7, min_samples=90)
# fit model and predict clusters
yhat = model.fit_predict(data_X)
# retrieve unique clusters
clusters = unique(yhat)
# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	plt.scatter(data_X[row_ix, 0], data_X[row_ix, 1])
# show the plot
plt.show()


# <a id = "8"></a><br>
# ## 5 - Mini Batch K-Means

# * Mini-Batch K-Means is a modified version of k-means that makes updates to the cluster centroids using mini-batches of samples rather than the entire dataset, which can make it faster for large datasets, and perhaps more robust to statistical noise.

# In[ ]:


model_mini = MiniBatchKMeans(n_clusters=2)
model_mini.fit(data)
#
yhat_mini = model_mini.predict(data)
clusters_mini = unique(yhat_mini)
print("Clusters of Mini Batch KMeans.",clusters_mini)
labels_mini = model_mini.labels_
centroids_mini = model_mini.cluster_centers_


# In[ ]:


wscc = []
for i in range(1,15): 
    mkmeans = MiniBatchKMeans(n_clusters=i, init="k-means++",random_state=0)
    mkmeans.fit(data)
    wscc.append(mkmeans.inertia_)  

plt.plot(range(1,15),wscc,marker="*",c="black")
plt.title("Elbow plot for Mini Batch KMeans")


# In[ ]:


model_mini = MiniBatchKMeans(n_clusters=5)
model_mini.fit(data)
#
yhat_mini = model_mini.predict(data)
clusters_mini = unique(yhat_mini)
print("Clusters of Mini Batch KMeans.",clusters_mini)
labels_mini = model_mini.labels_
centroids_mini = model_mini.cluster_centers_


# In[ ]:


plt.figure(figsize=(10, 8))

plt.scatter(data['Annual Income (k$)'], 
            data['Spending Score (1-100)'], 
            c=labels_mini, s=100)

plt.scatter(centroids_mini[:,0], centroids_mini[:,1], color='red', marker='*', s=200) 

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Mini Batch KMeans')
plt.grid()
plt.show()


# In[ ]:


score_mini = metrics.silhouette_score(data,labels_mini)

print("Score of Birch = ", score_mini)


# <a id = "9"></a><br>
# ## 6 - Mean Shift

# * Mean shift clustering involves finding and adapting centroids based on the density of examples in the feature space.

# In[ ]:


model_ms = MeanShift(bandwidth=25)
model_ms.fit(data)
#
yhat_ms = model_ms.predict(data)
clusters_ms = unique(yhat_ms)
print("Clusters of Mean Shift.",clusters_ms)
labels_ms = model_ms.labels_
centroids_ms = model_ms.cluster_centers_


# In[ ]:


plt.figure(figsize=(10, 8))

plt.scatter(data['Annual Income (k$)'], 
            data['Spending Score (1-100)'], 
            c=labels_ms, s=100)

plt.scatter(centroids_ms[:,0], centroids_ms[:,1], color='red', marker='*', s=200) 

plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Mean Shift')
plt.grid()
plt.show()


# In[ ]:


score_ms = metrics.silhouette_score(data,labels_ms)

print("Score of Mean Shift = ", score_ms)


# <a id = "10"></a><br>
# ## 7 - OPTICS

# * OPTICS clustering (where OPTICS is short for Ordering Points To Identify the Clustering Structure) is a modified version of DBSCAN described above.
# * In this case, I could not achieve a reasonable result on this dataset.

# In[ ]:


model_op = OPTICS(eps=0.8, min_samples=10)
#
yhat_op = model_op.fit_predict(data)
clusters_op = unique(yhat_op)
print("Clusters of Mean Shift.",clusters_op)
labels_op = model_op.labels_


# In[ ]:


score_op = metrics.silhouette_score(data,labels_op)

print("Score of Mean Shift = ", score_op)


# <a id = "11"></a><br>
# ## 8 - Spectral Clustering

# * Spectral Clustering is a general class of clustering methods, drawn from linear algebra.

# In[ ]:


model_sc = SpectralClustering(n_clusters=5)
#
yhat_sc = model_sc.fit_predict(data)
clusters_sc = unique(yhat_sc)
print("Clusters of Mean Shift.",clusters_sc)
labels_sc = model_sc.labels_


# In[ ]:


score_sc = metrics.silhouette_score(data,labels_sc)

print("Score of Mean Shift = ", score_sc)


# <a id = "12"></a><br>
# ## 9 - Gaussian Mixture Model

# * A Gaussian mixture model summarizes a multivariate probability density function with a mixture of Gaussian probability distributions as its name suggests.

# In[ ]:


from numpy import unique
from numpy import where
data_X = data.iloc[:,[0,1]].values


# In[ ]:


model_gb = GaussianMixture(n_components=5)
model_gb.fit(data_X)
#
yhat_gb = model_gb.predict(data_X)
clusters_gb = unique(yhat_gb)
# create scatter plot for samples from each cluster
for cluster in clusters_gb:
	# get row indexes for samples with this cluster
	row_ix = where(yhat_gb == cluster)
	# create scatter of these samples
	plt.scatter(data_X[row_ix, 0], data_X[row_ix, 1])
# show the plot
plt.show()


# <a id = "13"></a><br>
# ## 10 - Agglomerative Clustering

# * Agglomerative clustering involves merging examples until the desired number of clusters is achieved.

# In[ ]:


model_agg = AgglomerativeClustering(n_clusters=5)
#
yhat_agg = model_agg.fit_predict(data)
clusters_agg = unique(yhat_agg)
print("Clusters of Mini Batch KMeans.",clusters_agg)
labels_agg = model_agg.labels_


# In[ ]:


score_agg = metrics.silhouette_score(data,labels_agg)

print("Score of Mean Shift = ", score_agg)


# #  If you like my kernel, please upvote

# <a id = "14"></a><br>
# ## References

# * https://machinelearningmastery.com/clustering-algorithms-with-python/
