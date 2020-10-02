#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


#Loading the required modules to read the data
import pandas as pd


# **Reading data**

# In[ ]:


cereals = pd.read_csv("../input/Cereals.csv")
cereals.head()


# **Aggregrating "name", "shelf","rating" to make labels**

# In[ ]:


cereals['label'] = cereals['name']+'('+ cereals['shelf'].astype(str) + " - " + round(cereals['rating'],2).astype(str)+')'
cereals.drop(['name','shelf','rating'], axis=1, inplace=True)


# ****Data Exploration****

# In[ ]:


#check the head
cereals.head()


# **Checking the summary statistics**

# In[ ]:


cereals.describe()


# **Decouple label from the features**

# In[ ]:


#select all columns except "label"
cereals_label = cereals["label"]
cereals = cereals[cereals.columns.difference(['label'])]


# **check missing values**

# In[ ]:


cereals.isnull().sum()


# In[ ]:


cereals.isnull().sum().sum()


# **Imputation**

# In[ ]:



from sklearn.preprocessing import Imputer
mean_Imputer = Imputer()
Imputed_cereals = pd.DataFrame(mean_Imputer.fit_transform(cereals),columns=cereals.columns)
Imputed_cereals


# **Checking NA's again**

# In[ ]:


Imputed_cereals.isnull().sum(axis=0)


# **Standardization**

# In[ ]:


from sklearn.preprocessing import StandardScaler
standardizer = StandardScaler()
standardizer.fit(Imputed_cereals)
std_x = standardizer.transform(Imputed_cereals)
std_cereals = pd.DataFrame(std_x,columns=cereals.columns)
std_cereals.head()


# In[ ]:


std_cereals.describe()


# Note- Clustering is an unsupervised method and hence we're not concerned about train-test split or prediction accuracies.

# **Agglomerative Clustering**

# Parameter description
# 
# n_clusters: The number of clusters to find.
# 
# linkage: {"ward","complete", "average"}
# 
# ->ward minimizes the variance of the clusters being merged.
# ->comlete uses the maximum distances between all observations of the two sets.
# ->average uses the average of the distances of each observation of the two sets.
# 
# affinity:{"euclidean", "l1","l2", "manhattan","cosine"}
# 
# ::-->Metric used to compute the linkage

# In[ ]:


#loading the ethods
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'notebook')

#preparing linkage matrix
linkage_matrix = linkage(std_cereals, method = "ward", metric = 'euclidean')

##plotting
dendrogram(linkage_matrix,labels=cereals_label.as_matrix())
plt.tight_layout()
plt.show()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering

##Instantiating object
agg_clust = AgglomerativeClustering(n_clusters=6, affinity = 'euclidean', linkage = 'ward')

##Training model and return class labels
agg_clusters = agg_clust.fit_predict(std_cereals)

##Label - Cluster
agg_result = pd.DataFrame({'labels': cereals_label, "agg_cluster": agg_clusters})
agg_result.head()


# **K-Means Clustering**

# *Parameter description*

# n_clusters : The number of clusters to find
# 
# tol: Relative tolerance with regards to inertia to declare convergence
# 
# n_init: Number of time the k-means algorithm will be run with different centroid seeds. The final results will be best output of n_init consecutive runs in terms of inertia.
# 
# max_iter: max iterations of recomputing new cluster centroids
# 
# n_jobs: The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel.

# In[ ]:


from sklearn.cluster import KMeans
kmeans_object = KMeans(n_clusters = 5, random_state=123)
kmeans_object.fit(std_cereals)
kmeans_clusters= kmeans_object.predict(std_cereals)
kmeans_result = pd.DataFrame({"labels":cereals_label, "kmeans_cluster":kmeans_clusters})
kmeans_result.head()


# **Inspecting cluster centroids to understand average statistics of each cluster**

# In[ ]:


kmeans_object.cluster_centers_


# **Selecting the best k value - using Elbow plot**

# In[ ]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=10000).fit(std_cereals)
    std_cereals["cluster"] = kmeans.labels_
#print(data["cluster"])
    sse[k] = kmeans.inertia_ #Intertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# **Visualizing the elbow plot**

# **How to select best K valu for K- Means ->Silhouette Analysis**

# *Higher the silhouette score better the clustering*
# 
# THe silhouette value is a measure of how similar an object is to its own cluster(cohesion) caompared to other cluster(seperable). The silhouette range from -1 to +1, where a high value indicates that the object is well matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.

# In[ ]:




