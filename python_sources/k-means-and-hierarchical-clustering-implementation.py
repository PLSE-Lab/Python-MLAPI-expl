#!/usr/bin/env python
# coding: utf-8

# # K-Means and Hierarchical Clustering Implementation
# 
# * Clustering algorithms is being used for unlabelled datasets.
# * This is an implementation example of clustering algorithms.  We'll use K-Means an Hierarchical clustering algorithms for seperate the cancer data by "radius_mean" and "texture_mean"
# 
# ## Index of contents
# 
# * [DATA EXPLORATION](#1)
# * [K-MEANS CLUSTERING](#2)
# * [HIERARCHICAL CLUSTERING](#3)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# <a id="1"></a> 
# **DATA EXPLORATION**

# In[ ]:


# Read and upload data
data = pd.read_csv("../input/data.csv")


# In[ ]:


# We don't need id and NaN data.
data.drop(["Unnamed: 32", "id"], axis = 1, inplace = True)
data.head()


# In[ ]:


data["diagnosis"].value_counts()

# We have 357 B and 212 M labelled data


# In[ ]:


# For clustering we do not need labels. Because we'll identify the labels.

dataWithoutLabels = data.drop(["diagnosis"], axis = 1)
dataWithoutLabels.head()


# In[ ]:


dataWithoutLabels.info()


# In[ ]:


# radius_mean and texture_mean features will be used for clustering. Before clustering process let's check  how our data looks.

sns.pairplot(data.loc[:,['radius_mean','texture_mean', 'diagnosis']], hue = "diagnosis", height = 5)
plt.show()


# In[ ]:


# Our data looks like below plot without diagnosis label

plt.figure(figsize = (10, 10))
plt.scatter(dataWithoutLabels["radius_mean"], dataWithoutLabels["texture_mean"])
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.show()


# <a id="2"></a> 
# **K-MEANS CLUSTERING**
# * Define **K** centers and cluster data,
# * Assign random centroids,
# * Cluster data points according to distance from centroids (euclidean distance),
# * Repeat step 3 until centroid positions start not to change.
# 
# ![KMeansClustering.png](http://i67.tinypic.com/sdpueu.png)
# * WCSS is a metric used for k value selection process. After this operation elbow rule is used for k value.
# 
# ![KMeansElbow.png](http://i64.tinypic.com/2dw7ztg.png)

# In[ ]:


from sklearn.cluster import KMeans
wcss = [] # within cluster sum of squares

for k in range(1, 15):
    kmeansForLoop = KMeans(n_clusters = k)
    kmeansForLoop.fit(dataWithoutLabels)
    wcss.append(kmeansForLoop.inertia_)

plt.figure(figsize = (10, 10))
plt.plot(range(1, 15), wcss)
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.show()


# In[ ]:


# Elbow point starting from 2 

dataWithoutLabels = data.loc[:,['radius_mean','texture_mean']]
kmeans = KMeans(n_clusters = 2)
clusters = kmeans.fit_predict(dataWithoutLabels)
dataWithoutLabels["type"] = clusters
dataWithoutLabels["type"].unique()


# In[ ]:


# Plot data after k = 2 clustering

plt.figure(figsize = (15, 10))
plt.scatter(dataWithoutLabels["radius_mean"][dataWithoutLabels["type"] == 0], dataWithoutLabels["texture_mean"][dataWithoutLabels["type"] == 0], color = "red")
plt.scatter(dataWithoutLabels["radius_mean"][dataWithoutLabels["type"] == 1], dataWithoutLabels["texture_mean"][dataWithoutLabels["type"] == 1], color = "green")
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.show()


# In[ ]:


# Data centroids middle of clustered scatters

plt.figure(figsize = (15, 10))
plt.scatter(dataWithoutLabels["radius_mean"], dataWithoutLabels["texture_mean"], c = clusters, alpha = 0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = "red", alpha = 1)
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.show()


# In[ ]:


dataWithoutDiagnosis = data.drop(["diagnosis"], axis = 1)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar, kmeans)
pipe.fit(dataWithoutDiagnosis)
labels = pipe.predict(dataWithoutDiagnosis)
df = pd.DataFrame({'labels': labels, "diagnosis" : data['diagnosis']})
ct = pd.crosstab(df['labels'], df['diagnosis'])
print(ct)


# <a id="3"></a> 
# **HIERARCHICAL CLUSTERING**
# * Each data point is transformed to cluster,
# * Create cluster using closest 2 data point,
# * Create cluster using closest 2 cluster,
# * Repeat step 3.
# 
# ![dendogram.PNG](http://i66.tinypic.com/97piew.png)
# 
# * Above mentioned distances are euclidean distances
# * Dendogram is used for n_clusters value detection.

# In[ ]:


dataWithoutTypes = dataWithoutLabels.drop(["type"], axis = 1)
dataWithoutTypes.head()


# In[ ]:


from scipy.cluster.hierarchy import linkage,dendrogram
merg = linkage(dataWithoutTypes, method = "ward")
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 2, affinity = "euclidean", linkage = "ward")
cluster = hc.fit_predict(dataWithoutTypes)
dataWithoutTypes["label"] = cluster


# In[ ]:


dataWithoutTypes.label.value_counts()


# In[ ]:


# Data after hierarchical clustering

plt.figure(figsize = (15, 10))
plt.scatter(dataWithoutTypes["radius_mean"][dataWithoutTypes.label == 0], dataWithoutTypes["texture_mean"][dataWithoutTypes.label == 0], color = "red")
plt.scatter(dataWithoutTypes["radius_mean"][dataWithoutTypes.label == 1], dataWithoutTypes["texture_mean"][dataWithoutTypes.label == 1], color = "blue")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.show()


# # Conclusion

# *We used unsupervised learning algorithms for clustering cancer data. Thank you for reading my kernel. Please do not hesitate to leave comments.*
