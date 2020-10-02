#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# **Unsupervised Learning:** We have data that has hidden labels and our aim is to find these labels of the data

# **1. K Means Algorithm**
# 
# 1.  We choose a k value 
# 2. Then it is randomly created k centroids
# 3. Every data point is clusterd according to the nearest centroid
# 4. By taking average of all data points that belog to a centroid, it is created new centroids.
# 5. Using these new centroids repeat 3 and 4
# 6. Finally, when centroids remain stationary, the algorith stops there.
# 7. As a result, according to these centroids, data is clustered

# **How k value is selected**
# 
# 1. For k=1, run KMeans algorithm
# 2. For each cluster (k cluster we have), it is calculated WCSS (within cluster sum of squares) value
# 3. repeat 1 and 2 for 1<k<15
# 4. obtain k vs WCSS plot
# 5. Using elbow rule,  choose the optimum k value to be used in K Means Algorithm

# In[ ]:


data = pd.read_csv('../input/column_2C_weka.csv')
data.head()


# In[ ]:


# As you can see there is no labels in data
x = data['pelvic_radius']
y = data['degree_spondylolisthesis']
plt.figure(figsize=(13,5))
plt.scatter(x,y)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# **Lets find optimum k value**

# In[ ]:


df = data.loc[:, ['degree_spondylolisthesis', 'pelvic_radius']]
df.head()


# In[ ]:


# which k value to choose
from sklearn.cluster import KMeans
wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_) # kmeans.inertia : calculate wcss
    
plt.plot(range(1,15), wcss, '-o')
plt.xlabel('number of k (cluster) value')
plt.ylabel('wcss')
plt.show()


# using elbow rule we can select k=2, 3 or 4 (the elbow point is not quite obvious here)

# **for k=2**

# In[ ]:


# for k=2, lets write KMeans
from sklearn.cluster import KMeans
kmeans2 = KMeans(n_clusters = 2)
clusters =kmeans2.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters


# In[ ]:


# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# **for k=3**

# In[ ]:


# if we choose k=3
from sklearn.cluster import KMeans
kmeans3 = KMeans(n_clusters = 3)
clusters =kmeans3.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters

# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# **for k=4**

# In[ ]:


# if we choose k=4
from sklearn.cluster import KMeans
kmeans4 = KMeans(n_clusters = 4)
clusters =kmeans4.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters

# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# **Original data is as follow**

# In[ ]:


# plot
colors = [0 if i=='Abnormal' else 1 for i in data['class']] # to create colors
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = colors)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# **2. Hierarcical Clustering**
# 
# 1. Assign each data point as a cluster
# 2. Create a new cluster by choosing the closest two clusters  
# 3. repeat 2 until it remains only one cluster

# In[ ]:


# DENDOGRAM 
# here we will try to predict how many clusters we have 
from scipy.cluster.hierarchy import linkage, dendrogram # linkage: create dendrogram
df1 = data.loc[:, ['pelvic_radius', 'degree_spondylolisthesis']]
merg = linkage(df1, method='ward') # ward: cluster icindeki yayilimlari minimize et (wcss gibi bisey)
dendrogram(merg, leaf_rotation=90)
plt.xlabel('data points')
plt.ylabel('euclidian distance')
plt.show()


# * vertical lines are clusters
# * height on dendogram: distance between merging cluster
# * method= 'single' : closest points of clusters
# * we are going to choose the highest distance between merging clusters, which are not cut by horizontal line
# * this suggest that choose 3 clusters (draw a horizontal line approx from euc. distance=400, it cuts at 3 points)

# In[ ]:


from sklearn.cluster import AgglomerativeClustering

hierarcical_cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster = hierarcical_cluster.fit_predict(df1)

# add label for df1
df1['label'] = cluster

#plot
plt.scatter(df1['pelvic_radius'],df1['degree_spondylolisthesis'],c = cluster)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# **CONCLUSION**
# 
# * We wee that the predictions that we made using Kmeans algorihm does not suit well for this problem
# * But for Hierarcical Clustering, we have predicted the original data better as compared to the Kmeans. 
