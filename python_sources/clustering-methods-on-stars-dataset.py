#!/usr/bin/env python
# coding: utf-8

# **EXPLANATION**
# 
# In this kernel, I used 2 different clustering methods on Star Cluster dataset. They are K-Means and Hierarchical Clustering .
# 
# **CONTENTS**
# 
# **1. EDA**
# 
# I am looking data in general and delete columns like "id".
# 
# **2. Visualization**
# 
# **3. K-Means**
# 
# **4. Hierarchical Clustering**
# 
# **5. Conclusion**

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


# In[ ]:


data = pd.read_csv("../input/c_0000.csv",sep=",")


# **1. EDA**

# In[ ]:


data.head()


# In[ ]:


# We filter "id" columns because id has no effect on relations.
data = data.loc[:,data.columns != 'id']


# In[ ]:


data.head()


# In[ ]:


data2 = data.iloc[:,[0,3]]


# In[ ]:


data2.head()


# **2. Visaualization**

# In[ ]:


plt.scatter(data2.x,data2.vx,color="green")
plt.xlabel("X positon of stars")
plt.ylabel("Velocity in X axis of stars")
plt.show()


# **3. K-Means**

# In[ ]:


from sklearn.cluster import  KMeans
wcss = []

for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data2)
    wcss.append(kmeans.inertia_) # inertia means that find to value of wcss
    
plt.plot(range(1,15),wcss)
plt.xlabel("number of k (cluster) value")
plt.ylabel("wcss")
plt.show()


# It looks that 4 has an elbow point.

# In[ ]:


# we can take elbow as 4
kmean2 = KMeans(n_clusters=4)
clusters = kmean2.fit_predict(data2)

data2["label"] = clusters

plt.scatter(data2.x[data2.label == 0], data2.vx[data2.label == 0], color="red")
plt.scatter(data2.x[data2.label == 1], data2.vx[data2.label == 1], color="blue")
plt.scatter(data2.x[data2.label == 2], data2.vx[data2.label == 2], color="green")
plt.scatter(data2.x[data2.label == 3], data2.vx[data2.label == 3], color="purple")

plt.scatter(kmean2.cluster_centers_[:,0],kmean2.cluster_centers_[:,1], color="orange") # scentroidler

plt.show()


# In[ ]:


# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans3 = KMeans(n_clusters=i)
    kmeans3.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


# In[ ]:


data3 = data2.iloc[:,data2.columns != 'label'].head(1000)


#  **4. Hierarchical Clustering**

# In[ ]:


# dendrogram
from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data3, method="ward") # scipy is an algorithm of hiyerarchal clusturing
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()


# In[ ]:



# HC
from sklearn.cluster import AgglomerativeClustering

hiyerartical_cluster = AgglomerativeClustering(n_clusters=4, affinity="euclidean",linkage="ward")
cluster = hiyerartical_cluster.fit_predict(data3)

data3["label"] = cluster


# In[ ]:


plt.scatter(data3.x[data3.label == 0], data3.vx[data3.label == 0], color="red")
plt.scatter(data3.x[data3.label == 1], data3.vx[data3.label == 1], color="blue")
plt.scatter(data3.x[data3.label == 2], data3.vx[data3.label == 2], color="green")
plt.scatter(data3.x[data3.label == 3], data3.vx[data3.label == 3], color="purple")


# **5. Conclusion**
# 
# There is a great difference between K-Means and Hierarchical Clustering methods as you see in graphics.
# 
