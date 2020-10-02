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


# # Unsupervised Learning
# * Unsupervised learning: It uses data that has unlabeled and uncover hidden patterns from unlabeled data. Example, there are heart disease data that do not have labels. You do not know which heart disease target is 1 or 0.
# 

# In[ ]:


data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data
data.head()


# # K-Means
# 

# * As you know  heart disease data is labeled (supervised) data. It has target variables. In order to work on unsupervised learning, lets drop target variables and to visualize just consider chol and thalach.

# In[ ]:


# As you can see there is no labels in data
# we need to import matplot library.
import matplotlib.pyplot as plt
plt.scatter(data["chol"],data["thalach"])
plt.xlabel("chol")
plt.ylabel("thalach")
plt.show()


# In[ ]:


# KMeans Clustering
data2 = data.loc[:,["chol","thalach"]]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2) # we choose 2 cluster in our data.
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data["chol"],data["thalach"],c = labels)
plt.xlabel("chol")
plt.xlabel("thalach")
plt.show()


# # Evaluating of Clustering
# * We cluster data in two groups. Okey well is that correct clustering? In order to evaluate clustering we will use cross tabulation table.
# 
# * There are two clusters that are 0 and 1 (1 = male, 0 = female)
# * First class female(0) includes 62 0 and 54 1 patients.
# * Second class male(1) includes 76 0 and 111 1 patiens.

# In[ ]:


# cross tabulation table
df = pd.DataFrame({'labels':labels,"target":data["target"]})
ct = pd.crosstab(df['labels'],df["target"])
print(ct)


# * inertia: how spread out the clusters are distance from each sample
# * lower inertia means more clusters
# 

# In[ ]:


# inertia
inertia_list = np.empty(14)
for i in range(1,14):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,14),inertia_list,"-o")
plt.xlabel("Number of cluster")
plt.ylabel("Inertia")
plt.show()


# # Standardization
# * Standardization (or Z-score normalization) is the process of rescaling the features.
# * Do not forget standardization as pre-processing.
# * As we already have visualized data so you got the idea. Now we can use all features for clustering.
# * We can use pipeline like supervised learning.

# In[ ]:


data = pd.read_csv("/kaggle/input/heart-disease-uci/heart.csv")
data3 = data.drop("target",axis = 1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({"labels":labels,"target":data["target"]})
ct = pd.crosstab(df["labels"],df["target"])
print(ct)


# # Hierarchy
# * It has two different variations: Agglomerative (from part to whole) and Divisive (from whole to part).
# * A dendrogram is a tree diagram that shows relationships between similar datasets or a hierarchical cluster. It gives information about how many clusters we will create.
# 

# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram

merg = linkage(data,method = "ward")
dendrogram(merg,leaf_rotation = 90)
plt.show()


# In[ ]:


# HC
from sklearn.cluster import AgglomerativeClustering
hiyerartical_cluster = AgglomerativeClustering(n_clusters = 2,affinity="euclidean", linkage = "ward")
cluster = hiyerartical_cluster.fit_predict(data)

data["label"] = cluster
plt.scatter(data.chol[data.label == 0 ],data.thalach[data.label == 0],color = "red")
plt.scatter(data.chol[data.label == 1 ],data.thalach[data.label == 1],color = "green")
plt.show()


# As you can see it's same graphic with K-Means Clustering.
