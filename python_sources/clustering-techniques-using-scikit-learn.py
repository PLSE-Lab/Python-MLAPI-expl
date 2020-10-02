#!/usr/bin/env python
# coding: utf-8

# Created by: Sangwook Cheon
# 
# Date: Dec 24, 2018
# 
# This is step-by-step guide to Clustering using scikit-learn, which I created for reference. I added some useful notes along the way to clarify things. 
# This notebook's content is from A-Z Datascience course, and I hope this will be useful to those who want to review materials covered, or anyone who wants to learn the basics of clustering.
# 
# # Content:
# ### 1. K-means Clustering 
# ### 2. Hierarchical Clustering
# 
# ________
# _________
# 
# # K-means Clustering
# ![i1](https://i.imgur.com/p3GHQXL.png)
# 
# ### Random Initialization Trap
# If the initial centroids are placed at random, they can potentially determine the way clusters form, which is not idea as there exists optimal initial centroids that need to be placed so that the algorithm works properly. In order to achieve, this K-means ++ algorithm is used. This happens in the background of libraries, so no need to do this manually.
# 
# ### Choosing the right number of clusters (initial clusters)
# Within Cluster Sum of Squares (WSCC) algorithm
# ![i13](https://i.imgur.com/NZ2jRt3.png)
# 
# Elbow method:
# ![i14](https://i.imgur.com/0k6ALfB.png)
# 
# Choose the point where there is a significant drop from behind, and low drop after that point.
# 

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values

#Using elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []

#Iterating over 1, 2, 3, ---- 10 clusters
for i in range(1, 11): 
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_) # intertia_ is an attribute that has the wcss number
plt.plot(range(1,11), wcss)
plt.title("Elbow method applied to Mall_customers dataset")
plt.xlabel("number of clusters")
plt.ylabel("wcss")
plt.show()


# As can be seen here, the optimal value is 5 clusters. 

# In[ ]:


# Apply kmeans to the dataset
kmeans = KMeans(n_clusters = 5, max_iter = 300, init = 'k-means++', n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x) #predict which cluster each point belongs to
y_kmeans


# In[ ]:


#visualizing the clusters

plt.figure(2)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'red', label = 'Customer 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'Customer 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green', label = 'Customer 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'Customer 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'purple', label = 'Customer 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids' )
# _centers_ --> also an attribute that can be accessed.
# s --> size of each point
plt.title("K-means clustering applied to Mall_Customers")
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()


# In[ ]:


#Now that each group is identified, we can name each category.

plt.figure(3)
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 50, c = 'red', label = 'careful')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 50, c = 'blue', label = 'standard')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 50, c = 'green', label = 'target')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 50, c = 'cyan', label = 'reckless')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 50, c = 'purple', label = 'sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'centroids' )
# _centers_ --> also an attribute that can be accessed.
# s --> size of each point
plt.title("K-means clustering applied to Mall_Customers")
plt.xlabel('Annual Income ($)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()


# # Hierarchical Clustering
# ## Type 1: Agglomerative
# ![i154](https://i.imgur.com/aoMmHRy.png)
# Options to measure distance between clusters:
# ![i345](https://i.imgur.com/i3G9Lnt.png)
# How dendrograms work (which are the memory of HC model):
# ![i10923](https://i.imgur.com/6Fy4Cxv.png)
# Vertical height of the column represents the distance, and the horizontal lines signifies that the two clusters have been combined.
# 
# How to use the dendrogram: You can set the threshold value for dissimilarity (distance between the clusters), so that when combining two leads to a higher value, the algorithm would prevent merge from happening. This threshold can be different by context, but an optimal threshold can be roughly found.
# 
# --> Set the threshold that it crosses the largest distance of the vertical line there is on the dendrogram. The vertical line with the largest distance should not cross any extended horizontal lines of columns. Example:
# ![i83](https://i.imgur.com/7geN4Wi.png)
# 
# ## Type 2: Divisive

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('../input/Mall_Customers.csv')
x = dataset.iloc[:, [3, 4]].values
# y is not available, as we do not know the patterns we expect. It's our job to find patterns from x.

#Use dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram_1 = sch.dendrogram(sch.linkage(x, method = 'ward'))
plt.title('Dendograms')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
plt.show()
# ward method tries to minimize variance between clusters


# Here the longest distance would be:
# ![i238047](https://i.imgur.com/efEqQ5F.png)
# As no extended horizontal lines cross this line. Then, count the number of vertical lines that are included in the section (between two red horizontal lines). As shown on the image, there are 5 vertical lines, showing that 5 clusters are the optimal number. This result has already been shown by K-means clustering

# In[ ]:


# Fitting HC to the mall dataset

from sklearn.cluster import AgglomerativeClustering 
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(x)
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 50, c = 'red', label = 'Customer 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 50, c = 'green', label = 'Customer 1')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 50, c = 'blue', label = 'Customer 1')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 50, c = 'purple', label = 'Customer 1')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 50, c = 'orange', label = 'Customer 1')
plt.title('Applying Hierarchical Clustering to Mall_customer')
plt.xlabel('Annual income ($)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

#The above segment of code is only for visualizing 2D data, not higher-dimension

