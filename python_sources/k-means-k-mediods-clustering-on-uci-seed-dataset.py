#!/usr/bin/env python
# coding: utf-8

# **Introduction**
# 
# **Abstract:** Measurements of geometrical properties of kernels belonging to three different varieties of wheat. A soft X-ray technique and GRAINS package were used to construct all seven, real-valued attributes.
# 
# Classes - Kama (0), Rosa (1) and Canadian (2)
# 
# Link to dataset : https://archive.ics.uci.edu/ml/datasets/seeds
# 
# ![](https://uci-seed-dataset.s3.ap-south-1.amazonaws.com/Dataset.PNG)
# 
# > You will learn following things by reading this kernel.
# 1. [Elbow](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html) method to find optimal k (number of clusters)
# 2. k-means clustering with k value found.
# 3. Calculate silhouette coefficient to measure clustering quality.
# 4. Calculate purity to measure clustering quality.
# 5. k-mediods clutering with k value found.
# 6. Conclusions on methods used.

# In[ ]:





# **Workspace setting and loading dataset**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn import metrics

dataset = pd.read_csv('../input/seed-from-uci/Seed_Data.csv')
dataset.head()


# In[ ]:


dataset.describe(include = "all")


# [](http://)**[Elbow](https://www.scikit-yb.org/en/latest/api/cluster/elbow.html) method to find optimal k (number of clusters)**

# In[ ]:


features = dataset.iloc[:, 0:7]
target = dataset.iloc[:, -1]

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,10))

visualizer.fit(features)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# The lowest distortion, which is sum of square distances from each point to its assigned cluster, is found at k = 3 hence this clustering is optimal when 3 clusters are used. We used k-means and k-mediods with k = 3 in following clustering experiments.

# **Apply k-means clustering with k=3**

# In[ ]:


kmeans = KMeans(n_clusters=3)
kmeans.fit(features)
cluster_labels = kmeans.fit_predict(features)

kmeans.cluster_centers_


# **Calculate silhouette coefficient for above clustering**

# In[ ]:


silhouette_avg = metrics.silhouette_score(features, cluster_labels)
print ('silhouette coefficient for the above clutering = ', silhouette_avg)


# o.47 would be an average value for silhouette coeffiencit. -1 is the worst and +1 is the optimal.
# Read more about silhouette coeffient at wikipedia article. And read the paper [here](https://www.sciencedirect.com/science/article/pii/0377042787901257).

# **Calculate Purity of the above clustering**
# 
# You can use [this](https://pml.readthedocs.io/en/latest/clustering.html) library as well.

# In[ ]:


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 

purity = purity_score(target, cluster_labels)
print ('Purity for the above clutering = ', purity)


# **Install [pyclustering](https://pypi.org/project/pyclustering/) on kernel**
# 
# Enable internet setting for the kernel and run the following command to install custom python packge.
# Refer documentation for the package [here](https://codedocs.xyz/annoviko/pyclustering/classpyclustering_1_1cluster_1_1kmedoids_1_1kmedoids.html#a368ecae21ba8fabc43487d5d72fcc97e).

# In[ ]:


get_ipython().system('pip install pyclustering')

from pyclustering.cluster.kmedoids import kmedoids


# **Apply k-mediods clustering with k=3**

# In[ ]:


# Randomly pick 3 indexs from the original sample as the mediods
initial_medoids = [1, 50, 170]

# Create instance of K-Medoids algorithm with prepared centers.
kmedoids_instance = kmedoids(features.values.tolist(), initial_medoids)

# Run cluster analysis.
kmedoids_instance.process()

# predict function is not availble in the release branch yet.
# cluster_labels = kmedoids_instance.predict(features.values)

clusters = kmedoids_instance.get_clusters()

# Prepare cluster labels
cluster_labels = np.zeros([210], dtype=int)
for x in np.nditer(np.asarray(clusters[1])):
   cluster_labels[x] = 1
for x in np.nditer(np.asarray(clusters[2])):
   cluster_labels[x] = 2

cluster_labels


# In[ ]:


# Mediods found in above clustering, indexes are shouwn below.
kmedoids_instance.get_medoids()


# **Calculate silhouette coefficient for above clustering**

# In[ ]:


silhouette_avg = metrics.silhouette_score(features, cluster_labels)
print ('silhouette coefficient for the above clutering = ', silhouette_avg)


# **Calculate Purity of the above clustering**

# In[ ]:


purity = purity_score(target, cluster_labels)
print ('Purity for the above clutering = ', purity)


# **Conclusion**
# 
# 1. Both silhouette coefficient and purity values are very close, hence both clustering are done similarity. K-means is slightly better in both measures.
# 2. K-mediods results are sensitive to the initial mediods selected.

# In[ ]:




