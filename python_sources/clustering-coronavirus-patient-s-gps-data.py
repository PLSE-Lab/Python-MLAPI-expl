#!/usr/bin/env python
# coding: utf-8

# **1. Add Required Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv


# **2. Import Dataset**

# In[ ]:


path = '/kaggle/input/coronavirusdataset/'
route_data_path = path + 'route.csv'

df_route = pd.read_csv(route_data_path)


# In[ ]:


df_route.head(10)


# In[ ]:


df_route.tail(10)


# **3. Remove rows where the Longitude and/or Latitude are null values (if any)**

# In[ ]:


df_route.dropna(axis=0,how='any',subset=['latitude','longitude'],inplace=True)


# **4. Create data with related variables/features**

# In[ ]:


# Variable with the Longitude and Latitude
X=df_route.loc[:,['id','latitude','longitude']]
X.head(10)


# In[ ]:


X.tail(10)


# **5. Check optimal number of clusters using Elbow Curve analysis**

# In[ ]:


K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = df_route[['latitude']]
X_axis = df_route[['longitude']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# In the plot, we see that the graph levels off slowly after 4 clusters. 
# This implies that addition of more clusters will not help us that much.

# **6. Clustering using K-means algorithm**
# 
# In this case, we will try to run 4 number of clusters

# In[ ]:


kmeans = KMeans(n_clusters = 4, init ='k-means++')
kmeans.fit(X[X.columns[1:3]]) # Compute k-means clustering.
X['cluster_label'] = kmeans.fit_predict(X[X.columns[1:3]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(X[X.columns[1:3]]) # Labels of each point


# **7. Display the result**

# In[ ]:


X.plot.scatter(x = 'latitude', y = 'longitude', c=labels, s=50, cmap='viridis')
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)


# In[ ]:


X[0:60]

