#!/usr/bin/env python
# coding: utf-8

# In[7]:



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data =pd.read_csv("../input/CC GENERAL.csv")
missing = data.isna().sum()
print(missing)


# In[15]:


df=df.fillna(df.median())

X=df.iloc[:,1:].values


#  Using kmeans algorithm 

# In[16]:


from sklearn.cluster import KMeans
wcss=[]
for i in range(1,30):
    kmeans = KMeans(n_clusters=i, init="k-means++", n_init=10, max_iter=300) 
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
y_kmeans = kmeans.fit_predict(X)


# In[20]:


#visualisation

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'black', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

