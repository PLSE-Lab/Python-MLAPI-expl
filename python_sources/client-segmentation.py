#!/usr/bin/env python
# coding: utf-8

# # An application of clustering with K-means
# Video: https://youtu.be/xBIs5_ic5hU

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


# In[ ]:


df = pd.read_csv("../input/dataset.csv")
df.columns = ["CustomerID", "Gender", "Age", "Annual_Income", "Spending_Score"]
df.head(5)


# In[ ]:


df.shape


# In[ ]:


df.isnull().sum()


# In[ ]:


X = df.iloc[:, [3,4]].values
X


# In[ ]:


scaler = StandardScaler()
X = scaler.fit_transform(X)
X


# In[ ]:


df.iloc[:, [3,4]] = X


# In[ ]:


plt.scatter(X[:, 0], X[:, 1])
plt.show()


# In[ ]:


sse = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    
plt.plot(range(1,11), sse)
plt.title("Elbow")
plt.xlabel("number of clusters")
plt.ylabel("sse")
plt.show()


# In[ ]:


n_clusters = 5
model = KMeans(n_clusters=n_clusters, init="k-means++")
pred = model.fit_predict(X)


# In[ ]:


plt.figure(figsize=(20,10))
for i in range(0, n_clusters):
    plt.scatter(X[pred == i, 0], X[pred == i, 1], s=50, label="Cluster %d" % i)
    
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], s = 100, c = 'black', label='centroids')
plt.title("Clusters")
plt.xlabel("Annual Income")
plt.ylabel("Spending_Score")
plt.legend()
plt.show()


# # Who are the customers of best cluster?
# Cluster 0 is our TARGET (earning high and also spending high). Who are the customers of this cluster?

# In[ ]:


for row in X[pred == 0]:
    print(df[(df.iloc[:, 3] == row[0]) & (df.iloc[:,4] == row[1])]['CustomerID'].values)


# In[ ]:




