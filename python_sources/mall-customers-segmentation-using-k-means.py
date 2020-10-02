#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
sns.set_style('whitegrid')
from sklearn.cluster import KMeans
from copy import deepcopy


# In[ ]:


cust_df= pd.read_csv('../input/Mall_Customers.csv')


# In[ ]:


cust_df.head()


# In[ ]:


cust_df.describe()


# In[ ]:


cust_df.shape


# In[ ]:


plt.figure(figsize=(5,5))
sns.distplot(cust_df['Age'] , bins =50)


# In[ ]:


plt.figure(figsize=(5,5))
sns.distplot(cust_df['Spending Score (1-100)'] , bins =50)


# In[ ]:


plt.figure(figsize=(5,5))
sns.countplot("Gender",data = cust_df)


# In[ ]:


plt.figure(figsize=(5,5))
sns.lmplot(x="Annual Income (k$)",y="Spending Score (1-100)",data=cust_df,hue= "Gender")


# In[ ]:


X1 = cust_df[['Annual Income (k$)' , 'Spending Score (1-100)']]
inertia = []
for n in range(1 , 11):
    algorithm = (KMeans(n_clusters = n ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= None  , algorithm='auto') )
    algorithm.fit(X1)
    inertia.append(algorithm.inertia_)


# In[ ]:


plt.plot(range(1,11), inertia)
plt.title('The elbow method')
plt.xlabel('The number of clusters')
plt.ylabel('Inertia')
plt.show()


# Elbow point is at 5. Therefore Value of k is 5

# In[ ]:


f1 = cust_df['Annual Income (k$)'].values


# In[ ]:


f2 = cust_df['Spending Score (1-100)'].values


# In[ ]:


X = np.array(list(zip(f1, f2)))


# In[ ]:


plt.scatter(f1, f2, c='black', s=7)


# In[ ]:


k=5
C_x = np.random.randint(0, np.max(X)-20, size=k)
C_y = np.random.randint(0, np.max(X)-20, size=k)
C = np.array(list(zip(C_x, C_y)), dtype=np.float32)
print(C)


# In[ ]:


plt.scatter(f1, f2, c='#050505', s=7)
plt.scatter(C_x, C_y, marker='*', s=200, c='g')


# In[ ]:


#Euclidian calculator
def dist(a, b, ax=1):
    return np.linalg.norm(a - b, axis=ax)


# In[ ]:


# To store the value of centroids when it updates
C_old = np.zeros(C.shape)
clusters = np.zeros(len(X))
# Error func. - Distance between new centroids and old centroids
error = dist(C, C_old, None)
while error != 0:
    # Assigning each value to its closest cluster
    for i in range(len(X)):
        distances = dist(X[i], C)
        cluster = np.argmin(distances)
        clusters[i] = cluster
    # Storing the old centroid values
    C_old = deepcopy(C)
    # Finding the new centroids by taking the average value
    for i in range(k):
        points = [X[j] for j in range(len(X)) if clusters[j] == i]
        C[i] = np.mean(points, axis=0)
    error = dist(C, C_old, None)


# In[ ]:


colors = ['r', 'g', 'b', 'y', 'c', 'm']

fig, ax = plt.subplots()
for i in range(k):
        points = np.array([X[j] for j in range(len(X)) if clusters[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
ax.scatter(C[:, 0], C[:, 1], marker='*', s=200, c='#050505')


# In[ ]:




