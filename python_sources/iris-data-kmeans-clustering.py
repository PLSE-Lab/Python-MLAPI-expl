#!/usr/bin/env python
# coding: utf-8

# K Means Clustering

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

ir = load_iris()
# adding column names for iris data
iris = pd.DataFrame(ir.data, columns= (ir.feature_names))
iris.head()


# In[ ]:


# removing two columns and going to work on other two columns
iris.drop(['sepal length (cm)','sepal width (cm)'], axis='columns',  inplace=True)
iris.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


#scatter plot with existing data
plt.scatter(iris['petal length (cm)'], iris['petal width (cm)'] )


# After checking above plot we can decide it as two clusters and preparing model

# In[ ]:


km = KMeans(n_clusters= 2)
km
y_pre = km.fit_predict(iris[['petal length (cm)','petal width (cm)']])


# In[ ]:


# adding new column as cluster with predicted cluster data
iris['cluster'] = y_pre
iris.head()


# creating diffrenct dataframes with cluster data

# In[ ]:


iris1 = iris[iris['cluster']==0] 
iris2 = iris[iris['cluster']==1] 


# Scatter Plot with cluster data

# In[ ]:


plt.scatter(iris1['petal length (cm)'], iris1['petal width (cm)'] )
plt.scatter(iris2['petal length (cm)'], iris2['petal width (cm)'] )


# After checking above plot we need to adjust scalling and below scalling for two columns.

# In[ ]:


scaler = MinMaxScaler()
scaler.fit(iris[['petal width (cm)']])
iris['petal width (cm)'] = scaler.transform(iris[['petal width (cm)']])


scaler = MinMaxScaler()
scaler.fit(iris[['petal length (cm)']])
iris['petal length (cm)'] = scaler.transform(iris[['petal length (cm)']])

iris.head()


# model with scalling data

# In[ ]:


km = KMeans(n_clusters= 2)
km
y_pre = km.fit_predict(iris[['petal length (cm)','petal width (cm)']])


# In[ ]:


iris['cluster'] = y_pre
iris.head()


# creating diffrenct dataframes with cluster scalling data

# In[ ]:


iris1 = iris[iris['cluster']==0] 
iris2 = iris[iris['cluster']==1] 


# Scatter Plot with cluster scalling data

# In[ ]:


plt.scatter(iris1['petal length (cm)'], iris1['petal width (cm)'], label = 'petal width (cm)' )
plt.scatter(iris2['petal length (cm)'], iris2['petal width (cm)'], label = 'petal width (cm)')
plt.legend()


# In[ ]:


# to check centroid values for clusters 
km.cluster_centers_


# Adding centroids for each clusters in scatter plot

# In[ ]:


plt.figure(figsize = (12,6))
plt.scatter(iris1['petal length (cm)'], iris1['petal width (cm)'], label = 'petal width (cm)' )
plt.scatter(iris2['petal length (cm)'], iris2['petal width (cm)'], label = 'petal width (cm)')
plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1] , color = 'r', marker = '*', label ='centroid', s = 100 )
plt.legend()


# sum of squared error and Elbow - criterion:

# In[ ]:


k_rng = range(1,10)
ssr = []
for k in k_rng:
    km = KMeans(n_clusters= k)
    km.fit(iris)
    ssr.append(km.inertia_)


# In[ ]:


# to see ssr values
ssr


# So the goal is to choose a small value of k that still has a low SSE, and the elbow usually represents where we start to have diminishing returns by increasing k

# In[ ]:


plt.figure(figsize = (12,6))
plt.plot(k_rng, ssr)
plt.ylabel('sum of squared error', fontsize = 14)
plt.xlabel('Range', fontsize = 14)


# In the above plot no. of opitimal clusters are 3 that means K =3

# Feel free to comment with suggestions or any clarifications for further active discussions. 
# 
# Name: Mahesh Kumar
# e-mailid: mcommahesh@gmail.com
# 
