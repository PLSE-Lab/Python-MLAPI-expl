#!/usr/bin/env python
# coding: utf-8

# > ** Author: Kazi Amit Hasan <br>**
# Department of Computer Science & Engineering,<br>
# Rajshahi University of Engineering & Technology (RUET)<br>
# Website: https://amithasanshuvo.github.io/<br>
# Linkedin: https://www.linkedin.com/in/kazi-amit-hasan-514443140/<br>
# Email: kaziamithasan89@gmail.com<br>
# 
# 
# If you want some basic ideas on clustering, then you can follow the following link: https://www.kaggle.com/getting-started/160596
# 
# *
# **Please leave your feedback and upvote if you like it.***
# 
# Ref:
# 
# 1. https://www.analyticsvidhya.com/blog/2019/08/comprehensive-guide-k-means-clustering/
# 2. https://heartbeat.fritz.ai/k-means-clustering-using-sklearn-and-python-4a054d67b187

# In[ ]:


# Importing all the important libraries that we need

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets


# In[ ]:


# Loading the dataset from sklearn datasets

df = pd.read_csv('../input/iris/Iris.csv')
#y = iris.target


# In[ ]:


x = df.drop(['Id', 'Species'], axis = 1) 


# In[ ]:


# A glance of the dataset

#x


# In[ ]:


# Defining the number of clusters an

kmeans5 = KMeans(n_clusters=5)

#Output of Kmeans clustering with value 5
y_kmeans5 = kmeans5.fit_predict(x)
print(y_kmeans5)


# In[ ]:



#Print the centers of 5 clusters
kmeans5.cluster_centers_


# In[ ]:


# Printing the interia value

#  Inertia actually calculates the sum of distances of all the points within
#a cluster from the centroid of that cluster. It tells us how far the points 
#within a cluster are. The distance between them should be as low as possible.


kmeans5.inertia_


# ## How can we decide the optimum number of clusters? 
# 
# We can do that with elbow method. 
# We will plot a graph, where the x-axis will represent the number of clusters and the y-axis will be an evaluation metric (inertia)

# In[ ]:


SSE =[]
for clusters in range(1, 11):
    kmeans = KMeans(n_clusters = clusters).fit(x)
    kmeans.fit(x)
    SSE.append(kmeans.inertia_)
import matplotlib.pyplot as plt
plt.plot(range(1, 11), SSE)
plt.title('Elbow method')
plt.xlabel('No of clusters')
plt.ylabel('Inertia')
plt.show()


# The cluster value where this decrease in inertia value becomes constant can be chosen as the right cluster value for our data. 
# 
# Here, Our optimal cluster value is between 3 and 4. So, let's select 3 as our num of clusters.
# 
# ## Repeat the same process with no_clusters = 3

# In[ ]:


kmeans3 = KMeans(n_clusters = 3)


# In[ ]:


y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)


# In[ ]:


# Printing the center points
kmeans3.cluster_centers_


# In[ ]:


# Let's see how many data points are in these 3 clusters.

frame = pd.DataFrame(x)
frame['cluster'] = y_kmeans3
frame['cluster'].value_counts()


# ## Visualize the data

# In[ ]:


plt.scatter(x.iloc[:,0],x.iloc[:,1], c = y_kmeans3, cmap='rainbow')


# In[ ]:




