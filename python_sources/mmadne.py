#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install skylearn')


# In[ ]:


# support for large, multi-dimensional arrays and matrices
import numpy as np
# plotting library for the Python programming language and its numerical mathematics extension NumPy
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")
#for less number of datapoints KMeans for more than thousand of points 
#KMeansbatch can be used
from sklearn.cluster import KMeans


# In[ ]:


x= [1.8,3.1,2,6,5.5,6,2,5.3]
y= [2.1, 3, 2.8, 7, 6.6, 9,3.2,6.9]
plt.scatter(x,y)
plt.show()


# In[ ]:


#converting above co-ordinates into numpy list of list
X= np.array([[0.8,2],[1.5,4],[1, 1.8],[8,8],[6, 9.6],[7.5,10],[2, 3.6],[7.3,7.9]])


# In[ ]:


#assign number of clusters to machine
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)


# In[ ]:


#clustering performed based on equal variance from centroid. 
centroids = kmeans.cluster_centers_
#semisupervised approach based on these label new data 
#points will be assigned label by algorithm.
labels = kmeans.labels_


# In[ ]:


print(centroids)
print(labels)


# In[ ]:


colors = ["120","240"]


# In[ ]:


for i in range(len(X)):
    print("coordinate:", X[i], "label:", labels[i])
    plt.scatter(x,y, c=[colors[l_] for l_ in labels], label=labels)
    plt.scatter(centroids[:, 0],centroids[:, 1], c=[c for c in colors[:len(centroids)]], marker = "x", s=150, linewidths = 5, zorder = 10)
    plt.show()


# In[ ]:




