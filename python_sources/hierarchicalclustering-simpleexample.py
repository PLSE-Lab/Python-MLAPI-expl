#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#importing necessary libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[ ]:


#we can create own dataset. (gaussian distributation)

x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)


# In[ ]:


#almost dataset is ready.
x = np.concatenate((x1,x2,x3), axis = 0)
y = np.concatenate((y1,y2,y3), axis = 0)


# In[ ]:


dictionary = {"x":x, "y":y}


# In[ ]:


dictionary


# In[ ]:


df = pd.DataFrame(dictionary)
#dataset is ready, it called df.


# In[ ]:


plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()
#Unsupervised learning, it does'nt know labels.


# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram
#we need to import scipy libraries.


# In[ ]:


merg = linkage(df, method = "ward")
dendrogram(merg, leaf_rotation = 90)
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

#we should create dendrogram graphics. out distance type euclidean.


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
# we should import AgglomerativeClustering from cluster libraries.


# In[ ]:


model = AgglomerativeClustering(n_clusters=3, affinity = "euclidean", linkage="ward")
cluster = model.fit_predict(df)


# In[ ]:


df["labels"] = cluster
#we can create new column about labels.


# In[ ]:


plt.scatter(df.x[df.labels == 0], df.y[df.labels == 0], color="red")
plt.scatter(df.x[df.labels == 1], df.y[df.labels == 1], color="green")
plt.scatter(df.x[df.labels == 2], df.y[df.labels == 2], color="blue")
plt.show()
#our model is working! almost perfect.

