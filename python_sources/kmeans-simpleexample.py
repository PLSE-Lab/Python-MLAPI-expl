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


plt.scatter(x,y)
plt.show()
#Yes, this way.


# In[ ]:


from sklearn.cluster import KMeans
WCSS = [] #within clusters sum of squares


# In[ ]:


#hey, i have a problem, how can i know optimum k value?
#sure, i prefer elbow rules.
for k in range(1,15):
    model = KMeans(n_clusters = k)
    model.fit(df)
    WCSS.append(model.inertia_)


# In[ ]:


#can you see elbow point? i guess it is 3.
plt.plot(range(1,15), WCSS)


# In[ ]:


#yes we can do it, anymore.
model = KMeans(n_clusters = 3)
clusters = model.fit_predict(df)


# In[ ]:


# we can create new columns, labels!
df["labels"] = clusters


# In[ ]:


plt.scatter(df.x[df.labels == 0], df.y[df.labels == 0], color="red")
plt.scatter(df.x[df.labels == 1], df.y[df.labels == 1], color="green")
plt.scatter(df.x[df.labels == 2], df.y[df.labels == 2], color="blue")
plt.show()
#did you see, it is perfect example of unsupervised learning.


# In[ ]:


plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color = "black")


# In[ ]:


#if you wanna say that "where is the cluster centers ???" here you go.
plt.scatter(df.x[df.labels == 0], df.y[df.labels == 0], color="red")
plt.scatter(df.x[df.labels == 1], df.y[df.labels == 1], color="green")
plt.scatter(df.x[df.labels == 2], df.y[df.labels == 2], color="blue")
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color = "black")
plt.show()


# In[ ]:


#if you have any question or suggestion, i will be happy to hear it.

