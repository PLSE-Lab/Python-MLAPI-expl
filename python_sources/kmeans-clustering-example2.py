#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[ ]:


df = pd.read_csv('../input/customers.csv')
df.columns = ["Id","Gender","Age","Volume", "Salary"]
df = df.iloc[:,1:] # We don't need ID columns.
df.Gender = [1 if each == "K" else 0 for each in df.Gender] # Converting to boolean variables.


# In[ ]:


df.head()


# In[ ]:


x = df.iloc[:,2:]
#This is unsupervised learning and i can use all of the columns, but i want to visualization, so i choosed 2 columns.


# In[ ]:


model = KMeans(n_clusters = 3, init = 'k-means++')
model.fit(x)


# In[ ]:


print(model.cluster_centers_)
#This coordinate points, center of clusters of my algorithm.


# In[ ]:


plt.title('Salary vs Volume Graphic')
plt.xlabel('Volume')
plt.ylabel('Salary')
plt.scatter(x.iloc[:,0], x.iloc[:,1], color="yellow")
plt.show()


# In[ ]:


plt.title('Salary vs Volume Graphic and Centers of Clusters')
plt.xlabel('Volume')
plt.ylabel('Salary')
plt.scatter(x.iloc[:,0], x.iloc[:,1], color="yellow")
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color="red") # Black is cluster centers
plt.show()


# In[ ]:


conclusion = []
for each in range(1,11):
    model = KMeans(n_clusters = each, init = 'k-means++', random_state = 123)
    model.fit(x)
    conclusion.append(model.inertia_)


# In[ ]:


plt.plot(range(1,11), conclusion)
plt.show()
# Did you see? Our elbow point is 4. It means, we should use 4 cluster.


# In[ ]:


model = KMeans(n_clusters = 4, init = 'k-means++', random_state = 123)
clusters = model.fit_predict(x)


# In[ ]:


x["label1"] = clusters


# In[ ]:


x.sample(10)


# In[ ]:


plt.title('Salary vs Volume Clustering')
plt.xlabel('Volume')
plt.ylabel('Salary')
plt.scatter(x.Volume[x.label1 == 0], x.Salary[x.label1 == 0], color="red", alpha=0.4)
plt.scatter(x.Volume[x.label1 == 1], x.Salary[x.label1 == 1], color="blue",alpha=0.4)
plt.scatter(x.Volume[x.label1 == 2], x.Salary[x.label1 == 2], color="green",alpha=0.4)
plt.scatter(x.Volume[x.label1 == 3], x.Salary[x.label1 == 3], color="purple",alpha=0.4)
plt.scatter(model.cluster_centers_[:,0], model.cluster_centers_[:,1], color="black") # Black is cluster centers
plt.show()


# In[ ]:


# Let's start again , using all of the columns.
conclusion = [] 
for each in range(1,11):
    model = KMeans(n_clusters = each, init = 'k-means++', random_state = 123)
    model.fit(df)
    conclusion.append(model.inertia_)


# In[ ]:


plt.plot(range(1,11), conclusion)
plt.show()
#Again 4, sure.


# In[ ]:


model = KMeans(n_clusters = 4, init = 'k-means++', random_state = 123)
clusters = model.fit_predict(x)


# In[ ]:


df["label2"] = clusters
# df[label2] output, all of the columns.
# x[label1] output, just two columns.


# In[ ]:


df["label1"] = x["label1"]


# In[ ]:


df.sample(12)
#Almost label1 and label2 is same predicted.

