#!/usr/bin/env python
# coding: utf-8

# In[22]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[23]:


from sklearn.cluster import KMeans


# In[24]:


# K-Means++ Initialization by default.
cluster_count = 3
clustering_model = KMeans(n_clusters=cluster_count)


# In[25]:


import random

def generate_random_data(lower_bound=(0,0), upper_bound=(100,100), samples=100):
    random_samples = []
    for _ in range(samples):
        lat = random.randrange(lower_bound[0], upper_bound[0])
        lon = random.randrange(lower_bound[1], upper_bound[1])
        random_samples.append([lat, lon])
    return random_samples

X_train = np.array(generate_random_data())


# In[26]:


import matplotlib.pyplot as plt

plt.scatter(X_train[:,0], X_train[:,1])
plt.xlabel("lat")
plt.ylabel("lon")
plt.show()


# In[27]:


clustering_model.fit(X_train)


# In[28]:


predicted_labels = clustering_model.predict(X_train)


# In[31]:


clusters = []
for _ in range(cluster_count):
    clusters.append([])

for idx, predicted_label in enumerate(predicted_labels):
    clusters[predicted_label].append([X_train[idx, 0], X_train[idx, 1]])


# In[34]:


for cluster in clusters:
    cluster_array = np.array(cluster)
    plt.scatter(cluster_array[:,0], cluster_array[:,1])

    
# Plot centers.
plt.scatter(clustering_model.cluster_centers_[:, 0], clustering_model.cluster_centers_[:, 1],
            marker='x')
    
plt.xlabel("lat")
plt.ylabel("lon")
plt.show()


# In[ ]:




