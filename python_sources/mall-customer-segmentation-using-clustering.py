#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


dataset = pd.read_csv('../input/Mall_Customers.csv')


# ### Checking the head of the dataset

# In[ ]:


dataset.head()


# In[ ]:


x = dataset.iloc[:,[3,4]].values


# ## EDA

# ### Dendrogram

# In[ ]:


import scipy.cluster.hierarchy as sch


# In[ ]:


dendrogram = sch.dendrogram(sch.linkage(x,method = 'ward'))
plt.title("Dendrogram")
plt.xlabel('Customers')
plt.ylabel('Eucledian Distance')


# **ward is a method we chosen to build dendrogram,it will minimize within cluster variance,in KMEANS we will minimize within cluster sum of square** 

# **If we draw a threshold line in the dendrogram then we will get 5 clusters**

# ## Hierarchial Clustering

# In[ ]:


from sklearn.cluster import AgglomerativeClustering


# In[ ]:


cluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')


# In[ ]:


y_pred = cluster.fit_predict(x)


# In[ ]:


y_pred


# ## Clusters

# In[ ]:


plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s = 100, c = 'red', label = 'Careful')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s = 100, c = 'blue', label = 'Standard')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s = 100, c = 'green', label = 'Target')
plt.scatter(x[y_pred == 3, 0], x[y_pred == 3, 1], s = 100, c = 'cyan', label = 'Careless')
plt.scatter(x[y_pred == 4, 0], x[y_pred == 4, 1], s = 100, c = 'magenta', label = 'Sensible')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()


# In[ ]:




