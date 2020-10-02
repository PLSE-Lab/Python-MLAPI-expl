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





# In[ ]:


dataf=pd.read_csv("../input/CreditCardUsage.csv")


# In[ ]:


dataf.head()


# In[ ]:


dataf.info()


# In[ ]:


dataf.shape


# In[ ]:


dataf.isna().sum()


# In[ ]:


dataf.duplicated().sum()


# In[ ]:


dataf[dataf["PAYMENTS"]==0.00].shape


# In[ ]:


dataf["PAYMENTS"].value_counts()


# In[ ]:


dataf[dataf["PAYMENTS"]==0.00].shape


# In[ ]:


dataf["MINIMUM_PAYMENTS"].nunique()


# In[ ]:


import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot
from sklearn import cluster


# In[ ]:


data=dataf.drop(columns='CUST_ID')


# In[ ]:


data=data.fillna(0.00)


# In[ ]:





# In[ ]:


data.isna().sum()


# In[ ]:


df = dataf.drop('CUST_ID', axis=1)
df.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# In[ ]:


clusterNum = 7
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Sum of within sum square')
pl.title('Elbow Curve')
pl.show()


# In[ ]:


df["Clus_km"] = labels
df.head(5)


# In[ ]:


df["Clus_km"] = labels
df.head(5)


# In[ ]:


df.groupby('Clus_km').mean()


# In[ ]:


list(labels)


# In[ ]:





# In[ ]:





# In[ ]:


import scipy.cluster.hierarchy as sch

plt.figure(figsize=(15,6))
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distances')
#plt.grid(True)
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
plt.show()


# In[ ]:





# In[ ]:




