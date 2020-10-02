#!/usr/bin/env python
# coding: utf-8

# In[12]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import seaborn as sns
import matplotlib.pyplot as plt

# Any results you write to the current directory are saved as output.


# # Loading data
# It's a small dataset with only 178 enteries.

# In[22]:


df = pd.read_csv("../input/Wine.csv")
df.info()


# As seen from above there are no null values in this dataset. Now let's remove the 'Customer_Segment' column as we are clustering and there is no need of this target variable. 

# In[23]:


df.head()


# Let's drop the Customer_Segment column as there is no need of that because we are doing clustering here. 

# In[24]:


df.drop('Customer_Segment',axis = 1, inplace = True)


# In[25]:


get_ipython().run_line_magic('matplotlib', 'inline')
for i in df.columns:
    plt.figure(figsize=(8,8))
    df[i].hist()
    plt.xlabel(str(i))
    plt.ylabel("Frequency")


# In[26]:


corrmat = df.corr(method='spearman')
f, ax = plt.subplots(figsize=(8, 8))

# Draw the heatmap using seaborn
sns.heatmap(corrmat, vmax=.8, square=True)
plt.show()


# In[21]:


y = []
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
matrix = df.as_matrix()
for n_clusters in range(2,30):
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=30)
    kmeans.fit(matrix)
    clusters = kmeans.predict(matrix)
    silhouette_avg = silhouette_score(matrix, clusters)
    y.append(silhouette_avg)
    print("For n_clusters =", n_clusters, "The average silhouette_score is :", silhouette_avg)

plt.figure(figsize=(12,8))
plt.plot(range(2,30),y)
plt.xlabel('No of Clusters')
plt.ylabel('Silhouette_avg')
plt.title('Silhoutte Score for different clusters')


# In[ ]:




