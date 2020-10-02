#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


# read data
data = pd.read_csv('../input/sample-dataset-for-clustering/KMeans Dataset.csv')
# sample data
data.head()


# In[ ]:


# data describe
data.describe()


# In[ ]:


# plot income and spend of money
plt.figure(figsize=[16, 5])
plt.plot(data['INCOME'], color='green', label='Income')
plt.plot(data['SPEND'], color='red', label='Spend')
plt.legend()
plt.show()


# In[ ]:


# I want to know how much money have after money spend
data['Net Money'] = data['INCOME'] - data['SPEND']
plt.figure(figsize=[16, 5])
plt.plot(data['Net Money'], color='blue', label='Net Money')
plt.legend()
plt.show()


# In[ ]:


# use min, max and mean to make initila centeroid
wasteful = data.iloc[data['Net Money'].idxmin(), :].values
moderate = np.array([data['INCOME'].mean(), data['SPEND'].mean(), data['Net Money'].mean()])
thrifty = data.iloc[data['Net Money'].idxmax(), :].values
centeroid = np.array([wasteful, moderate, thrifty])


# In[ ]:


centeroid


# In[ ]:


X = data.iloc[:, [1]].values


# In[ ]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


# make model
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=100, random_state = 42)


# In[ ]:


# cluster
data['Result'] = kmeans.fit_predict(data)


# In[ ]:


data['INCOME'][data['Result']==0]


# In[ ]:


# plot
plt.figure(figsize=[7, 7])
plt.scatter(data['INCOME'][data['Result']==0], data['SPEND'][data['Result']==0], color='red', label='Wasteful')
plt.scatter(data['INCOME'][data['Result']==1], data['SPEND'][data['Result']==1], color='blue', label='Moderate')
plt.scatter(data['INCOME'][data['Result']==2], data['SPEND'][data['Result']==2], color='green', label='Thrifty')
plt.legend()
plt.xlabel('Income')
plt.ylabel('Spend')
plt.show()


# In[ ]:


plt.figure(figsize=[7, 7])
plt.scatter(data['INCOME'][data['Result']==0], data['SPEND'][data['Result']==0], color='red', label='Wasteful')
plt.scatter(data['INCOME'][data['Result']==1], data['SPEND'][data['Result']==1], color='blue', label='Moderate')
plt.scatter(data['INCOME'][data['Result']==2], data['SPEND'][data['Result']==2], color='green', label='Thrifty')
plt.scatter(data['INCOME'][data['Result']==3], data['SPEND'][data['Result']==3], color='black', label='target')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')

plt.legend()
plt.xlabel('Income')
plt.ylabel('Spend')
plt.show()


# In[ ]:




