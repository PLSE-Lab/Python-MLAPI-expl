#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
cars = pd.read_csv("cars.cv")
cars.head()


# In[ ]:


X = cars.iloc[]values
X


# In[ ]:


for in range(1,11):
    kmeans = KMeans(n_clusters = 'random')
    kmeans.fit(X)
    print i,kmeans.inertia_
    wcss.append(kmeans.inertia_)
    plt.plot(range(1,11),wcss)
    plt.title('O Metodo Elbow')
    plt.xlabel('Numero de Clusters')
    plt.ylabel('WSS')
    plt.show()


# In[ ]:


plt.scatter(X[:, 0], X[:,1], s = 100, c = kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'red',label = 'Centroids')
plt.title('Cars')
plt.xlabel('SepalLength')
plt.ylabel('SepalWidth')
plt.legend()

plt.show()

