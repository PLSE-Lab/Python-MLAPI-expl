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


pwd


# In[ ]:


cd ..


# In[ ]:


cd input/customer-segmentation-tutorial-in-python/


# In[ ]:


data = pd.read_csv('Mall_Customers.csv', index_col='CustomerID')


# In[ ]:


data.head()


# In[ ]:


data.info()
data.describe()


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


gender = pd.get_dummies(data['Gender'], drop_first=True)
data.head()


# In[ ]:


sns.set_style('whitegrid')
sns.lmplot(x='Annual Income (k$)', y='Spending Score (1-100)', data=data, hue='Gender', palette='hls', height=6, aspect=1, fit_reg=False)


# In[ ]:


sns.lmplot(x='Age', y='Annual Income (k$)', data=data, hue='Gender', palette='hls', height=6, aspect=1, fit_reg=False)


# In[ ]:


sns.lmplot(x='Age', y='Spending Score (1-100)', data=data, hue='Gender', palette='hls', height=6, aspect=1, fit_reg=False)


# In[ ]:


sns.set_style('darkgrid')
g = sns.FacetGrid(data=data, hue='Gender', aspect=1, height=6)
g = g.map(plt.hist, 'Spending Score (1-100)', bins = 20, alpha = 0.7)


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


def wcss(data):
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()


# In[ ]:


data['Sex'] = pd.concat([gender], axis=1)


# In[ ]:


data.head()


# In[ ]:


data.drop(['Gender'], axis=1, inplace=True)


# In[ ]:


data.head()


# In[ ]:


wcss(data)


# In[ ]:


kmeans = KMeans(n_clusters=6)


# In[ ]:


kmeans.fit(data)


# In[ ]:


pred_y = kmeans.fit_predict(data)


# In[ ]:


facet = sns.lmplot(data=data, x='Spending Score (1-100)', y='Annual Income (k$)', hue='Sex', fit_reg=False, legend=True, legend_out=True)

