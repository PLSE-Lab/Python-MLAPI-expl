#!/usr/bin/env python
# coding: utf-8

# Hello everyone
# 
# In this kernel we will learn K-means clustering and Hierarchical clustering.
# 
# Let's begin

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# 
# # UNSUPERVISED LEARNING
# * Unsupervised learning: It uses data that has unlabeled and uncover hidden patterns from unlabeled data. Example, there are orthopedic patients data that do not have labels. You do not know which orthopedic patient is normal or abnormal.
# * As you know orthopedic patients data is labeled (supervised) data. It has target variables. In order to work on unsupervised learning, lets drop target variables and to visualize just consider pelvic_radius and degree_spondylolisthesis

# 
# # KMEANS
# * Lets try our first unsupervised method that is KMeans Cluster
# * KMeans Cluster: The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity
# * KMeans(n_clusters = 2): n_clusters = 2 means that create 2 cluster

# In[ ]:


# As you can see there is no labels in data
data = pd.read_csv('../input/column_2C_weka.csv')
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'])
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# In[ ]:


# KMeans Clustering
data2 = data.loc[:,['degree_spondylolisthesis','pelvic_radius']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = labels)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# # EVALUATING OF CLUSTERING
# We cluster data in two groups. Okey well is that correct clustering? In order to evaluate clustering we will use cross tabulation table.
# 
# * There are two clusters that are 0 and 1
# * First class 0 includes 138 abnormal and 100 normal patients
# * Second class 1 includes 72 abnormal and 0 normal patiens *The majority of two clusters are abnormal patients.

# In[ ]:


# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# The new question is that we know how many class data includes, but what if number of class is unknow in data. This is kind of like hyperparameter in KNN or regressions.
# 
# * inertia: how spread out the clusters are distance from each sample
# * lower inertia means more clusters
# * What is the best number of clusters ? *There are low inertia and not too many cluster trade off so we can choose elbow

# In[ ]:


# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


# # STANDARDIZATION
# * Standardizaton is important for both supervised and unsupervised learning
# * Do not forget standardization as pre-processing
# * As we already have visualized data so you got the idea. Now we can use all features for clustering.
# * We can use pipeline like supervised learning.

# In[ ]:


data = pd.read_csv('../input/column_2C_weka.csv')
data3 = data.drop('class',axis = 1)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# # HIERARCHY
# * vertical lines are clusters
# * height on dendogram: distance between merging cluster
# * method= 'single' : closest points of clusters

# In[ ]:


from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3.iloc[200:220,:],method = 'single')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()

