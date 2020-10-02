#!/usr/bin/env python
# coding: utf-8

# 1. [Introduction)](#1)
# 1. [Differences between Supervised and Unsupervised](#2)
#  1. [Unsupervised Algorithms](#3)
#     1. [K Means Algorithm](#4)
#        1. [Selection of k value](#5)
#        1. [Find optimum K value](#6)
#   1. [Standardization](#7)
#   1. [Hierarcial Clustering](#8)
#   1. [Principle Component Analysis](#9)
# 1. [Conclusion](#10)  

# <a id="1"></a> 
# ### Introduction
#  * Unsupervised learning: It uses data that has unlabeled and uncover hidden patterns from unlabeled data. Example, there are orthopedic patients data that do not    have labels. You do not know which orthopedic patient is normal or abnormal.
#  * As you know orthopedic patients data is labeled (supervised) data. It has target variables. In order to work on unsupervised learning, lets drop target variables and to visualize just consider pelvic_radius and degree_spondylolisthesis

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import plotly.figure_factory as ff
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/column_2C_weka.csv')


# In[ ]:


data.head()


# In[ ]:


# As you can see there is no labels in data
x = data['pelvic_radius']
y = data['degree_spondylolisthesis']
plt.figure(figsize=(13,5))
plt.scatter(x,y)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# <a id="2"></a> 
# ## Differences between Supervised and Unsupervised Algorithm 
# 
# <a href="https://ibb.co/MpqvNM9"><img src="https://i.ibb.co/xGVk5J3/1001.jpg" alt="1001" border="0"></a><br /><a target='_blank' href='https://babynamesetc.com/boy/unusual'>baby name malou</a><br />
# 
# <a href="https://imgbb.com/"><img src="https://i.ibb.co/0GL7vNt/100.jpg" alt="100" border="0"></a>
# 
# 

# <a id="3"></a> 
# ## Unsupervised Algorithms

# <a id="4"></a> 
# ### K Means Algorithm
# 
# * We choose a k value
# * Then it is randomly created k centroids
# * Every data point is clusterd according to the nearest centroid
# * By taking average of all data points that belog to a centroid, it is created new centroids.
# * Using these new centroids repeat 3 and 4
# * Finally, when centroids remain stationary, the algorith stops there.
# * As a result, according to these centroids, data is clustered
# 
# <a href="https://ibb.co/GcwmGn7"><img src="https://i.ibb.co/GcwmGn7/102.jpg" alt="102" border="0"></a>

# <a id="5"></a> 
# ### Selection of k value
# 
# 1. For k=1, run KMeans algorithm
# 2. For each cluster (k cluster we have), it is calculated WCSS (within cluster sum of squares) value
# 3. repeat 1 and 2 for 1<k<15
# 4. obtain k vs WCSS plot
# 5. Using elbow rule, choose the optimum k value to be used in K Means Algorithm

# <a id="6"></a> 
# ### Find optimum k value

# In[ ]:


df = data.loc[:, ['degree_spondylolisthesis', 'pelvic_radius']]
df.head()


# In[ ]:


# which k value to choose
from sklearn.cluster import KMeans
wcss = []
for k in range(1,15):
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)
    wcss.append(kmeans.inertia_) # kmeans.inertia : calculate wcss
    
plt.plot(range(1,15), wcss, '-o')
plt.xlabel('number of k (cluster) value')
plt.ylabel('wcss')
plt.show()


# * According to elbow rule we can select 2,3 or 4  but the elbow point is not quite obvious here

# In[ ]:


#if we choose k=2
from sklearn.cluster import KMeans
kmeans4 = KMeans(n_clusters = 2)
clusters =kmeans4.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters

# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# In[ ]:


#if we choose k=3
from sklearn.cluster import KMeans
kmeans4 = KMeans(n_clusters = 3)
clusters =kmeans4.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters

# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# In[ ]:


# if we choose k=4
from sklearn.cluster import KMeans
kmeans4 = KMeans(n_clusters = 5)
clusters =kmeans4.fit_predict(df) # fit first and then predict

# add labels for df
df['label'] = clusters

# plot
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = clusters)
plt.xlabel('pelvic_radius')
plt.xlabel('degree_spondylolisthesis')
plt.show()


# #### Original data is as follow

# In[ ]:


# plot
colors = [0 if i=='Abnormal' else 1 for i in data['class']] # to create colors
plt.scatter(data['pelvic_radius'],data['degree_spondylolisthesis'],c = colors)
plt.xlabel('pelvic_radius')
plt.ylabel('degree_spondylolisthesis')
plt.show()


# <a id="7"></a> 
# ### STANDARDIZATION
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


# <a id="8"></a> 
# ## Hierarcical Clustering
# * Assign each data point as a cluster
# * Create a new cluster by choosing the closest two clusters
# * Repeat 2 until it remains only one cluster

# <a href="https://ibb.co/xztqkwv"><img src="https://i.ibb.co/MPxkvKT/21.jpg" alt="21" border="0"></a><br /><a target='_blank' href='https://freeonlinedice.com/'>play craps for free</a><br />
# 
# <a href="https://ibb.co/mhrgcyX"><img src="https://i.ibb.co/nQqtL1n/22.jpg" alt="22" border="0"></a>
# 
# <a href="https://ibb.co/bJGQ4n1"><img src="https://i.ibb.co/pykbHp2/23.jpg" alt="23" border="0"></a>

# In[ ]:


# DENDOGRAM 
# here we will try to predict how many clusters we have 
from scipy.cluster.hierarchy import linkage, dendrogram # linkage: create dendrogram
df1 = data.loc[:, ['pelvic_radius', 'degree_spondylolisthesis']]
merg = linkage(df1, method='ward') # ward: cluster icindeki yayilimlari minimize et (wcss gibi bisey)
dendrogram(merg, leaf_rotation=90)
plt.xlabel('data points')
plt.ylabel('euclidian distance')
plt.show()


# <a id="9"></a> 
# ## Principle Component Analysis (PCA)
# 1. Fundemental dimension reduction technique
# 2. first step is decorrelation:
# 3. rotates data samples to be aligned with axes
# 4. shifts data asmples so they have mean zero
# 5. no information lost
# 6. fit() : learn how to shift samples
# 7. transform(): apply the learned transformation. It can also be applies test data
# 8. Resulting PCA features are not linearly correlated
# 9. Principle components: directions of variance

# In[ ]:


# PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)


# In[ ]:


# PCA variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)

plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()


# * Second step: intrinsic dimension: number of feature needed to approximate the data essential idea behind dimension reduction
# * PCA identifies intrinsic dimension when samples have any number of features
# * intrinsic dimension = number of PCA feature with significant variance
# * In order to choose intrinsic dimension try all of them and find best accuracy

# In[ ]:


# apply PCA
color_list=["red","blue"]
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list)
plt.show()


# <a id="10"></a>
# ## Conclusion
# 

# Reference : 
# * https://www.displayr.com/what-is-hierarchical-clustering/
# * https://www.kaggle.com/kanncaa1/machine-learning-tutorial-for-beginners

# In[ ]:




