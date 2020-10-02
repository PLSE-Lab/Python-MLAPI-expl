#!/usr/bin/env python
# coding: utf-8

# Clustering Analysis in Python

# Clustering is an unsupervised machine learning technique,this is because unlike supervised machine learning approach where the ground truth is labeled(the variable to be predicted/dependent variable),unsupervised learning algorithm takes no such labled data as input,rather this technique is suited for data where the ground reality is not available but the data could be grouped into categories.
# 
# 

# Type of clustering algorithms
# 1.Partition Clustering
# 2.Hierarchial Clustering
# 3.Fuzzy clustering

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.cluster import KMeans
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelBinarizer
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


iris = pd.read_csv("../input/Iris.csv")


# In[ ]:


iris.info()


# In[ ]:


iris.head()


# Splitting dependent and independent variables.

# In[ ]:


y = iris.Species


y.head()


# In[ ]:


x = iris[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]


# In[ ]:


x
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y = encoder.fit_transform(y)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 101)


# **Partition Clustering**
#     
#     This method is used to partition the data to clusters which are non-overlapping.
#     The most commonly used method for this type of clustering is **k-means **clustering.
#     
# Algorithm,
# 
# 1.Begin with a decision on the number of clusters or 'K'.
# 2.Put any initial partition that classifies into number of clusters on the data.
# 3.Take each sample and compute the distance from the cluster center.
# 4.If the data point is closer to the cluster center,then add the point to the custer.
# 5.Repeat steps until non overlapping clusters are formed.
# Having described about the working of this technique,we will apply this technique to identify the type of species.
# 

# In[ ]:


kms = KMeans(n_clusters=3)
y = y.reshape(-1, 1)
kms.fit(y)


# In[ ]:


print(kms.cluster_centers_)

