# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
#print(os.listdir("../input/College.csv"))

# Any results you write to the current directory are saved as output.
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 20:05:53 2019

@author: USER
"""
# Importing the Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
#dataset = pd.read_csv("../input/college.csv")
dataset=pd.read_csv("../input/College.csv")

# Checking the presence of Null Values
dataset.isnull().sum()

# Checking the presence of Categorical Values
dataset.dtypes # 2 present

#dataset.drop(['Unnamed: 0'], axis = 1, inplace = True) # We do not need this
dataset= dataset.iloc[:, 1:]

Private = pd.get_dummies(dataset['Private'], drop_first = True)

dataset.drop(['Private'], axis= 1, inplace= True)
new_dataset = pd.concat([Private, dataset], axis = 1)

X= new_dataset.iloc[:, :].values
# We need to scle this
from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X= sc.fit_transform(X)

# Using Kmeans Clustering
from sklearn.cluster import KMeans
wss=[]
for i in range (1,11):
    kmeanscluster = KMeans(n_clusters = i, init = 'k-means++')
    kmeanscluster.fit(X)
    kmeanscluster.inertia_
    wss.append(kmeanscluster.inertia_)
    
plt.plot(range(1,11), wss) # Selecting 5

kmeanscluster = KMeans(n_clusters= 5, init = 'k-means++')
kmeanscluster.fit(X)
Y_pred = kmeanscluster.predict(X)

""" now trying Dendrogram """
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method= 'ward', metric = 'euclidean'))
# Selecting 3

from sklearn.cluster import AgglomerativeClustering
cluster= AgglomerativeClustering(n_clusters  = 3)

cluster.fit(X)
Y_pred_dendrogram = cluster.fit_predict(X)

    
