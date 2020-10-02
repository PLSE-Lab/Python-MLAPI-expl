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
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 20:27:44 2019

@author: USER
"""

# importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the datasets
#dataset= pd.read_csv(r"C:\Users\USER\Desktop\Python\My_Project\Crd_Data_CLUSTERING\CC GENERAL.csv")
dataset= pd.read_csv('../input/CC GENERAL.csv')

# checking the presence of null values
dataset.isnull().sum()
 #CREDIT_LIMIT                          1
 #MINIMUM_PAYMENTS                    313
 
# Treating Credit limit
dataset.CREDIT_LIMIT.mean() #4494
dataset.CREDIT_LIMIT.mode() #3000
dataset.CREDIT_LIMIT.median() #3000
dataset.CREDIT_LIMIT.std() #3639
 
dataset['CREDIT_LIMIT'].fillna(3000, inplace = True)
dataset.isnull().sum()
 
# Treating MINIMUM_PAYMENTS
dataset.MINIMUM_PAYMENTS.mean() #864
dataset.MINIMUM_PAYMENTS.median() # 312
dataset.MINIMUM_PAYMENTS.mode() #300
dataset.MINIMUM_PAYMENTS.std() #2372
 
dataset['MINIMUM_PAYMENTS'].fillna(864, inplace = True) # using mean
dataset.isnull().sum()
# We see no null values

dataset.drop(['CUST_ID'], axis= 1, inplace = True)

# No Categorical Values found
X = dataset.iloc[:,:].values

# Using standard scaler
from sklearn.preprocessing import StandardScaler
standardscaler= StandardScaler()
X = standardscaler.fit_transform(X)

"""K MEANS CLUSTERING """
from sklearn.cluster import KMeans
wss= []
for i in range(1, 11):
    kmeans= KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    wss.append(kmeans.inertia_)

plt.plot(range(1,11), wss) # seelecting 4

kmeans = KMeans(n_clusters = 4, init= 'k-means++', random_state = 0)
kmeans.fit(X)
Y_pred_K= kmeans.predict(X) 

""" Now Trying Hierachial Clustering method """
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X, method= 'ward', metric= 'euclidean'))
# selecting 2 categories

from sklearn.cluster import AgglomerativeClustering
D_cluster = AgglomerativeClustering(n_clusters= 2)

X_D=D_cluster.fit(X)
Y_pred_D = D_cluster.fit_predict(X)
