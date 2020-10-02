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

"""
    @script-author: Joy Preetha
    @script-name: K_Means clustering
    @script-description: Implementation of K_Means clustering algorithm Iris dataset 
    @external-packages-used: sklearn, matplotlib
"""
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

iris=datasets.load_iris() #Load Iris dataset

X = iris.data[:, 2:4]  #Selecting independent variable
y = iris.target        ##Selecting target variable

plt.scatter(X[:,0], X[:,1], c=y, cmap='gist_rainbow')   #Plot the original graph
plt.xlabel('Peta1 Length', fontsize=18)
plt.ylabel('Peta1 Width', fontsize=18)

K_means = KMeans(n_clusters = 3, n_jobs = 4, random_state=21)   #Fitting K-means clustering model with 3 clusters
K_means.fit(X)

centers = K_means.cluster_centers_  #Get the final centroid values
print(centers)

new_labels = K_means.labels_   #Get Predicted values

#Plot for both Actual and Predicted values
fig, axes = plt.subplots(1, 2, figsize=(14,6))
axes[0].scatter(X[:, 0], X[:, 1], c=y, cmap='gist_rainbow',
edgecolor='k', s=150)
axes[1].scatter(X[:, 0], X[:, 1], c=new_labels, cmap='jet',
edgecolor='k', s=150)
axes[0].set_xlabel('Peta1 length', fontsize=14)
axes[0].set_ylabel('Peta1 width', fontsize=14)
axes[1].set_xlabel('Peta1 length', fontsize=14)
axes[1].set_ylabel('Peta1 width', fontsize=14)
axes[0].tick_params(direction='in', length=12, width=7, colors='k', labelsize=15)
axes[1].tick_params(direction='in', length=12, width=7, colors='k', labelsize=15)
axes[0].set_title('Actual', fontsize=14)
axes[1].set_title('Predicted', fontsize=14)






