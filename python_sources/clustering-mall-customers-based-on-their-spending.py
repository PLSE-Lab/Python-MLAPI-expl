#!/usr/bin/env python
# coding: utf-8

# **This is my first Kernal, If there are any mistakes or any improvements that I have to do Please let me know in the commments.**
# 
#     The Dataset contains details of customers like age, sex, annual income, spending score,.. Spending Score is determined by the mall and the value of the spending score is between (1-100). The spending score near to 100 indicates that the customer has spent more money in the mall and when spending score is near 1 that indicates customer has spent less in the past in the mall. But the dataset does not tell in which category the customers belongs which indicates clustering should be performed.
# 
#     When the mall advertisement team has to send advertisement to the customers on the new products and if the advertisement is send to 10000 customers there are chances that only 10% of the customers will buy the product which will not be efficient. Inorder to improve the efficiency the Customer data set should be analized to cluster the customers based on the annual income and spending score so that the the customers who spend more can be targetted first.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt #plot the graph 

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


#importing the dataset
dataset = pd.read_csv('../input/Mall_Customers.csv')
X = dataset.iloc[:, [3,4]].values


# * **From the dataset only annual income and the spending score of the customers is needed so index 3 and 4 alone are selected from the dataset.**

# In[ ]:


#using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendogram = sch.dendrogram(sch.linkage(X , method = 'ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')
plt.show()


# <img src="https://i.imgur.com/U8YZP0B.jpg" width="700px">
# 
# ***The longest Line which is not intersected by any other line is taken to set the threshold for the number of clusters.***
# 
# 
# <img src="https://i.imgur.com/p7GTaA6.jpg" width="700px">
# 
# ***From the threshold line we can determine that we get 5 clusters as the threshold line intersects with 5 lines.***
# 

# ![](http://<blockquote class="imgur-embed-pub" lang="en" data-id="a/G6cEhpa"><a href="//imgur.com/a/G6cEhpa"></a></blockquote><script async src="//s.imgur.com/min/embed.js" charset="utf-8"></script>)
# 

# In[ ]:


#fit the Hierarchial clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters= 5, affinity = 'euclidean', linkage= 'ward') #n_cluster is set 5 as it is optimal as shown in dendogram
y_hc= hc.fit_predict(X)


# * Importing AgglomerativeClustering class from scikit-learn library.
# * The number of clusters is set to 5 as we know 5 clusters is optimal from the dendogram.
# * Affinity is chosen euclidean as we choose euclidean distance between the clusters.
# 

# In[ ]:


#visualizing the cluster
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# **Here as it is seen in the graph, **
# *      Customers in cluster-1 are considered as careful customers as they spend less even though their annual Income is More.
# *      Customers in cluster-2 are considered as Average Customers as thier Annual Income is average and they spend average money in the mall.
# *      Customers in cluster-3 are considered to be the target Customers as they have more annual income and they spend more in the mall.
# *      Customers in cluster-4 are considered as careless customers because they have low annual income but still they spend more in the mall.
# *      Customers in cluster-5 are considered as lest spending customers as they have low annual income and they spend very less in the mall.
# 
# **Now when the mall has to send advertisement to the customers to sell their products in their mall, instead of sending advertisement to random customers, the customers in the cluster-3 and customers in the cluster-4 can be targetted first as they spend more in the mall and thete will be higher chance of buying their product seeing the advertisement. 
# This will increase the chances of selling the product by 70-80%.**
# 
