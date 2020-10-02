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


# Importing the dataset
import pandas as pd
dataset = pd.read_csv("../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv")


# # 1. Data Analysis

# In[ ]:


# Displaying the head of out dataset
dataset.head()


# In[ ]:


# Displaying the datatype and shape of the dataset

print(dataset.shape)
dataset.dtypes


# In[ ]:


# Displaying the describe statistical info of each attribute
dataset.describe()


# # a. Histogram Distribution Visualisation

# In[ ]:


# Histogram visualisation for age column and know what kind of distribution  it is ?

import seaborn as sb
sb.distplot(dataset['Age'])


# Age Attribute having similar kind of normal distribution with wider Standard Deviation.

# In[ ]:


# Histogram visualisation for Annual income column and know what kind of distribution  it is ?

sb.distplot(dataset['Annual Income (k$)'])


# Above Histogram plot Annual income column data looks like normal distribution with wider Standard deviation

# In[ ]:


# Histogram visualiation for Spending Score column and to know what kind of distribution it is?

sb.distplot(dataset['Spending Score (1-100)'])


# # b. HeatMap Correlation Visualisation

# In[ ]:


# Visualisation correlation coefficient of each attribute.

corr_value=dataset.corr()
sb.heatmap(corr_value,square=True)


# We can't display the correlation coefficient values in heatmap, Because each attribute finds the coefficient value with output attribute.
# 
# Here we dont have output attribute because it is an un-supervised learning. Here we are segmenting the culster of categories based on annual income and spending score.

# # Feature Engineering
# 
# # a. Data Cleaning

# In[ ]:


# Displying any empty or null values in our dataset
dataset.info()


# In[ ]:


# Displying the empty or null value in our dataset to understand better how many missing cells there in each attribute

dataset.isna().sum()


# Perfect we don't have any missing values in our dataset so no need to remove any columns and rows..
# 
# CustomerID is not required to make segementation cluster

# In[ ]:


# Dropping CustomerID column 

dataset=dataset.drop(['CustomerID'],axis=1)
dataset.head()


# # b. Label Encoder

# In[ ]:


# Encoding the Gender column from categorical value into numerical value

dataset['Gender'].unique()


# In[ ]:


dataset['Gender']=dataset['Gender'].map({'Male':0,'Female':1})
dataset['Gender'].unique()


# In[ ]:


dataset.head()


# # c. Outliers
# 
# In unsupervised algorithm we wont have ouput attribute so we cant predict the outliters here.

# # d. OneHotEncoder
# 
# As we done encoding the label of gender column, We dont need to apply onehotencoder because labeled values in between 0 and 1 only. So no need to apply one Hot Encoder.

# # e. Feature Split
# 
# Split the dataframe feature into input attribute of array of matrix

# In[ ]:


# Feature Split
x=dataset.values

print(x[:5,:])


# # f. Feature Scale
# 
# Applying the rescale technique to keep all input attribute value in the range of 0 to 1 by using MinMaxScaler

# In[ ]:


# Feature Scale

from sklearn.preprocessing import MinMaxScaler
minmaxscaler=MinMaxScaler()
x=minmaxscaler.fit_transform(x)
print(x[:5,:])


# # 2. Modeling

# To find optimal number of segmentation (Clusters) we are going to use Elbow Method.
# 
# Elbow Method is used get optimal no.of cluster value with elbow visualisation graph.

# # a. K-Means
# 
# # K-Means Elbow Method

# In[ ]:


# Elbow Method

seed=5

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

wcss=[]
# n_init ----- Number of kmeans will run with different init centroids
# max_iter------ Max Number of iterations to define that the final clusters
# init='k-means++' ---- random initlization to handle random intialization trap
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=500,n_init=20,random_state=seed)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title("Elbow method")
plt.xlabel("No.of Clusters")
plt.ylabel('WCSS')
plt.show()


# As per the above optimal Elbow method graph 4 cluster segemnetation will be great...

# In[ ]:


# K-Means Cluster Algorithm
kmeans=KMeans(n_clusters=4,init='k-means++',random_state=seed,max_iter=500,n_init=20)
y_kmeans=kmeans.fit_predict(x)

# Predicting the Customers with different segments
print(y_kmeans)


# # Visualising Result And Its Clusters.

# In[ ]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='red',label='Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='blue',label='Cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='green',label='Cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,color='cyan',label='cluster 4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,color='yellow',label='Centroid')
plt.title("Cluster Clients")
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


# # b. Hierarchical Cluster
# 
# # Hierarchical Dendo Gram

# In[ ]:


# Dendo Gram plot is used to find optimal number of cluster..

import scipy.cluster.hierarchy as sch
dendogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('customers')
plt.ylabel('Eulidean distance')
plt.show()


# As per Eulidean distance 3 giving the 4 optimal no.of clusters and because those 4 lines not interceting any lines.

# In[ ]:


# Hierarchical Clustering Algorithm to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=4)
hc.fit(x)


# In[ ]:



# Predict the cluster categories based on mall dataset
y_hc=hc.fit_predict(x)
print(y_hc)


# In[ ]:


plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,color='red',label='Cluster 1')
plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,color='blue',label='Cluster 2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,color='green',label='Cluster 3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,color='cyan',label='cluster 4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=200,color='yellow',label='Centroid')
plt.title("Cluster Clients")
plt.xlabel('Annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()


# Both K-Means and Hierarchical Cluster will be great algorithms for unsupervised cluster kind of problems but K-Means will give great performance......
# 
# If any questions please let me know...
