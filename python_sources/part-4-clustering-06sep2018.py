#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# the focus of this notebook is to discuss Clustering in detail - applications, usage, etc


# In[ ]:


import numpy as np


# In[ ]:


# Clustering in simple terms is the n Dimensional equivalent of a scatter plot


# In[ ]:


# look at the IRIS dataset and plot the various classes using a scatter plot


# In[ ]:


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets


# In[ ]:


iris = datasets.load_iris()
X = iris.data
y = iris.target


# In[ ]:


X.shape


# In[ ]:


# the 4 cols are Petal Length , Petal Width , Sepal Length , Sepal width


# In[ ]:


y.shape


# In[ ]:


#let's plot a scatter plot of 2 dimensions - say petal length and petal width
X_2cols = X[:, :2]


# In[ ]:


X_2cols.shape


# In[ ]:


X_2cols[: , 1].shape


# In[ ]:


plt.scatter(x=X_2cols[:, 0], y=X_2cols[:, 1], edgecolor = 'k', c = y)
plt.show()


# In[ ]:


#we can clearly see some groups in the above 2D plot; the top left has a lot of blues 
#while the other part is a mix of the other two classes

#let's try a 3D plot with one more dimension


# In[ ]:


fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], edgecolor = 'k', c = y)

ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_zlabel('Sepal Length')

plt.show()


# In[ ]:


# now we can clearly see the three distinct classes separated nicely


# In[ ]:


#lets try out a clustering algorithm on this data


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


iris_kmeans = KMeans(n_clusters=3).fit(X)


# In[ ]:


X.shape


# In[ ]:


y.shape


# In[ ]:


iris_kmeans


# In[ ]:


#the output of the clustering exercise is the labels tagged to each record; this is available in the labels parameter


# In[ ]:


iris_kmeans.labels_


# In[ ]:


fig = plt.figure(figsize=(10,8))

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], edgecolor = 'k', c = iris_kmeans.labels_)

ax.set_xlabel('Petal Length')
ax.set_ylabel('Petal Width')
ax.set_zlabel('Sepal Length')

plt.show()


# In[ ]:


y #the actual (true) lable


# In[ ]:


#confusion matrix to check how close to the truth was our clustering algo output


# In[ ]:


from sklearn.metrics import confusion_matrix


# In[ ]:


confusion_matrix(y_pred=iris_kmeans.labels_, y_true=y)


# In[ ]:


#we see that the first class is identified without any error while there are errors in the other two classes


# In[ ]:


iris_kmeans.inertia_


# # what has been achieved so far?
# If we had no knowledge of the actual classes of this dataset, clustering would have provided with a grouping of the data points. And, this is an extension of the scatter plot we saw above

# # Elbow Plot:
# How do we know what is the current number of clusters for this dataset? Remember that clustering is 'unsupervised learning'.
# Answer is the Elbow plot

# In[ ]:


#initialize a list to hold the within cluster sum of squares
wcss = []

#run kmeans for different n cluster values and save the wcss in this list
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_) # inertia is Sum of squared distances of samples to their closest cluster center
    
#Plotting the results onto a line graph, allowing us to observe 'The elbow'
plt.plot(range(1, 15), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# The sharpest fall is at 2 and 3; we can choose 3 as the n clusters

# # How does K Means Clustering Work?

# 1. Choose k points at random in the data set
# 2. These points would be called centroids and they represent the k clusters
# 3. Take a point (other than the centroids), find the centroid that is closest to the point and assign the point to that centroid (cluster)
# 4. Repeat this for all points
# 5. Once all the initial assignments are done, calculate the new centroid for each cluster (centroid - mid point)
# 6. Repear steps 3 - 5 till there is no change in assignment happens

# # Hierarchical Clustering

# In[ ]:


from scipy.cluster.hierarchy import dendrogram, linkage


# In[ ]:


iris_hierarchical = linkage(X, method='ward')


# In[ ]:


type(iris_hierarchical)


# In[ ]:


iris_hierarchical[0]


# In[ ]:


iris_hierarchical[1]


# In[ ]:


iris_hierarchical[:8]


# In[ ]:


iris_hierarchical.shape


# In[ ]:


iris_hierarchical[148]


# #the way to read the elements in the above array: <br>
# 1st element, 2nd element - index of the sample that was merged in this iteration<br>
# 3rd element - distance between the above 2 samples<br>
# 4th element - #samples in the cluster at this level. Notice the value above is 150 corresponding to the top node<br>

# In[ ]:


#https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/


# In[ ]:


# calculate full dendrogram
plt.figure(figsize=(8, 6))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance')
dendrogram(
    iris_hierarchical,
    leaf_rotation=90.,  # rotates the x axis labels
    leaf_font_size=8.,  # font size for the x axis labels
)
plt.show()


# The algorithm starts assuming each sample is a cluster in itself <br>
# try to find the nearest cluster to your cluster - different methods are there; average linkage, min linkage, etc.<br>
# add the nearest cluster to your cluster<br>
# keep going till you have only one cluster (top node)<br>
# <br>
# http://www.analytictech.com/networks/hiclus.htm

# # How does K Means work

# # How is Hierarchical different from K Means

# https://www.quora.com/What-are-the-pros-and-cons-of-kmeans-vs-hierarchical-clustering/answer/Shehroz-Khan-2 <br>
# https://www.cs.utah.edu/~piyush/teaching/4-10-print.pdf <br>
# https://www.quora.com/What-are-the-advantages-of-K-Means-clustering

# # 18 Apr 2018

# https://github.com/dgrtwo/dgrtwo.github.com/blob/master/_R/2015-01-16-kmeans-free-lunch.Rmd <br>
# http://varianceexplained.org/r/kmeans-free-lunch/

# In[ ]:




