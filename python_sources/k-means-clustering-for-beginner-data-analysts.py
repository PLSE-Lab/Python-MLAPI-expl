#!/usr/bin/env python
# coding: utf-8

# # **K-means clustering using a randomly generated dataset**
# 
# *K-means clustering is an unsupervised learning algorithm where we do not need to pass output labels to the algorithm for it to be trained. The training exercise is done without supervision from the users and the end result is a set of 'K' non-overlapping partitions.*

# * **We will create a dataset using the make_blobs function which we will use in our clustering exercise. We will also be using matplotlib to visualize our clusters.**

# In[ ]:


# importing the packages
import random
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# * **Setting up a seed for repeatability of results. '42' because it is the answer to everything(google if you don't understand :P)**

# In[ ]:


np.random.seed(42)


# * **We shall now create a dataset using the make_blobs function. Kindly note the following:** 
# 
# 
# 1. The centers are the ones which we have randomly selected since we are creating the dataset ourselves. We can also select 'K' points on our own which we can choose to call as cluster centroids. This is best done when we have an external dataset with us.
# 2. The cluster_std is the cluster standard deviation
# 3. n_samples is the number of rows(training samples) we want to create

# In[ ]:


X, y = make_blobs(n_samples=5000, centers= [[4,4], [-2, -1], [2, -3], [1, 1]], cluster_std= 0.9)


# * **Let us visualize how the data looks**

# In[ ]:


plt.figure(figsize = (10,7))
plt.title("The different clusters generated from make_blobs")
plt.scatter(x = X[:, 0], y = X[:,1], c=y[:])
plt.show()


# * **We will now be using the K-means clustering algorithm for K=4**
# 
# Note that the value of K is a hard problem in K-means and often multiple different values have to be selected to determine the best K. But here, we have created our own dataset using 4 centroids hence we can easily assign k=4

# In[ ]:


k_means = KMeans(n_clusters=4, init = 'k-means++', n_init= 12)
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_labels


# * *K means gave us the following cluster centers*

# In[ ]:


k_means_cluster_centers = k_means.cluster_centers_
k_means_cluster_centers


# * Visualizing the results using the below code for a beautiful representation of K means

# In[ ]:


# Initialize the plot with the specified dimensions.
fig = plt.figure(figsize=(10, 7))

# Colors uses a color map, which will produce an array of colors based on
# the number of labels there are. We use set(k_means_labels) to get the
# unique labels.
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))

# Create a plot
ax = fig.add_subplot(1, 1, 1)

# For loop that plots the data points and centroids.
# k will range from 0-3, which will match the possible clusters that each
# data point is in.
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):

    # Create a list of all data points, where the data poitns that are 
    # in the cluster (ex. cluster 0) are labeled as true, else they are
    # labeled as false.
    my_members = (k_means_labels == k)
    
    # Define the centroid, or cluster center.
    cluster_center = k_means_cluster_centers[k]
    
    # Plots the datapoints with color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    
    # Plots the centroids with specified color, but with a darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,  markeredgecolor='k', markersize=6)

# Title of the plot
ax.set_title('KMeans')

# Remove x-axis ticks
ax.set_xticks(())

# Remove y-axis ticks
ax.set_yticks(())

# Show the plot
plt.show()


# *Hope that you liked it. K-means is just one of the many types of clustering algorithms but the one you wouldn't want to miss learning. Hope that you liked the notebook and its explanation. Kindly upvote and comment what you learned*
