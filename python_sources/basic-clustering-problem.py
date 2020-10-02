#!/usr/bin/env python
# coding: utf-8

# 
# ### This notebook was completed in accordance to [THIS TUTORIAL](https://towardsdatascience.com/k-means-clustering-algorithm-applications-evaluation-methods-and-drawbacks-aa03e644b48a) on Towards Data Science.
# ___
# 
# The main objective is to see if there is a relationship between the time you'd wait for a geyser eruption and the how long the duration of the eruption is for Ol' Faithful geyser at Yellowstone National Park.
# 
# TL;DR
# 
# There seems to be a corellation between ['waitng', 'eruption] points of data. The data suggests there is a cyclic activity that happens below Yellowstone that allows for a short waiting time and a short eruption followed by a longer waiting time with a longer eruption that could especially be seen on the dist and scatter plots. If the dataset is equipped with the date, time and also the height of the eruption, a more precise conclusion could be made to either support or rebut the hypothesis. 

# ### Process
# ---
# 
# 1. Evaluate and explore the data to figure out additional information that could be useful for analysis
# 2. Prepare the data through scaling and selecting only relevant features in the dataset.
# 3. Process data through clustering methods with scaled data.
# 4. Visualise the data that is processed to gain insight.
# 5. Make what you will of the data :) (Ethically, you should tell the story of the data, not make the data fit your story)
# 

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.datasets.samples_generator import(make_blobs, make_circles, make_moons)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances_argmin


df = pd.read_csv("../input/old-faithful/faithful.csv")
df.head()


# In[ ]:


fig = plt.figure(figsize = (12,6))
plt.subplot(1,2,1)
sns.distplot(df.waiting, bins = 10)

plt.subplot(1,2,2)
sns.distplot(df.eruptions, bins = 10)

plt.show()


# Graphs above shows that there are two different sets of data that could be clustered together. Since this is two dimentional data, it is possible to plot these points on a graph using a scatter plot. 

# In[ ]:


plt.scatter(df.eruptions, df.waiting)
plt.xlabel('Eruption time (min)')
plt.ylabel('Waiting interval (min)')
plt.title('Ol\' Faithful Geyser Eruption')


# This reveals that there are two clustering points, ie., the data can be categorised into two distinctive segments. But, that is what's seen to us as humans, sometimes there are hidden patterns that we are not able to see. 
# 
# To prove this, we will deploy Machine Learning methods, specifically, K-Means clustering, the most common form of clustering.

# In[ ]:


elbow = []
x = StandardScaler().fit_transform(df)
for i in range(1,10):
    km = KMeans(n_clusters = i, max_iter = 20, random_state = 20)
    km.fit(x)
    elbow.append(km.inertia_)

#Plot cluster
plt.plot(range(1,10), elbow)
plt.xlabel('Num of cluster')
plt.title('Elbow Method')
plt.ylabel('WCSS')
plt.show()


# ### [Elbow Method]((https://uc-r.github.io/kmeans_clustering#elbow)
# ---
# 
# Determines the optimal number of clusters in the data set which can be seen at the sharpest curve of the graph, hence the name. 
# 
# K-Means in general utilises the Within Cluster Sum of Squares(WCSS) algorithm = distance of data point with centroid, the closer it is, the more similar they are.
# 
# In this data, it is clear that 2 is the best K as it has the sharpest curve. This will be evaluated furthur below. For now with K = 2, the below scatterplot is acheived.

# In[ ]:


df = df[['eruptions', 'waiting']]
x = StandardScaler().fit_transform(df)

#kmeans
km = KMeans(n_clusters = 2, max_iter = 20, random_state = 20)
km.fit(x)
#Plot cluster
kmCenter = km.cluster_centers_
fig, ax = plt.subplots(figsize = (6,6))
plt.scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')
plt.scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')
plt.scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')

plt.legend()
plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
plt.title('Visualization of clustered data')
ax.set_aspect('equal')


# In[ ]:


# very long way to visualise k-means scatterplot
fig, ax = plt.subplots(2, 2, figsize = (12,12), sharex = True, sharey = True)

#kmeans
km = KMeans(n_clusters = 1, max_iter = 20, random_state = 20)
km.fit(x)
kmCenter = km.cluster_centers_
#Plot cluster
ax[0,0].set_title('K = 1')
ax[0,0].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')
ax[0,0].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')

km = KMeans(n_clusters = 2, max_iter = 20, random_state = 20)
km.fit(x)
kmCenter = km.cluster_centers_
#Plot cluster
ax[0,1].set_title('K = 2')
ax[0,1].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')
ax[0,1].scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')
ax[0,1].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')

km = KMeans(n_clusters = 3, max_iter = 20, random_state = 20)
km.fit(x)
kmCenter = km.cluster_centers_
#Plot cluster
ax[1,0].set_title('K = 3')
ax[1,0].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')
ax[1,0].scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')
ax[1,0].scatter(x[km.labels_ == 2,0], x[km.labels_ == 2,1], c = 'yellow', label = 'Cluster 3')
ax[1,0].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')

km = KMeans(n_clusters = 4, max_iter = 20, random_state = 20)
km.fit(x)
kmCenter = km.cluster_centers_
#Plot cluster
ax[1,1].set_title('K = 4')
ax[1,1].scatter(x[km.labels_ == 0,0], x[km.labels_ == 0,1], c = 'green', label = 'Cluster 1')
ax[1,1].scatter(x[km.labels_ == 1,0], x[km.labels_ == 1,1], c = 'blue', label = 'Cluster 2')
ax[1,1].scatter(x[km.labels_ == 2,0], x[km.labels_ == 2,1], c = 'yellow', label = 'Cluster 3')
ax[1,1].scatter(x[km.labels_ == 3,0], x[km.labels_ == 3,1], c = 'purple', label = 'Cluster 4')
ax[1,1].scatter(kmCenter[:, 0], kmCenter[:,1], c = 'r', label = 'Centroid', marker = '*')


plt.xlim([-2, 2])
plt.ylim([-2, 2])
plt.xlabel('Eruption time in mins')
plt.ylabel('Waiting time to next eruption')
fig.suptitle('Visualization of clustered data')


# ### [Silhouette Analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html)
# ---
# Used for K-Means clustering evaluation, checks the distance between centroid and data points. Values in this graph ranges from [-1, 1]:
# 
# Silhouette Coefficients(silCo) = values calculated
# 
# * silCo nearing +1 = samples are far away from their neighbor clusters
# * silCo 0 = very close to decision boundary between neighboring clusters
# * silCo nearing -1, negative values = smaple maybe assigned to wrong clusters
# 
# 
# It should also be noted that using this method to code and iterate through the data is a more precise and elegant way to create scatterplots compated to the one above. 

# In[ ]:


# Conciese and eficient method of iterating through differenct k-means along with it's silhouette analysis
c = [2,3,4]
for n in c:
    fig, (ax, ay) = plt.subplots(1,2,figsize = (12,8))
    ax.set_xlim([-0.1, 1])
    cluster = KMeans(n_clusters = n, max_iter = 20, random_state = 20)
    clusterLabel = cluster.fit_predict(x)
    silAvg = silhouette_score(x, clusterLabel)
    print("K = ", n," average silhouette scoe : ", silAvg)
    sampleSilVal = silhouette_samples(x, clusterLabel)
    
    yLower = 10
    for i in range(n): 
        clusterSilVal = sampleSilVal[clusterLabel == i]
        clusterSilVal.sort()
        
        iClusterSize = clusterSilVal.shape[0]
        yUpper = yLower + iClusterSize
        
        color = cm.nipy_spectral(float(i) / n)
        ax.fill_betweenx(np.arange(yLower, yUpper), 0, clusterSilVal, facecolor = color, edgecolor = color, alpha = 0.7)
        
        ax.text(-0.05, yLower + 0.5 * iClusterSize, str(i))
        yLower = yUpper + 10
    ax.axvline(x = silAvg, color = 'red', linestyle = '--')
    
    colors = cm.nipy_spectral(clusterLabel.astype(float) / i)
    ay.scatter(x[:,0] ,x[:,1] ,c = colors, edgecolor='k')

    
    centers = cluster.cluster_centers_
    # Draw white circles at cluster centers
    ay.scatter(centers[:, 0], centers[:, 1], marker='o',c="white", alpha=1, s=400, edgecolor='k')

    for i, c in enumerate(centers):
        ay.scatter(c[0], c[1], marker='$%d$' % i, cmap = 'winter')

    ay.set_title("Ol' Faithful")
    ay.set_xlabel("Eruption Time")
    ay.set_ylabel("Waiting Time")

    
plt.show()


# In the sample above k = 3 and 4 are bad picks due to some samples having negative values, this means that there are data values that are clustered wrongly. Therefore, according to the elbow method, the evaluation using silhouette analysis proves correct as the best cluster for this dataset is k = 2. What can we infer from this?

# ### Refferences
# 1. Everything in General
#     https://uc-r.github.io/kmeans_clustering#elbow
# 2. Subplots
#     https://jakevdp.github.io/PythonDataScienceHandbook/04.08-multiple-subplots.html
#     https://stackoverflow.com/questions/31726643/how-do-i-get-multiple-subplots-in-matplotlib
# 3. Silhouette Analysis
#     https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
