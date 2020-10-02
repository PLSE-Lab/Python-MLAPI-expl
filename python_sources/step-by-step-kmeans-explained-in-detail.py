#!/usr/bin/env python
# coding: utf-8

# ![](http://cdn-images-1.medium.com/max/2400/1*PRC6tdXpTekQ6X7qdUDehg.jpeg)

# ## Super Data Science and Udemy! Big Thanks! 
# 
# ![](https://preview.redd.it/o4mshdf4hui01.jpg?width=750&auto=webp&s=2d647d5d30a1f1b7507411929ff077e1df967e00)

# ### *Table of content*
# 
# [1. What does KMeans do?](#1)
# 
# 
# [2. Applications](#2)
# 
# 
# [3. Working](#3)
# 
# 
# [4. Choosing the right K](#4)
# 
# 
# 
# [5. Centroid Random Initialisation Trap](#5)
# 
# 
# 
# [6. Implemention](#6)
# 
# 
# 
# [7. Visualisation](#7)

# <a id="1"></a>
# ## 1. What does KMeans do?
# 
# K-means clustering is a type of unsupervised learning, which is used when you have unlabeled data (i.e., data without defined categories or groups). The goal of this algorithm is to find groups in the data, with the number of groups represented by the variable K. The algorithm works iteratively to assign each data point to one of K groups based on the features that are provided. Data points are clustered based on feature similarity. The results of the K-means clustering algorithm are:
# 
# * The centroids of the K clusters, which can be used to label new data
# * Labels for the training data (each data point is assigned to a single cluster)

# ![](https://imgur.com/a/wVDJPuZ)

# ![](https://i.imgur.com/rwkQNbv.png)

# <a id="2"></a>
# ## 2. Applications
# The K-means clustering algorithm is used to find groups which have not been explicitly labeled in the data. This can be used to confirm business assumptions about what types of groups exist or to identify unknown groups in complex data sets. Once the algorithm has been run and the groups are defined, any new data can be easily assigned to the correct group.
# 
# This is a versatile algorithm that can be used for any type of grouping. Some examples of use cases are:
# 
# * Behavioral segmentation:
# * * Segment by purchase history
# * * Segment by activities on application, website, or platform
# * * Define personas based on interests
# * * Create profiles based on activity monitoring
# * Inventory categorization:
# * * Group inventory by sales activity
# * * Group inventory by manufacturing metrics
# * Sorting sensor measurements:
# * * Detect activity types in motion sensors
# * * Group images
# * * Separate audio
# * * Identify groups in health monitoring
# * Detecting bots or anomalies:
# * * Separate valid activity groups from bots

# <a id="3"></a>
# ## 3. Working
# 
# Let's now discuss the working of KMeans algorithm. The aim is to break the explanation down in the simplest way possible. 
# 
# 
# #### It begins with choosing the number of K clusters. The K signifies the number of clusters that the algorithm would find in the dataset. Now choosing the right K is very important. Sometimes the K is clearly visible from the dataset when visualized. However most of the times this is not the case and in a short time we'll see about how to choose the right K value.
# 
# 
# 
# ![](https://i.imgur.com/RBK4dtA.png)
# 

# #### The second step is to allocate K random points as centroids. These K points could be points from the dataset or outside. There's one thing to note however. The random initialisation of centroids can sometimes cause random initialisation trap which we would see in this section soon.
# 
# ![](https://i.imgur.com/LfI2qfl.png)

# #### In the third step the dataset points would be allocated to the centroid which is closest to them.
# 
# 
# 
# ![](https://i.imgur.com/9I5JH3m.png)
# 

# #### The fourth step is to calculate the centroid of the individual clusters and place the old centroid there.
# 
# 
# 
# 
# ![](https://i.imgur.com/FyIeKuA.png)

# #### The fifth step is to reassign points like we did in step 3. If reassignment takes place then we need to go back to step four. If no reassignment takes place then we can say that our model has converged and its ready.
# 
# 
# 
# 
# ![](https://i.imgur.com/aRaGcKB.png)

# ## Step Summary
# ### To summarise the steps we can say :
# ![](https://i.imgur.com/3jTk7Y0.png)

# <a id="4"></a>
# ## 4. Choosing the right K
# 
# The way to evaluate the choice of K is made using a parameter known as WCSS. WCSS stands for **Within Cluster Sum of Squares**.
# It should be low. Here's the formula representation for example when K = 3
# 
# Summation Distance(p,c) is the sum of distance of points in a cluster from the centroid.
# 
# 
# ![](https://i.imgur.com/5W63xul.png)

# The Elbow Method is then used to choose the best K value. In the depiction below we can see that after 3 there's no significant decrease in WCSS so 3 is the best here. Therefore there's an elbow shape that forms and it is usually a good idea to pick the number where this elbow is formed. There would be many times when the graph wouldn't be this intuitive but with practice it becomes easier.
# 
# ![](https://i.imgur.com/gi9p7V5.png)

# <a id="5"></a>
# ## 5. Centroid Random Initialisation Trap
# 
# Through these images let's see how two different random initialisations can cause a totally different outcome.
# 
# ### Init 1
# 
# 
# 
# ![](https://i.imgur.com/zsC9z0z.png)
# 
# 
# 
# 
# ### Init 2
# 
# 
# 
# ![](https://i.imgur.com/kU5BX6j.png)

# So we saw that even with clear distinction possible visually, wrong randomisation can produce wrong results.
# There have been researches carried out and one of the most famous ways to initialise centroids is KMeans++.
# The best thing is that the whole algorithm remains the same but the only difference is that we provide an argument to SKlearn to use KMeans++ for initialisation. There are many papers explaining the KMeans++ but the explanation is beyond this notebook for now. :)

# ![](https://cdn-images-1.medium.com/max/1200/1*x7P7gqjo8k2_bj2rTQWAfg.jpeg)

# <a id="6"></a>
# ## 6. Implementation

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(context="notebook", palette="Spectral", style = 'darkgrid' ,font_scale = 1.5, color_codes=True)
print(os.listdir("../input"))


# In[2]:


# Importing the dataset
dataset = pd.read_csv('../input/Mall_Customers.csv',index_col='CustomerID')


# In[3]:


dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset.describe()


# In[6]:


dataset.isnull().sum()


# No Nans found! Great

# In[7]:


dataset.drop_duplicates(inplace=True)


# In[8]:


# using only Spending_Score and income variable for easy visualisation
X = dataset.iloc[:, [2, 3]].values


# In[9]:


# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    # inertia method returns wcss for that model
    wcss.append(kmeans.inertia_)


# In[10]:


plt.figure(figsize=(10,5))
sns.lineplot(range(1, 11), wcss,marker='o',color='red')
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[47]:


# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# <a id="7"></a>
# ## 7. Visualisation

# In[46]:


# Visualising the clusters
plt.figure(figsize=(15,7))
sns.scatterplot(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], color = 'yellow', label = 'Cluster 1',s=50)
sns.scatterplot(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], color = 'blue', label = 'Cluster 2',s=50)
sns.scatterplot(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], color = 'green', label = 'Cluster 3',s=50)
sns.scatterplot(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], color = 'grey', label = 'Cluster 4',s=50)
sns.scatterplot(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], color = 'orange', label = 'Cluster 5',s=50)
sns.scatterplot(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color = 'red', 
                label = 'Centroids',s=300,marker=',')
plt.grid(False)
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# Big Thanks to:
# 
# * https://www.datascience.com/blog/k-means-clustering
# * https://www.superdatascience.com
# * https://www.udemy.com
# 

# In[ ]:




