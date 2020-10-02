#!/usr/bin/env python
# coding: utf-8

# * **Let's Import the required libraries**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # Graphs & Visualization 
import seaborn as sns

import os
print(os.listdir("../input"))


# * **Now let's import the our Mall dataset**

# In[ ]:


dataset = pd.read_csv('../input/Mall_Customers.csv')


# In[ ]:


#Let's check the data
dataset.head()


# In[ ]:


#Let's check the shape of data
dataset.shape


# In[ ]:


#Let's check datatypes
dataset.dtypes


# * **Now we have to check a NULL values in dataset**

# In[ ]:


dataset.isnull().sum()


# * **Now let's visualize the data**

# * **Let's plot the Histogram**

# In[ ]:


plt.figure(1 , figsize = (17 , 8))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    sns.distplot(dataset[x] , bins = 20)
    plt.title('Distplot of {}'.format(x))
plt.show()


# * **Now let's plot the count of gender with Countplot**

# In[ ]:


plt.figure(1 , figsize = (17 , 8))
sns.countplot(y = 'Gender' , data = dataset)
plt.show()


# * **Now let's select the features**

# In[ ]:


### Feature sleection for the model
#Considering only 2 features (Annual income and Spending Score) and no Label available
x = dataset.iloc[:, [3,4]].values


# In[ ]:


print(x)


# * **Now we have to find numbers of cluster which we can plot so we can use Elbow method on Kmeans++ Calculations**

# In[ ]:


from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# * **We can clearly see that(Zoom on ELBOW) ELBOW comes at k = 5 so we will choose a k = 5 so let's create the Clusters**

# In[ ]:


#KMeans is our Algorithms which provided in SKlearn
#n_clusters is a nummber of clusters which we will define 
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
#Let's predict the x
y_kmeans = kmeans.fit_predict(x)


# In[ ]:


print(y_kmeans)
#We convert our prediction to dataframe so we can easily see this prediction in table form
df_pred = pd.DataFrame(y_kmeans)
df_pred.head()


#     Let's see a df_pred, it's our prediction means
# - 0 number customer belongs to 2 number cluster
# - 1 number customer belongs to 3 number cluster

# * **Let's Visualize the all 5 Clusters**

# In[ ]:


plt.figure(1 , figsize = (17 , 8))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'aqua', label = 'Cluster 3')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'violet', label = 'Cluster 4')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'lightgreen', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'navy', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# * **Let's Study the out Clusters**

# In[ ]:


#Cluster 1 (Red Color) -> Earning medium but spending medium
#cluster 2 (Yellow Colr) -> Earning High but spending very less 
#cluster 3 (Aqua Color) -> Earning is low & spending is low
#cluster 4 (Violet Color) -> Earning is less but spending more -> Mall can target this type of people
#Cluster 5 (Lightgereen Color) -> Earning High & spending more -> Mall can target this type of people
#Navy color small circles is our Centroids


# * **Now we make a cluster once again with perfect labels**

# In[ ]:


plt.figure(1 , figsize = (17 , 8))
plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Standard people')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s = 100, c = 'yellow', label = 'Tightwad people')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s = 100, c = 'aqua', label = 'Normal people')
plt.scatter(x[y_kmeans == 3, 0], x[y_kmeans == 3, 1], s = 100, c = 'violet', label = 'Careless people(TARGET)')
plt.scatter(x[y_kmeans == 4, 0], x[y_kmeans == 4, 1], s = 100, c = 'lightgreen', label = 'Rich people(TARGET)')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'navy', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()


# * **So finally we make the cutomer segmantaion of Mall dataset **

# **Refrence : - Machine learning A-Z course on Udemy**
