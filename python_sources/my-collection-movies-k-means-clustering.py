#!/usr/bin/env python
# coding: utf-8

# For this tutorial, we will use the **My Movie Collection** data set available on the kaggle datasets :
# 
# [**My Movie Collection**](https://www.kaggle.com/fazilbtopal/my-movie-collection)

# In[ ]:


# libraries
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# # Importing Dataset

# In[ ]:





# In[ ]:


df = pd.read_excel('/kaggle/input/movies_collection.xlsx')


# In[ ]:


df.head()


# In[ ]:


# transform categorical data
cols_to_transform = ['Seen']
 
df = pd.get_dummies( columns = cols_to_transform , data = df)
# Seen --- 2 value 0 or 1 
# we gonna remove one column
df = df.drop(columns=['Seen_False'])


# Only keep the numeric columns for our analysis. However, we'll keep Original Name also to interpret the results at the end of clustering. Note that this Original Name column will not be used in the analysis.

# In[ ]:


df_numeric = df[['Orginal Name','Seen_True',
                 'Budget','Year','Duration',
                 'Votes','Rating' ,
                 'Personal Rating']]


# In[ ]:


df_numeric.head()


# Check if rows contain any null values

# In[ ]:


df_numeric.isnull().sum()


# Drop all the rows with null values

# In[ ]:


df_numeric.dropna(inplace=True)


# Let's see the statistics for the movies data

# In[ ]:


df_numeric['Budget'].describe()


# In[ ]:


df_numeric['Duration'].describe()


# In[ ]:


df_numeric['Votes'].describe()


# In[ ]:


df_numeric['Rating'].describe()


# In[ ]:


df_numeric['Personal Rating'].describe()


# # Normalize data

# Normalize the data with MinMax scaling provided by sklearn

# In[ ]:


from sklearn import preprocessing
minmax_processed = preprocessing.MinMaxScaler().fit_transform(df_numeric.drop('Orginal Name',axis=1))
df_numeric_scaled = pd.DataFrame(minmax_processed, index=df_numeric.index, columns=df_numeric.columns[:-1])
df_numeric_scaled.head()


# # Apply K-Means Clustering

# **What k to choose?**                                                                    
# Let's fit cluster size 1 to 20 on our data and take a look at the corresponding score value.

# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]


# In[ ]:


score = [kmeans[i].fit(df_numeric_scaled).score(df_numeric_scaled) for i in range(len(kmeans))]


# These score values signify how far our observations are from the cluster center. We want to keep this score value around 0. A large positive or a large negative value would indicate that the cluster center is far from the observations.
# 
# Based on these scores value, we plot an Elbow curve to decide which cluster size is optimal. Note that we are dealing with tradeoff between cluster size(hence the computation required) and the relative accuracy.

# In[ ]:


pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# Our Elbow point is around cluster size of 5. We will use k=5 to further interpret our clustering result. I'm prefering this number for ease of interpretation in this demo. We can also pick a higher number like 9.

# **Fit K-Means clustering for k=5**

# In[ ]:


kmeans = KMeans(n_clusters=5)
kmeans.fit(df_numeric_scaled)


# As a result of clustering, we have the clustering label. Let's put these labels back into the original numeric data frame.

# In[ ]:


len(kmeans.labels_)


# In[ ]:


df_numeric['cluster'] = kmeans.labels_


# In[ ]:


df_numeric.head()


# # Interpret clustering results                             
# Let's see cluster sizes first.

# In[ ]:


plt.figure(figsize=(12,7))
axis = sns.barplot(x=np.arange(0,5,1),y=df_numeric.groupby(['cluster']).count()['Budget'].values)
x=axis.set_xlabel("Cluster Number")
x=axis.set_ylabel("Number of movies")


# We clearly see that one cluster is the largest and one cluster has the fewest number of movies.
# 
# Let's look at the cluster statistics.

# In[ ]:


df_numeric.groupby(['cluster']).mean()


# We see that one cluster which is also the smallest, is the cluster of movies that received maximum number of votes(in terms of counts) and also have very high rating and duration . Let's see some of the movies that belong to this cluster.

# In[ ]:


size_array = list(df_numeric.groupby(['cluster']).count()['Budget'].values)


# In[ ]:


size_array


# In[ ]:


df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[0])].sample(5)


# In[ ]:


df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[1])].sample(5)


# In[ ]:


df_numeric[df_numeric['cluster']==size_array.index(sorted(size_array)[-1])].sample(5)


# # Visualising Clusters 

# First, let's generate a two-dimensional dataset containing five distinct blobs. To emphasize that this is an unsupervised algorithm, we will leave the labels out of the visualization

# In[ ]:


y_kmeans = kmeans.fit_predict(df_numeric_scaled)

X = df_numeric_scaled.as_matrix(columns=None)

# Plot the 5 clusters

from sklearn.datasets.samples_generator import make_blobs
X, y_true = make_blobs(n_samples=300, centers=5,
                       cluster_std=0.60, random_state=0)
plt.scatter(X[:, 0], X[:, 1], s=50)


# By eye, it is relatively easy to pick out the five clusters. The k-means algorithm does this automatically, and in Scikit-Learn uses the typical estimator API.

# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5)


# # Summary
# In this entry we have learned about the k-means algorithm, including the data normalization before we execute it, the choice of the optimal number of clusters (elbow criterion) and the visualization of the clustering.
# 
# It has been a pleasure to make this post, I have learned a lot! Thank you for reading and if you like it, please upvote it.
