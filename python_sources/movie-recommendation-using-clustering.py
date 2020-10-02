#!/usr/bin/env python
# coding: utf-8

# ## Clustering (HClust and Kmeans)

# # importing libraries

# In[ ]:


import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.spatial.distance import pdist
from sklearn.cluster import KMeans as kmeans, AgglomerativeClustering as hclust
from scipy.cluster.hierarchy import dendrogram, ward


# ## 1. Hierarchical Clustering (HClust)

# # reading data

# In[ ]:


movies=pd.read_csv("http://files.grouplens.org/datasets/movielens/ml-100k/u.item",header=None,sep='|',encoding='iso-8859-1')


# In[ ]:


movies


# # columns labels are missing 

# In[ ]:


movies.columns=["ID", "Title", "ReleaseDate", "VideoReleaseDate", "IMDB", "Unknown", "Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary", "Drama", "Fantasy", "FilmNoir", "Horror", "Musical", "Mystery", "Romance", "SciFi", "Thriller", "War", "Western"]


# In[ ]:


movies.shape


# # dropping unnecessary columns

# In[ ]:


movies=movies.drop(["ID","ReleaseDate","VideoReleaseDate","IMDB"],axis=1)


# In[ ]:


movies.head()


# # droppimg duplicate values

# In[ ]:


movies.drop_duplicates(inplace=True)


# In[ ]:


movies.shape


# ## Calculating distances between each and every point

# In[ ]:


dist=pdist(movies.iloc[:,1:20],'euclidean')  # euclidean distance


# In[ ]:


1664*1663/2


# In[ ]:


len(dist)


# # making model

# In[ ]:


model = hclust(affinity="euclidean",linkage="ward",n_clusters=10)  #lets make 10 clusters


# In[ ]:


model.fit(movies.iloc[:,1:20])


# In[ ]:


model.n_clusters


# In[ ]:


model.labels_   # labels of 10 clusters  


# In[ ]:


len(model.labels_)  #each data point is assigned a label between o to 9


# In[ ]:


np.unique(model.labels_,return_counts=True)  #return_counts will give number of data points in each cluster


# In[ ]:


Z=ward(movies.iloc[:,1:20],)
Z


# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt 
plt.figure(figsize=(8,8))
x = dendrogram(Z)


# In[ ]:


movies['hclust_label'] = model.labels_       #make a new column in cluster in movies dataset which shows clister label for each movie


# In[ ]:


movies


# In[ ]:


movies.groupby(by='hclust_label').mean()


# # cluster no 0 contains movies with comedy genre
# # cluster 1 contains movies with horror and documentary genre

# # Display ten  movies that belong to cluster no 1

# In[ ]:


movies[movies.hclust_label==1].head(10)  # we can see that all these movies are either of horror genre or docimentary or both
                                    # and also our cluster 1 is of hooror and documentary genre
                                    # our model is performing good


# ## 2. Kmeans Clustering

# In[ ]:


km = kmeans(n_clusters=10,random_state=42)


# In[ ]:


km.fit(movies.iloc[:,1:20])


# In[ ]:


km.labels_


# In[ ]:


np.unique(km.labels_,return_counts=True)


# In[ ]:


movies['kmeansclust_labels'] = km.labels_


# In[ ]:


movies.head()


# In[ ]:


movies.groupby('kmeansclust_labels').mean()


# # now if we want to find movies related to a particular movie
# 
# # step1: find the cluster of that movie
# 
# # step2 : find all movies in that cluster

# In[ ]:


# we want to find movies similar to toy story\
movies[movies.Title.str.match("toy story",case=False)]  # FINDING INFO ABOUT THIS MOVIE
# THIS MOVIES BELONGS TO hclust_label 6 and kmeansclust_labels 4
# we can use any of them


# In[ ]:


movies[movies["hclust_label"]==6].head(10)  #lets say 10 movies


# In[ ]:




