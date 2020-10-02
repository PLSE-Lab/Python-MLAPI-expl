#!/usr/bin/env python
# coding: utf-8

# This is a comparative analysis of different clustering algorithms. 
# The data consists of 2000 rows of different product names. 
# 

# **Imports**

# In[ ]:


import numpy as np  
import re  
import nltk  
from sklearn import metrics
from nltk.corpus import stopwords  
import pandas as pd
import csv 
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer
from yellowbrick.text import TSNEVisualizer
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import MeanShift
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch


# **Reading the data file**

# In[ ]:



stemmer = SnowballStemmer('english')
lemmer=WordNetLemmatizer()

df = pd.read_csv("../input/Productnames.csv", sep=',',header=None,lineterminator='\n')
documents = []


# **Cleaning the data**
# 
# It includes stemming and lemmatizing. 

# In[ ]:


for sen in df.iterrows(): 
    # Remove all the special characters 
    document = re.sub(r'\W', ' ', str(sen))
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', '', document)
    #remove numbers
    document = re.sub(r'[0-9]+', '', document)
    # Remove single characters from the start
    document = re.sub(r'\^[a-zA-Z]\s+', '', document) 
    # Substituting multiple spaces with single space
    document = re.sub(r'\s+', ' ', document, flags=re.I)
    # Removing prefixed 'b'
    document = re.sub(r'^b\s+', '', document)
    # Converting to Lowercase
    document = document.lower()
    # Stemming and Lemmatization
    document = document.split()
    document=[stemmer.stem(word) for word in document]
    document = [lemmer.lemmatize(word) for word in document]
    document = ' '.join(document)
    document = re.sub(r'name dtype object', '', document)
    document = re.sub(r'\W*\b\w{1,2}\b', '', document)

    documents.append(document)


# Converting document into TF IDF matrix. If we include the ngram, the sillhouette score increases but there is also increase in the size of the biggest cluster. 

# In[ ]:


vectorizer = TfidfVectorizer(max_features=2000,
                         		stop_words='english',
                                 use_idf=True)#, ngram_range = (2,3))
X = vectorizer.fit_transform(documents)


# **K-Means**

# In[ ]:


true_k = 50	
model = KMeans(n_clusters=true_k, init='k-means++', max_iter=1000)


# In[ ]:


cluster_label=model.fit_predict(X)


# Counting of number of products in each cluster is done through Counter. We save the documents with their cluster number in a file.

# In[ ]:


Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_Kmeans.csv')


# The Sillouette score is a measure of an object's similarity to its own cluster compared to other clusters. It's range is from -1 to +1. High value indicates the object matches it's cluster greatly and poorly to neighbouring clusters.

# In[ ]:


score_kmeans = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_kmeans)


# Visualization

# In[ ]:


tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()


# **Affinity Propogation**

# In[ ]:


model = AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.5, max_iter=200, preference=None, verbose=False)


# In[ ]:


cluster_label=model.fit_predict(X)


# In[ ]:


Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_Affinity.csv')


# In[ ]:


score_affinity = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_affinity)


# In[ ]:


tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()


# **Spectral Clustering**

# In[ ]:


model = SpectralClustering(n_clusters=true_k, assign_labels="discretize",random_state=0)


# In[ ]:


cluster_label=model.fit_predict(X)


# In[ ]:


Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_Spectral.csv')


# In[ ]:


score_spectral = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_spectral)


# In[ ]:


tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()


# **Birch**

# In[ ]:


model = Birch(branching_factor=100, n_clusters=true_k, threshold=0.5,compute_labels=True)


# In[ ]:


cluster_label=model.fit_predict(X)


# In[ ]:


Counter(cluster_label)
c = pd.DataFrame(cluster_label,documents)
c.to_csv('clusterfile_birch.csv')


# In[ ]:


score_birch = metrics.silhouette_score(X, model.labels_, metric='euclidean')
print('Silhouette score: ',score_birch)


# In[ ]:


tsne = TSNEVisualizer()
tsne.fit(X,cluster_label)
tsne.poof()


# In[ ]:


print('For comparison ')
print(score_kmeans)
print(score_affinity)
print(score_spectral)
print(score_birch)


# There is one big cluster that takes most of the products in it. 
# The big cluster can be reclustered as well. The smaller cluster can be merged into bigger clusters. We can choose to increase the number of clusters as well but there was no significant change in the clustering. There were clusters formed with 1 product as well. 
# As mentioned before, sillouette score closer to 1 is considered best. Among the ones we have, Affinity Propogation perfomed better. If we increase the ngram_range to (3,4) of TFIDFVectorizer, the score goes to 0.3xx but the big cluster is even more enlarged. 
