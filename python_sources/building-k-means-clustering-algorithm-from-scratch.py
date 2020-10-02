#!/usr/bin/env python
# coding: utf-8

# # K means-clustering from scratch

# In this notebook, we will attempt to build a k-means clustering model from scratch and use it to cluster the unsupervised learning countries dataset. The k-means algorithm is one of the most popular clustering algorithms in Data Science. The standard algorithm was first proposed by Stuart Lloyd of Bell Labs way back in 1957. One important note is that the algorithm only works for numerical data. There are, however, variant which also work with categorical data.

# <img src="https://miro.medium.com/max/2000/1*IXGsBrC9FnSHGJVw9lDhQA.png" width="800">
# 
# ![Source](https://www.medium.com/@tarlanahad/a-friendly-introduction-to-k-means-clustering-algorithm-b31ff7df7ef1)

# ## Table of Contents
# 
# #### 1.  [Exploratory Data Analysis](#section-one)
# #### 2.  [Algorithm Overview](#section-two)
# #### 3.  [Class Structure](#section-three)
# #### 4.  [Implementation](#section-four)
# #### 5.  [Clustering Countries](#section-five)
# #### 6.  [Next Steps](#section-six)

# In[ ]:


from abc import ABC, abstractmethod
import itertools
import numpy as np
import pandas as pd
import tqdm
from pandas_profiling import ProfileReport
import plotly.express as px
from sklearn.preprocessing import StandardScaler


# In[ ]:


PATH = "/kaggle/input/unsupervised-learning-on-country-data/Country-data.csv"
df = pd.read_csv(PATH)


# <a id="section-one"></a>
# ## Exploratory Data Analysis

# Let's have a look at a brief overview of our data. 

# In[ ]:


report = ProfileReport(df)


# In[ ]:


report


# It seems like most of our data is numerical. Moreover, there aren't any missing values either. It looks like we won't need to do too much data cleaning.

# <a id="section-two"></a>
# ## Algorithm Overview

# Let's quickly have a look at the k-means algorithm. The k-means algorithm will try to partition our data into a set of clusters. We will compute the minimum 'distance' between the center of the cluster or cluster means and each data point.

# <img src="https://www.researchgate.net/profile/Pei_Yuan_Zhou/publication/273063437/figure/fig2/AS:391964093632512@1470462930009/The-pseudo-code-for-K-means-clustering-algorithm.png" width="600">
# 
# [Source](http://https://www.researchgate.net/figure/The-pseudo-code-for-K-means-clustering-algorithm_fig2_273063437)

# The above image is the pseudocode for the k-means algorithm. In Step 1 under the subheading Repeat, the notion of 'similarity' is determined by the euclidean distance between a data point the cluster mean.
# 
# In essence, we are trying to minimize the within cluster sum of squares.

# <img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/8dc15ec63e0676fc07e790f61efd89484a6b7922" width="800">
# 
# [Source](http://https://en.wikipedia.org/wiki/K-means_clustering)

# The above function is the objective function of the k-means.

# <a id="section-three"></a>
# ## Class Structure

# Let's try to build a class structure for our k-means algorithm. We will define an abstract base class called KMeansInterface. We will then use a concrete class to define the methods of the abstract class and thus, create our very own k-means clustering algorithm.

# In[ ]:


class KMeansInterface(ABC):
    
    @abstractmethod
    def _init_clusters(self, m):
        """Initialize the clusters for our data"""
        raise NotImplementedError
    
    @abstractmethod
    def _cluster_means(self, X,clusters):
        """Compute the cluster means"""
        raise NotImplementedError
    
    @abstractmethod
    def _compute_cluster(self, X):
        """Assign closest cluster to data point"""
        raise NotImplementedError
        
    @abstractmethod
    def fit(self, X):
        """Run the algorithm"""
        raise NotImplementedError


# We will now use a concrete class to override the abstract methods and develop our implementation for k-means.

# <a id="section-four"></a>
# ## Implementation

# In[ ]:


class KMeans(KMeansInterface):
    def __init__(self, k=3):
        self.k = k
        self.means = None
        self._cluster_ids = None

    @property
    def cluster_ids(self):
        return self._cluster_ids

    def _init_clusters(self, m):
        return np.random.randint(0, self.k, m)

    def _cluster_means(self, X, clusters):
        m, n = X.shape[0], X.shape[1]
        # Extra column to store cluster ids
        temp = np.zeros((m, n + 1))
        temp[:, :n], temp[:, n] = X, clusters
        result = np.zeros((self.k, n))
        for i in range(self.k):
            subset = temp[np.where(temp[:, -1] == i), :n]
            if subset[0].shape[0] > 0:
                result[i] = np.mean(subset[0], axis=0)
            # Choose random data point if a cluster does not 
            # have any data associated with it
            else:
                result[i] = X[np.random.choice(X.shape[0], 1, replace=True)]

        return result

    def _compute_cluster(self, x):
        # Computes closest means to a data point x
        return min(range(self.k), key=lambda i: np.linalg.norm(x - self.means[i])**2)

    def fit(self, X):
        m = X.shape[0]
        # Initialize clusters
        initial_clusters = self._init_clusters(m)
        new_clusters = np.zeros(initial_clusters.shape)
        with tqdm.tqdm(itertools.count()) as t:
            for _ in t:
                # Compute cluster means
                self.means = self._cluster_means(X, initial_clusters)
                for i in range(m):
                    # Assign new cluster ids
                    new_clusters[i] = self._compute_cluster(X[i])
                # Check for data points that have switched cluster ids.
                count_changed = (new_clusters != initial_clusters).sum()
                if count_changed == 0:
                    break
                initial_clusters = new_clusters
                t.set_description(f"changed: {count_changed} / {X.shape[0]}")

        self._cluster_ids = new_clusters


# Note that we also define three properties `k`, `means` and `cluster_ids`. These will store data at intermediate steps which will come in handy later.
# 
# Now, let's walkthrough the code for this algorithm. We will specifically look at the `fit()` method as this method is used to run our algorithm.
# 
# **Steps:**
# 
# 1. We first initialize our data points randomly to clusters. The parameter `k` is the number of clusters hyperparameter. Each data point in our dataset is randomly assigned to one of k cluster ids.
# 2. We also define a variable `new_clusters` to store the new cluster assignments after the minimum distance between the data point and each cluster is computed.
# 3. We now setup an infinite loop using `itertools.count()`. and begin iterating.
# 4. The cluster means are computed. Note that if a particular cluster id fails to be randomly assigned to a data point, we take a random data point and assign it to be the mean of that cluster.
# 5. We then assign the new clusters to each data point based on the minimum squared euclidean distance between a data point and each cluster means.
# 6. Lastly, we check if any of the data points have changed clusters. If they have then we assign the new clusters to serve as the input for computing the cluster means. 
# 7. We repeat steps 4-6 until no data point changes clusters.
# 

# <a id="section-five"></a>
# ## Clustering Countries

# In[ ]:


scaler = StandardScaler()
X = df.iloc[:,1:].values

scaler.fit(X)
X = scaler.transform(X)

k = 4
model = KMeans(k)
model.fit(X)
cluster_ids = model.cluster_ids


# In[ ]:


cluster_ids = cluster_ids.tolist()
cluster_ids = [str(s) for s in cluster_ids]


# In[ ]:


fig = px.scatter(x=X[:, -1],
                     y=X[:, 2],
                     color=cluster_ids,
                     color_discrete_sequence=px.colors.qualitative.D3,
                     hover_name=df["country"].values,
                 size=df.iloc[:,4],
                     opacity=0.7)
fig.update_layout(showlegend=True,
                  xaxis_title="GDP Per Capita",
                  yaxis_title="Total health spending per capita",
                  title="Country Clusters (k = {})".format(k),
                  coloraxis_showscale=False,
                  legend_title_text = "Cluster ids")
fig.show()


# There you have it folks! You have successfully built your own k-means clustering algorithm and used it on a dataset!

# <a id="section-six"></a>
# ## Next Steps

# You can implement your own version of the k-means algorithm using the code. For example, you could use a different stopping condition for the infinite loop. Or, you could use a different distance metric to compute similarity between data points. You can also go ahead and try to implement a function that plots the elbow curve to predict optimal number of clusters.

# I hope all of you enjoyed this mini tutorial. Do let me know what you think! :)
