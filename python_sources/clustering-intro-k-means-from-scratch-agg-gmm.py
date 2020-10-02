#!/usr/bin/env python
# coding: utf-8

# The following is adapted from a Homework assignment for which my professor was kind enough to give me permission to post. I hope this notebook serves as a good starting point example for anyone looking to jump into basic unsupervised learning methods. 
# 
# ### The hardest problem 
# The trouble with clustering algorithms is that we don't know great ways to evaluate our clusters. I'll attempt to evaluate mine with:
# * Graphical comparison to the true clusters
# * SSE within and between clusters
# * A Scree plot
# 
# I'll explain how all of those work later on when we get there.

# In[ ]:


# Author: Caleb Woy

import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt # plotting
import seaborn as sb # plotting
import os # Reading data
import matplotlib.pylab as plt # plotting hyperparamter cost curves
from sklearn import preprocessing # scaling features
import random # random centroid generation


# # Loading Data

# In[ ]:


path_to_data = "/kaggle/input/"

# Loading the training and test data sets into pandas
small_xy = pd.read_csv(path_to_data + "/small_Xydf.csv", header=0)
small_xy = small_xy.drop(columns=["Unnamed: 0"])

large_xy = pd.read_csv(path_to_data + "/large_Xydf.csv", header=0)
large_xy = large_xy.drop(columns=["Unnamed: 0"])

red_wine = pd.read_csv(path_to_data + "/winequality-red.csv", sep=';', 
                       header=0)
red_wine['y'] = red_wine.apply(lambda x: int(x['quality'] - 3), axis=1)


# In[ ]:


# A two dimensional data set of 100 observations.
# y is the label representing the true clustering.
small_xy.head()


# In[ ]:


# A two dimensional data set of 1000 observations.
# y is the label representing the true clustering.
large_xy.head()


# In[ ]:


# An eleven dimensional data set of 1600 observations.
# quality is the label representing the true clustering.
# I've added the y label to make feeding the data into 
# my own algorithm easier.
red_wine.head()


# # Implementing K-Means Algorithm

# In[ ]:


"""
Implementing K-means clustering algorithm from scratch.

data: pandas dataframe, X data to be used for clustering
k: int, the number of centroids to assign
verbose: Boolean, default is true, if false nothing is printed
"""
def Kmeans(data, k, verbose = True):
    random.seed(k)
    if verbose:
        print(f'TESTING FOR K = {k}')
    # scale the x data uniformly
    x = data.values # Get data as np array
    standard_scaler = preprocessing.StandardScaler()
    x_scaled = standard_scaler.fit_transform(x)
    data = pd.DataFrame(x_scaled)
    # calculating the max and min values for each column and storing them in 
    # an array
    min_max = data.apply(lambda x:
        pd.Series(index=['min','max'],data=[x.min(),x.max()]))
    min_max_list = min_max.T.values.tolist()
    # Randomly initialize centroids within max and min range of feature vectors
    centroids = {}
    for i in range(k):
        centroids[i] = [ 
            random.uniform(x[0], x[1]) for x in min_max_list]
    # Create arrays for assignments to check for convergence
    curr = np.array([0 for i in range(data.shape[0])])
    prev = np.array([1 for i in range(data.shape[0])])
    # iterate until assignments don't change
    while not (curr == prev).all():
        # copy current into prev
        prev = np.copy(curr)
        if verbose:
            print(f'.', end=' ')
        # Assignment step
        # iterate over each row in the data 
        for index, row in data.iterrows():
            # set min container and flag
            min_dist, min_centroid = float("inf"), 0
            # iterate over each centroid
            for l in range(k):
                # get the centroid vector
                a = np.array(centroids[l])
                # get the row vector
                b = np.array(row.T.values.tolist())
                # calculate euclidean distance
                dist = np.linalg.norm(a - b)
                # set min distance if min
                if dist < min_dist:
                    min_dist, min_centroid = dist, l
            # assign current datum
            curr[index] = min_centroid
        # Getting the counts of each cluster assignment
        curr_counter = np.array(curr)
        unique, counts = np.unique(curr_counter, return_counts=True)
        counts = dict(zip(unique, counts))
        # init container for new centroids
        new_centroids = {}
        for i in range(k):
            new_centroids[i] = [0 for x in min_max_list]
        # Update step
        # iterate over all data
        for index, row in data.iterrows():
            # retrieve row
            row = row.T.values.tolist()
            # retrieve centroid
            cent = new_centroids[curr[index]]
            # iterate over each dimension of the centroid and accumulate the
            # avg distance
            for j in range(len(cent)):
                cent[j] += row[j] / counts[curr[index]]
            new_centroids[curr[index]] = cent
        # reassign centroid
        centroids = new_centroids
    if verbose:    
        print('FINISHED')
    return curr


# ## Implement cluster SSE, total SSE, and SSB calculations 

# In[ ]:


"""
Prints the cluster SSE.

data: pandas dataframe, X data
verbose: Boolean, default is true, if false nothing is printed
"""
def cluster_sse(data, verbose = True):
    # calculating cluster centroids
    labels = data['y']
    counts = labels.value_counts()
    data = data.drop(columns=['y'])
    centroids = {}
    for i in range(len(counts)):
        centroids[i] = [0 for j in range(data.shape[1])]
    for index, row in data.iterrows():
        point = np.array(row.T.values.tolist())
        cent = centroids[labels[index]]
        for j in range(len(cent)):
            cent[j] += row[j] / counts[labels[index]]
        centroids[labels[index]] = cent
    # calculating cluster sse
    sse = [0 for j in range(len(counts))]
    for index, row in data.iterrows():
        a = np.array(centroids[labels[index]])
        b = np.array(row.T.values.tolist())
        dist = np.linalg.norm(a - b)
        sse[labels[index]] += dist
    if verbose:
        print(f'CLUSTER SSE IS: {sse}')
    return sse


# In[ ]:


"""
Prints the total SSE.

data: pandas dataframe, X data
verbose: Boolean, default is true, if false nothing is printed
"""
def total_sse(data, verbose = True):
    # calculating cluster centroids
    labels = data['y']
    counts = labels.value_counts()
    data = data.drop(columns=['y'])
    centroids = {}
    for i in range(len(counts)):
        centroids[i] = [0 for j in range(data.shape[1])]
    for index, row in data.iterrows():
        point = np.array(row.T.values.tolist())
        cent = centroids[labels[index]]
        for j in range(len(cent)):
            cent[j] += row[j] / counts[labels[index]]
        centroids[labels[index]] = cent
    # calculating cluster sse
    sse = 0
    for index, row in data.iterrows():
        a = np.array(centroids[labels[index]])
        b = np.array(row.T.values.tolist())
        dist = np.linalg.norm(a - b)
        sse += dist
    if verbose:
        print(f'TOTAL SSE IS: {sse}')
    return sse


# In[ ]:


"""
Prints the cluster SSB.

data: pandas dataframe, X data
verbose: Boolean, default is true, if false nothing is printed
"""
def cluster_ssb(data, verbose = True):
    # calculating cluster centroids
    labels = data['y']
    counts = labels.value_counts()
    data = data.drop(columns=['y'])
    centroids = {}
    for i in counts.iteritems():
        centroids[i[0]] = [0 for j in range(data.shape[1])]
    for index, row in data.iterrows():
        point = np.array(row.T.values.tolist())
        cent = centroids[labels[index]]
        for j in range(len(cent)):
            cent[j] += row[j] / counts[labels[index]]
        centroids[labels[index]] = cent
     # calculating global centroid
    glob_centroid = [0 for j in range(data.shape[1])]
    for index, row in data.iterrows():
        point = np.array(row.T.values.tolist())
        for j in range(len(glob_centroid)):
            glob_centroid[j] += row[j] / data.shape[0]
    # calculating total ssb
    ssb = 0
    a = np.array(glob_centroid)
    for key, b in centroids.items():
        b = np.array(b)
        dist = np.linalg.norm(a - b) * counts[key]
        ssb += dist
    if verbose:
        print(f'TOTAL SSB IS: {ssb}')
    return ssb


# ## Implement Testing Functions

# In[ ]:


def test_Kmeans_small_large(data, k_collection, k_special = 0):
    x = data.drop(columns=['y'])
    for k in k_collection:
        x['y'] = Kmeans(x, k)
        # Calculating test SSE, SSB
        cluster_sse(x)
        sse = total_sse(x)
        ssb = cluster_ssb(x)
        print(f'SSE RATIO: {sse / (sse + ssb)}')
        if k == k_special:
            plt.scatter(x['X0'], x['X1'], c=x['y'])
            plt.show()
        x = x.drop(columns=['y'])
        print()


# In[ ]:


def test_Kmeans_redwine(data, k, k_special = 0):
    x = data.drop(columns=['quality', 'y'])
    x['y'] = Kmeans(x, k)
    # Calculating test SSE, SSB
    cluster_sse(x)
    sse = total_sse(x)
    ssb = cluster_ssb(x)
    print(f'SSE RATIO: {sse / (sse + ssb)}')
    if k == k_special:
        sb.pairplot(x, 
                    vars=x.loc[:, x.columns != 'y'], 
                    hue ='y',
                    diag_kind = 'hist')
    x = x.drop(columns=['y'])
    print()


# ## Implement Evaluation Functions

# In[ ]:


def Eval_Kmeans_scree(data, k_collection):
    x = data.drop(columns=['y'])
    scores = [0 for i in k_collection]
    for index, k in enumerate(k_collection):
        x['y'] = Kmeans(x, k, False)
        # Calculating test SSE, SSB
        sse = total_sse(x, False)
        ssb = cluster_ssb(x, False)
        scores[index] = (k, sse / (sse + ssb))
        x = x.drop(columns=['y'])
    plt.plot(*zip(*scores))
    plt.title("Scree Plot")
    plt.ylabel("sse / (sse + ssb)")
    plt.xlabel("k")
    plt.show()


# ## Testing on small dataset

# In[ ]:


# Calculating true cluster SSE
print('TRUE ', end='')
sse = cluster_sse(small_xy)


# In[ ]:


# Calculating total SSE
print('TRUE ', end='')
sse = total_sse(small_xy)


# In[ ]:


# Calculating true cluster SSB
print('TRUE ', end='')
ssb = cluster_ssb(small_xy)


# In[ ]:


# plotting true clustering
plt.scatter(small_xy['X0'], small_xy['X1'], c=small_xy['y'])


# To make observations and further evaluate the model I'll use a Scree plot. The Scree plot is the relationship between [cluster see / (cluster sse + cluster ssb)] and k. To pick the best k value, the technique is to look for an elbow in the graph. An elbow is a spot where the rate of decrease changes from a sharp drop to a more shallow one. The elbow's interpretation is the k value such that we stop seeing benefit from a true clustering and start seeing the benefit of overfitting.

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_Kmeans_scree(small_xy, k_collection)


# There's a pretty clear elbow here at k = 4. That's probably the best clustering. I'll plot the clustering below along with relevant statistics.

# In[ ]:


k_collection = [4]
test_Kmeans_small_large(small_xy, k_collection, 4)


# ## Testing on large dataset

# In[ ]:


# Calculating true cluster SSE
print('TRUE ', end='')
sse = cluster_sse(large_xy)


# In[ ]:


# Calculating total SSE
print('TRUE ', end='')
sse = total_sse(large_xy)


# In[ ]:


# Calculating true cluster SSB
print('TRUE ', end='')
ssb = cluster_ssb(large_xy)


# In[ ]:


# plotting true clustering
plt.scatter(large_xy['X0'], large_xy['X1'], c=large_xy['y'])


# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_Kmeans_scree(large_xy, k_collection)


# It appears the elbow of the Scree plot is at k = 5. That's probably the best clustering. Afterwards we see an increase at k = 6 and then further decrease.

# In[ ]:


k_collection = [5]
test_Kmeans_small_large(large_xy, k_collection, 5)


# ## Testing on wine dataset

# In[ ]:


# Calculating true cluster SSE
print('TRUE ', end='')
red_wine_no_qual = red_wine.drop(columns=['quality'])
sse = cluster_sse(red_wine_no_qual)


# In[ ]:


# Calculating total SSE
print('TRUE ', end='')
sse = total_sse(red_wine_no_qual)


# In[ ]:


# Calculating true cluster SSB
print('TRUE ', end='')
ssb = cluster_ssb(red_wine_no_qual)


# In[ ]:


# plotting true clustering
sb.pairplot(red_wine_no_qual, 
            hue='y', 
            vars=red_wine_no_qual.columns[:-1],
            diag_kind = 'hist')


# In[ ]:


# Testing on wine dataset, had to split them out of the loop to get
# the seaborn plot to show in proper order. It takes a while to load.


# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_Kmeans_scree(red_wine.drop(columns=['quality']), k_collection)


# It appears the elbow of the Scree plot is at k = 5. That's likely the best clustering. I'll plot it below long with the relevant stats.

# In[ ]:


test_Kmeans_redwine(red_wine, 5, 5)


# # Comparing with Sklearn K-means

# In[ ]:


from sklearn import cluster as cl # for testing K-means


# In[ ]:


def Eval_SKLearn_Kmeans_scree(data, k_collection):
    x = data.drop(columns=['y'])
    scores = [0 for i in k_collection]
    for index, k in enumerate(k_collection):
        kmeans = cl.KMeans(n_clusters = k)
        x['y'] = kmeans.fit_predict(x)
        # Calculating test SSE, SSB
        sse = total_sse(x, False)
        ssb = cluster_ssb(x, False)
        scores[index] = (k, sse / (sse + ssb))
        x = x.drop(columns=['y'])
    plt.plot(*zip(*scores))
    plt.title("Scree Plot")
    plt.ylabel("sse / (sse + ssb)")
    plt.xlabel("k")
    plt.show()


# In[ ]:


def test_SKLearn_Kmeans_small_large(data, k_collection, k_special = 0):
    x = data.drop(columns=['y'])
    for k in k_collection:
        kmeans = cl.KMeans(n_clusters = k)
        x['y'] = kmeans.fit_predict(x)
        # Calculating test SSE, SSB
        cluster_sse(x)
        sse = total_sse(x)
        ssb = cluster_ssb(x)
        print(f'SSE RATIO: {sse / (sse + ssb)}')
        if k == k_special:
            plt.scatter(x['X0'], x['X1'], c=x['y'])
            plt.show()
        x = x.drop(columns=['y'])
        print()


# In[ ]:


def test_SKLearn_Kmeans_redwine(data, k, k_special = 0):
    x = data.drop(columns=['quality', 'y'])
    kmeans = cl.KMeans(n_clusters = k)
    x['y'] = kmeans.fit_predict(x)
    # Calculating test SSE, SSB
    cluster_sse(x)
    sse = total_sse(x)
    ssb = cluster_ssb(x)
    print(f'SSE RATIO: {sse / (sse + ssb)}')
    if k == k_special:
        sb.pairplot(x, 
                    vars=x.loc[:, x.columns != 'y'], 
                    hue ='y',
                    diag_kind = 'hist')
    x = x.drop(columns=['y'])
    print()


# ## Evaluating on small data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_Kmeans_scree(small_xy, k_collection)


# The Scree plot here is pretty radically different than the one I produced with my kmeans algorithm. The rate of change is decreasing from k = 2 to 4 but then starts increasing and levelling off after k = 4. I'll print the cluster for k = 4 below along with relevant statistics.

# In[ ]:


test_SKLearn_Kmeans_small_large(small_xy, [4], k_special = 4)


# ## Evaluating on large data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_Kmeans_scree(large_xy, k_collection)


# The elbow here appears at 3. It's more apparent than on the small data set. K = 3 is again, the same as the k value I chose before. I'll plot the clustering below.

# In[ ]:


test_SKLearn_Kmeans_small_large(large_xy, [3], k_special = 3)


# ## Evaluating on Red Wine data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_Kmeans_scree(red_wine.drop(columns=['quality']), k_collection)


# The best looking cluster here appears to be at k = 5. I'll plot it below.

# In[ ]:


test_SKLearn_Kmeans_redwine(red_wine, 5, k_special = 5)


# # Gaussian Mixture Model Clustering
# 
# From sklearn:
# 
#     A Gaussian mixture model is a probabilistic model that assumes all the data points are generated from a mixture of a finite number of Gaussian distributions with unknown parameters. One can think of mixture models as generalizing k-means clustering to incorporate information about the covariance structure of the data as well as the centers of the latent Gaussians.

# In[ ]:


from sklearn.mixture import GaussianMixture as GMM


# In[ ]:


def Eval_SKLearn_GMM_scree(data, k_collection):
    x = data.drop(columns=['y'])
    scores = [0 for i in k_collection]
    for index, k in enumerate(k_collection):
        gmm = GMM(n_components = k)
        x['y'] = gmm.fit_predict(x)
        # Calculating test SSE, SSB
        sse = total_sse(x, False)
        ssb = cluster_ssb(x, False)
        scores[index] = (k, sse / (sse + ssb))
        x = x.drop(columns=['y'])
    plt.plot(*zip(*scores))
    plt.title("Scree Plot")
    plt.ylabel("sse / (sse + ssb)")
    plt.xlabel("k")
    plt.show()


# In[ ]:


def test_SKLearn_GMM_small_large(data, k_collection, k_special = 0):
    x = data.drop(columns=['y'])
    for k in k_collection:
        gmm = GMM(n_components = k)
        x['y'] = gmm.fit_predict(x)
        # Calculating test SSE, SSB
        cluster_sse(x)
        sse = total_sse(x)
        ssb = cluster_ssb(x)
        print(f'SSE RATIO: {sse / (sse + ssb)}')
        if k == k_special:
            plt.scatter(x['X0'], x['X1'], c=x['y'])
            plt.show()
        x = x.drop(columns=['y'])
        print()


# In[ ]:


def test_SKLearn_GMM_redwine(data, k, k_special = 0):
    x = data.drop(columns=['quality', 'y'])
    gmm = GMM(n_components = k)
    x['y'] = gmm.fit_predict(x)
    # Calculating test SSE, SSB
    cluster_sse(x)
    sse = total_sse(x)
    ssb = cluster_ssb(x)
    print(f'SSE RATIO: {sse / (sse + ssb)}')
    if k == k_special:
        sb.pairplot(x, 
                    vars=x.loc[:, x.columns != 'y'], 
                    hue ='y',
                    diag_kind = 'hist')
    x = x.drop(columns=['y'])
    print()


# ## Evaluating on small data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_GMM_scree(small_xy, k_collection)


# In[ ]:


test_SKLearn_GMM_small_large(small_xy, [5], k_special = 5)


# ## Evaluating on large data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_GMM_scree(large_xy, k_collection)


# In[ ]:


test_SKLearn_GMM_small_large(large_xy, [3], k_special = 3)


# ## Evaluating on Red Wine data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_GMM_scree(red_wine.drop(columns=['quality']), k_collection)


# In[ ]:


test_SKLearn_GMM_redwine(red_wine, 5, k_special = 5)


# # Agglomerative Clustering
# 
# From sklearn:
# 
#     The AgglomerativeClustering object performs a hierarchical clustering using a bottom up approach: each observation starts in its own cluster, and clusters are successively merged together. The linkage criteria determines the metric used for the merge strategy:
# 
#     Ward minimizes the sum of squared differences within all clusters. It is a variance-minimizing approach and in this sense is similar to the k-means objective function but tackled with an agglomerative hierarchical approach.
# 
#     Maximum or complete linkage minimizes the maximum distance between observations of pairs of clusters.
# 
#     Average linkage minimizes the average of the distances between all observations of pairs of clusters.
# 
#     Single linkage minimizes the distance between the closest observations of pairs of clusters.
#     
# I'll be testing with the ward strategy

# In[ ]:


from sklearn.cluster import AgglomerativeClustering as AGG


# In[ ]:


def Eval_SKLearn_agg_scree(data, k_collection):
    x = data.drop(columns=['y'])
    scores = [0 for i in k_collection]
    for index, k in enumerate(k_collection):
        agg = AGG(n_clusters = k)
        x['y'] = agg.fit_predict(x)
        # Calculating test SSE, SSB
        sse = total_sse(x, False)
        ssb = cluster_ssb(x, False)
        scores[index] = (k, sse / (sse + ssb))
        x = x.drop(columns=['y'])
    plt.plot(*zip(*scores))
    plt.title("Scree Plot")
    plt.ylabel("sse / (sse + ssb)")
    plt.xlabel("k")
    plt.show()


# In[ ]:


def test_SKLearn_agg_small_large(data, k_collection, k_special = 0):
    x = data.drop(columns=['y'])
    for k in k_collection:
        agg = AGG(n_clusters = k)
        x['y'] = agg.fit_predict(x)
        # Calculating test SSE, SSB
        cluster_sse(x)
        sse = total_sse(x)
        ssb = cluster_ssb(x)
        print(f'SSE RATIO: {sse / (sse + ssb)}')
        if k == k_special:
            plt.scatter(x['X0'], x['X1'], c=x['y'])
            plt.show()
        x = x.drop(columns=['y'])
        print()


# In[ ]:


def test_SKLearn_agg_redwine(data, k, k_special = 0):
    x = data.drop(columns=['quality', 'y'])
    agg = AGG(n_clusters = k)
    x['y'] = agg.fit_predict(x)
    # Calculating test SSE, SSB
    cluster_sse(x)
    sse = total_sse(x)
    ssb = cluster_ssb(x)
    print(f'SSE RATIO: {sse / (sse + ssb)}')
    if k == k_special:
        sb.pairplot(x, 
                    vars=x.loc[:, x.columns != 'y'], 
                    hue ='y',
                    diag_kind = 'hist')
    x = x.drop(columns=['y'])
    print()


# ## Evaluating on Small data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_agg_scree(small_xy, k_collection)


# In[ ]:


test_SKLearn_agg_small_large(small_xy, [6], k_special = 6)


# ## Evaluating on Large data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_agg_scree(large_xy, k_collection)


# In[ ]:


test_SKLearn_agg_small_large(large_xy, [5], k_special = 5)


# ## Evaluating on Red Wine data set

# In[ ]:


k_collection = [x for x in range(2, 10)]
Eval_SKLearn_agg_scree(red_wine.drop(columns=['quality']), k_collection)


# In[ ]:


test_SKLearn_agg_redwine(red_wine, 4, k_special = 4)


# # Summary of results
# 
# The best model for each data set is different and different values of K were chosen for many of the models. The K-means from scratch and sklearn K-means produced similar performing clusters on the small and large data set but varied wildly on the Red Wine data set. This disparity must come from something caused by higher dimensionality. On the Red Wine data set the Gaussian Mixture Model performed much wore than the other algorithms. This was likely due to the features of Red_wine being mostly non-normal. The features of the large and small data sets are closer to normal. On all data sets the Agglomerative clustering algorithm performed pretty well. No algorithm was able to produce the embedded clusters present in the true clusterings, however, any of these top choice clusterings could be deployed for further evaluation against true popultion data in the future. 
# 
# ### Small data set
# 
# 1. Sklearn K-Means
# 
#     CLUSTER SSE IS: [25.113650316705517, 27.786510770471924, 28.88352039998379, 29.171406839346304, 23.197524969970992, 16.210872383529896]<br/>
#     TOTAL SSE IS: 150.36348568000844<br/>
#     TOTAL SSB IS: 434.4472418710884<br/>
#     SSE RATIO: 0.2571147870519709<br/>
# 
# 2. Gaussian Mixture Model
# 
#     CLUSTER SSE IS: [22.259240947405164, 19.92423302517606, 29.943740736444642, 26.036592357641275, 20.790905667554412, 34.93146095209374]<br/>
#     TOTAL SSE IS: 153.88617368631526<br/>
#     TOTAL SSB IS: 438.35478003475976<br/>
#     SSE RATIO: 0.25983710298898094<br/>
# 
# 3. Agglomerative Clustering
# 
#     CLUSTER SSE IS: [32.24772795411403, 25.22911100567976, 31.20294475079321, 27.592873393793326, 19.92423302517606, 17.463462019390224]<br/>
#     TOTAL SSE IS: 153.66035214894663<br/>
#     TOTAL SSB IS: 432.4894887114212<br/>
#     SSE RATIO: 0.26215199840948433<br/>
# 
# 4. From scratch K-Means
# 
#     CLUSTER SSE IS: [14.321048894655888, 23.197524969970992, 27.786510770471924, 19.443766260326196, 25.22911100567976, 47.97577893804139]<br/>
#     TOTAL SSE IS: 157.95374083914618<br/>
#     TOTAL SSB IS: 433.7257603360633<br/>
#     SSE RATIO: 0.26695827813100553<br/>
# 
# ### Large data set
# 
# 1. Agglomerative Clustering
# 
#     CLUSTER SSE IS: [574.1139319868796, 615.2491341845708, 242.14940863136596, 456.39404810917864, 301.95288952523384]<br/>
#     TOTAL SSE IS: 2189.859412437227<br/>
#     TOTAL SSB IS: 4061.450681598995<br/>
#     SSE RATIO: 0.3503040769848168<br/>
# 
# 2. From scratch K-Means
# 
#     CLUSTER SSE IS: [662.0238124444755, 549.0543932592994, 389.12173771755505, 656.2668969006731]<br/>
#     TOTAL SSE IS: 2256.466840322003<br/>
#     TOTAL SSB IS: 3980.4496480032612<br/>
#     SSE RATIO: 0.36179205614598653<br/>
# 
# 3. Gaussian Mixture Model
# 
#     CLUSTER SSE IS: [376.6374562533373, 702.1533787373509, 544.9836781878048, 618.9660942181022]<br/>
#     TOTAL SSE IS: 2242.7406073965976<br/>
#     TOTAL SSB IS: 3938.3921164928233<br/>
#     SSE RATIO: 0.3628365070901396<br/>
# 
# 4. Sklearn K-Means
# 
#     CLUSTER SSE IS: [537.5157636165864, 666.3510261776227, 390.04468936603956, 646.4919803291158]<br/>
#     TOTAL SSE IS: 2240.403459489365<br/>
#     TOTAL SSB IS: 3891.086218694983<br/>
#     SSE RATIO: 0.36539300840065864<br/>
# 
# ### Red Wine data set
# 
# 1. Sklearn K-Means
# 
#     CLUSTER SSE IS: [2471.767864327411, 3994.8735714567297, 3783.6483217172804, 11.0, 1116.7179271251641, 3297.3475044500997]<br/>
#     TOTAL SSE IS: 14675.35518907669<br/>
#     TOTAL SSB IS: 41888.52501338517<br/>
#     SSE RATIO: 0.25944746252464423<br/>
# 
# 2. Agglomerative Clustering
# 
#     CLUSTER SSE IS: [4806.092560012689, 4724.766607466435, 5114.886149354945, 2770.9914134637097]<br/>
#     TOTAL SSE IS: 17416.736730297715<br/>
#     TOTAL SSB IS: 41716.80859601336<br/>
#     SSE RATIO: 0.29453225972142505<br/>
# 
# 3. Gaussian Mixture Model
# 
#     CLUSTER SSE IS: [10954.087313831726, 3429.649604372425, 2638.962615217396, 1071.0795756489704, 660.6866912068739, 8562.14100201374]<br/>
#     TOTAL SSE IS: 27316.606802291084<br/>
#     TOTAL SSB IS: 39418.009047600324<br/>
#     SSE RATIO: 0.40933189551484517<br/>
# 
# 4. From scratch K-Means
# 
#     CLUSTER SSE IS: [10888.685381436475, 1496.5269048973569, 10536.206011510541, 34.022488281135466, 6280.87021648497, 749.2728542794449]<br/>
#     TOTAL SSE IS: 29985.58385688987<br/>
#     TOTAL SSB IS: 35692.36077359333<br/>
#     SSE RATIO: 0.456554845398932<br/>

# In[ ]:




