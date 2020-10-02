#!/usr/bin/env python
# coding: utf-8

# # How to reveal the presence of clusters in a high-dimensional dataset?

# ***Liana Napalkova***
# 
# ***6 October 2018***

# # Table of contents
# 1. [Introduction](#introduction)
# 2. [Data loading](#load_data)
# 3. [Dimensionality reduction](#dimred_data)
# 4. [Conclusions](#concl)

# ## 1. Introduction <a name="introduction"></a>

# Have you ever tried to reveal **the presence of clusters in a high-dimensional dataset**? Indeed this task arises very often. For example, let's imagine that our goal is to understand which factors impact on the ranking of calls in the customer support call center using historical reports. The ranks are in a range from 1 and 5, which means that we have 5 classes. The obvious approach is to build the multi-class classification model in order to reach the goal. We may spend a lot of time trying to build an accurate model that distinguishes well between 5 classes in the target variable. What if after wasting a lot of time, we are still unable to achieve good results? May it happen that  we deal with a "not well behaved" class structures? Maybe we lack explanatory factors in the dataset?
# Of course, it is possible to analyze learning curves in order to understand if the model has a low or high variance of the parameter estimates across samples (bias-variance tradeoff) in order to conclude if more samples or more factors should be added to the dataset. However, there are other complementary approaches that may **help revealing "not well behaved" class structures at an early stage of the analysis**. This is what we are going to investigate in this notebook.

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## 2. Data loading <a name="load_data"></a>

# ![Ref](https://www.worldofchemicals.com/article/104/image/Wine-chemistry.jpg)

# For our analysis we will use the [wine dataset](http://archive.ics.uci.edu/ml/datasets/wine). This dataset contains the results of a chemical analysis of wines grown in the same region in Italy but derived from 3 different cultivars. Our goal will be to reveal the presence of clusters in the wine dataset. In other words, we will check if 3 cultivators are distinguishable in the dataset.
# 
# As we can see, the dataset does not contain null values and all features are numeric, which simplifies the data preprocessing.

# In[ ]:


df=pd.read_csv("../input/Wine.csv")
df.info()


# In[ ]:


df["Customer_Segment"].unique()


# As dependent variable (y), we will select the "Customer_Segment" that specifies the cultivators of wine. The rest of variables will be considered as independent ones (X).

# In[ ]:


X = df.drop("Customer_Segment",axis=1)
y = df["Customer_Segment"]


# ## 3. Dimensionality reduction <a name="dimred_data"></a>

# As it was mentioned in the introduction, the goal of this notebook is to show how to reveal the presence of clusters in a high-dimensional dataset. In our case we have only 13 independent features, but it will be enough to make a quick demo. To reach this goal, we will use dimensionality reduction methods.
# 
# What is the dimensionality reduction? Dimensionality reduction methods convert the high-dimensional dataset into two or three-dimensional data that can be displayed, for example, in a scatterplot.  In other words, it means that we map each high-dimensional vector (in our case we deal with 13 dimensions + the target variable used for coloring the points) into a low-dimensional vector (e.g. 2D). 
# 
# Three different dimensionality reduction methods will be compared:
# 
# * Principal component analysis (PCA)
# * Isometric Mapping
# * t-Distributed Stochastic Neighbor Embedding (t-SNE)

# ### Principal component analysis (PCA)
# 
# Principal component analysis is the traditional dimensionality reduction technique that focuses on keeping the low-dimensional representations of dissimilar datapoints far apart.  This is the **linear method** which means that data can only be summarized by a linear combination of features making it impossible to discover, for example, S-shaped curves. Thus, the limitation of linear methods is their inability to discover more complex structures.

# PCA is available in sklearn. To get more information about how to tune the parameters of PCA, you can refer to `help(PCA)`.

# In[ ]:


X_pca = PCA(n_components="mle",svd_solver='auto').fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)


# What we can conclude by looking at this graphic? Basically, we can see that possibly similar points are quite widely spread. In general, the wine dataset represents the "well behaved" class structure, however if we would deal with a more complex case, we would see that the linear mapping does not allow keeping the low-dimensional representations of very similar datapoints close together.

# ### Isometric Mapping
# 
# Isometric Mapping is a non-linear dimensionality reduction method. Nonlinear dimensionality reduction means that components of the low-dimensional vector are given by non-linear functions of the components of the corresponding high-dimensional vector. Please refer to `help(Isomap)`in order to find more information about the parameters of Isomap.
# 
# One of the disadvantages is that the functionality of this algorithm depends almost on the choice of the number of neighbors. This means that just a few outliers can break the mapping. The sensitivity of the algorithm to the number of neighbors can be seen below. Let's set `n_neighbors` to be equal to 5 and 40.

# In[ ]:


embedding = Isomap(n_components=2,n_neighbors=5)
X_isomap = embedding.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y)


# In[ ]:


embedding = Isomap(n_components=2,n_neighbors=40)
X_isomap = embedding.fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y)


# Besides the sensitivity to the number of neighbors, Isometric Mapping produces solutions in which there are large overlaps between the classes.

# Until now we have seen two dimensionality reduction methods: PCA and Isometric Mapping. The key difference between these two methods is nicely summarize in the below-given graphic ([Source](http://stats.stackexchange.com/questions/124534/how-to-understand-nonlinear-as-in-nonlinear-dimensionality-reduction)).

# ![Difference between linear and non-linear dimensionality reduction methods](https://i.stack.imgur.com/vbxE9.jpg)

# ### t-Distributed Stochastic Neighbor Embedding (t-SNE)

# Finally, we have arrived to the third method called t-Distributed Stochastic Neighbor Embedding (t-SNE). This method is capable of capturing much of the local structure of the high-dimensional data very well, while also revealing global structure such as the presence of clusters at several scales. **t-SNE attempts to preserve local structure: points that are close (according to some metric) in high-dimensional space remain close in the new, low-dimensional space.**
# 
# To find more details about t-SNE, please refer to [this article](http://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf).
# 
# If we run `help(TSNE)`, we can find an interesting parameter of the TSNE class, that is called `metric`. In fact, it is possible to predefine the metric that will be used to calculate distances between samples in a feature array. For example, "euclidean", "cosine" or user-defined function.

# In[ ]:


# Random state
RS = 20150101
X_tsne = TSNE(random_state=RS,learning_rate=5,metric="euclidean").fit_transform(X)

plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)


# ## 4. Conclusions <a name="concl"></a>

# In[ ]:


fig = plt.figure(figsize=(6, 4))
fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(15, 4))
ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
_ = ax1.set_title('PCA')
ax2.scatter(X_isomap[:, 0], X_isomap[:, 1], c=y)
_ = ax2.set_title('Isomap')
ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y)
_ = ax3.set_title('t-SNE')


# If we compare three methods on the wine dataset, we can notice that t-SNE constructs a map in which the separation between 3 classes is quite good. We can very clearly observe the existance of 3 different clusters in our dataset, which basically means that it should be relatively easy to build an accurate classification model for this dataset.
# 
# Thus, we were able to quickly **reveal the presence of clusters in the dataset before even dedicating efforts in building the classification model**.
