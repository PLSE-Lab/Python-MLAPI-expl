#!/usr/bin/env python
# coding: utf-8

# # Dimensionality Reduction Techniques
# 
# This notebook consists in exploring some dimensionality reduction techniques as PCA, t-SNE, MDS and Isomap. 
# 
# The same techniques are applied to 2 different Datasets: The Iris Species and the Breast Cancer Wisconsin. The idea is to work with Iris data set as a benchmark where expected outcomes are clearer to get a feeling for what the methods do, and secondly with the Breast Cancer dataset where data and expected results are not that clear.

# # Imports

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
import seaborn as sn
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from pandas.plotting import scatter_matrix


# # Load Data

# ## Iris Species

# In[ ]:


dataset_iris = pd.read_csv("../input/iris/Iris.csv")
dataset_iris.head()


# In[ ]:


dataset_iris.info()


# In[ ]:


dataset_iris.describe()


# In[ ]:


labels_iris = dataset_iris['Species']
data_iris = dataset_iris.drop("Species",axis=1)
data_iris = data_iris.drop("Id",axis=1)
data_iris.head()


# In[ ]:


data_iris.boxplot(figsize=(40,15))


# In[ ]:


scatter_matrix(data_iris, alpha=0.2, figsize=(10, 6), diagonal='kde');


# ## Breast Cancer Data Set

# In[ ]:


dataset_cancer = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")
dataset_cancer.head()


# In[ ]:


dataset_cancer.info()


# In[ ]:


dataset_cancer.describe()


# In[ ]:


labels_cancer = dataset_cancer['diagnosis']
data_cancer = dataset_cancer.drop("diagnosis",axis=1)
data_cancer = data_cancer.drop("Unnamed: 32", axis=1)
data_cancer = data_cancer.drop("id",axis=1)
data_cancer.head()


# # Initialize Models

# In[ ]:


pca = decomposition.PCA(n_components = 2)
mds = MDS(n_components = 2)
isomap = Isomap(n_components=2)

#Perplexity values for t-SNE
perplexity = [5,20,30,50]


# # Iris Dataset Analysis

# ## PCA

# In[ ]:


pca_data_iris = pca.fit_transform(data_iris)

#Labels incorporation
pca_data_iris = np.vstack((pca_data_iris.T, labels_iris)).T


# In[ ]:


pca_df_iris = pd.DataFrame(data=pca_data_iris, columns=("dim1", "dim2", "Label"))
sn.FacetGrid(pca_df_iris, hue="Label", height=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("PCA for Iris species data set")
plt.show()


# ## t-SNE

# In[ ]:


kl_divergence_iris = []
fig, ax = plt.subplots(1, 4, figsize=(30, 6))
for idx, p in enumerate(perplexity):
    tsne = TSNE(n_components=2, random_state=0, perplexity=p)
    tsne_data_iris = tsne.fit_transform(data_iris)
    kl_divergence_iris.append(tsne.kl_divergence_)
    #Label incorporation
    tsne_data_iris = np.vstack((tsne_data_iris.T, labels_iris)).T
    
    tsne_df_iris = pd.DataFrame(data=tsne_data_iris, columns=("dim1", "dim2", "Label"))
    sn.scatterplot(x=tsne_df_iris['dim1'], y=tsne_df_iris['dim2'], hue=tsne_df_iris["Label"], ax=ax[idx]).set_title("t-SNE for Iris data set with perplexity="+str(p))

    
plt.show()


# In[ ]:


kl_divergence_iris


# ## MDS

# Dimension reduction via MDS is achieved by taking the original set of samples and calculating a dissimilarity (distance) measure for each pairwise comparison of samples.
# 
# 
# The samples are then represented graphically in two dimensions such that the distance between points on the plot approximates their multivariate dissimilarity as closely as possible.
# 
# MDS is using by default the euclidean distance as the dissimilarity method.

# In[ ]:


mds_data_iris = mds.fit_transform(data_iris)

#Label incorporation
mds_data_iris = np.vstack((mds_data_iris.T, labels_iris)).T


# In[ ]:


mds_df_iris = pd.DataFrame(data=mds_data_iris, columns=("dim1", "dim2", "Label"))
sn.FacetGrid(mds_df_iris, hue="Label", height=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("MDS for Iris species data set")
plt.show()


# In[ ]:


mds.stress_


# ## Isomap

# In[ ]:


isomap_data_iris = isomap.fit_transform(data_iris)

#Label incorporation
isomap_data_iris = np.vstack((isomap_data_iris.T, labels_iris)).T


# In[ ]:


isomap_df_iris = pd.DataFrame(data=isomap_data_iris, columns=("dim1", "dim2", "Label"))
sn.FacetGrid(isomap_df_iris, hue="Label", height=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("Isomap for Iris species data set")
plt.show()


# # Breast Cancer Winsconsin Dataset

# ## PCA

# In[ ]:


pca_data_cancer = pca.fit_transform(data_cancer)

#Labels incorporation
pca_data_cancer = np.vstack((pca_data_cancer.T, labels_cancer)).T
print(pca_data_cancer.shape)


# In[ ]:


pca_df_cancer = pd.DataFrame(data=pca_data_cancer, columns=("dim1", "dim2", "Label"))
sn.FacetGrid(pca_df_cancer, hue="Label", height=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("PCA for Breast Cancer data set")
plt.show()


# ## t-SNE

# In[ ]:


kl_divergence_cancer = []
fig, ax = plt.subplots(1, 4, figsize=(30, 6))
for idx, p in enumerate(perplexity):
    tsne = TSNE(n_components=2, random_state=0, perplexity=p)
    tsne_data_cancer = tsne.fit_transform(data_cancer)
    kl_divergence_cancer.append(tsne.kl_divergence_)
    #Label incorporation
    tsne_data_cancer = np.vstack((tsne_data_cancer.T, labels_cancer)).T
    
    tsne_df_cancer = pd.DataFrame(data=tsne_data_cancer, columns=("dim1", "dim2", "Label"))
    sn.scatterplot(x=tsne_df_cancer['dim1'], y=tsne_df_cancer['dim2'], hue=tsne_df_cancer["Label"], ax=ax[idx]).set_title("t-SNE for Cancer data set with perplexity="+str(p))

plt.show()


# In[ ]:


kl_divergence_cancer


# ## MDS

# In[ ]:


mds_data_cancer = mds.fit_transform(data_cancer)

#Label incorporation
mds_data_cancer = np.vstack((mds_data_cancer.T, labels_cancer)).T


# In[ ]:


mds_df_cancer = pd.DataFrame(data=mds_data_cancer, columns=("dim1", "dim2", "Label"))
sn.FacetGrid(mds_df_cancer, hue="Label", height=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("MDS for Breast Cancer data set")
plt.show()


# In[ ]:


mds.stress_


# ## Isomap

# In[ ]:


isomap_data_cancer = isomap.fit_transform(data_cancer)

#Label incorporation
isomap_data_cancer = np.vstack((isomap_data_cancer.T, labels_cancer)).T


# In[ ]:


isomap_df_cancer = pd.DataFrame(data=isomap_data_cancer, columns=("dim1", "dim2", "Label"))
sn.FacetGrid(isomap_df_cancer, hue="Label", height=6).map(plt.scatter, 'dim1', 'dim2').add_legend()
plt.title("Isomap for Breast Cancer data set")
plt.show()

