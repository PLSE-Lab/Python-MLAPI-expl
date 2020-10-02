#!/usr/bin/env python
# coding: utf-8

# ## Introduction
# This kernel performs the cancer RNA-Seq data visualizations using Dimensionality Reduction techniques.  

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score


# In[ ]:


#Creating dataframe for data.
data = pd.read_csv('/kaggle/input/data.csv', delimiter=',',index_col=0)
nRow, nCol = data.shape
print(f'There are {nRow} rows and {nCol} columns in data.')


# In[ ]:


#Inspect first 5 rows of data.
data.head(5)


# In[ ]:


#Creating dataframe for labels.
labels = pd.read_csv('/kaggle/input/labels.csv', delimiter=',')
nRow, nCol = labels.shape
print(f'There are {nRow} rows and {nCol} columns in labels dataframe.')


# In[ ]:


#Inspect first 5 rows of labels.
labels.head(5)


# In[ ]:


#Find unique classes of cancer subtypes.
labels['Class'].unique()


# In[ ]:


#Create a 2D numpy array of values in data.
X = data.values
X[0:5]


# In[ ]:


#Standardize the features before performing dimensionality reduction, (mean=0,standard deviation =1)
X_std = StandardScaler().fit_transform(X)


# In[ ]:


#Visualize data using Principal Component Analysis.
print("Performing Principal Component Analysis (PCA) ...")
pca = PCA(n_components = 2).fit_transform(X_std)
pca_df = pd.DataFrame(data=pca, columns=['PC1','PC2']).join(labels)
pca_df.head(5)
palette = sns.color_palette("muted", n_colors=5)
sns.set_style("white")
sns.scatterplot(x='PC1',y='PC2',hue='Class',data=pca_df, palette=palette, linewidth=0.2, s=30, alpha=1).set_title('PCA')


# In[ ]:


#Fitting PCA on Data
pca_std = PCA().fit(X_std)
percent_variance=pca_std.explained_variance_ratio_*100

#Plotting Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca_std.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Variance (%)') #for each component
plt.title('Cancer Dataset - Cumulative Explained Variance')
plt.show()


# In[ ]:


#Visualize data using t-SNE.
print("Performing T-Distributed Stochastic Neighbor Embedding (tSNE) ...")
model = TSNE(learning_rate = 10, n_components = 2, perplexity = 30)
tsne = model.fit_transform(X_std)
tsne_df = pd.DataFrame(data=tsne, columns=['t-SNE1','t-SNE2']).join(labels)
palette = sns.color_palette("muted", n_colors=5)
sns.set_style("white")
sns.scatterplot(x='t-SNE1',y='t-SNE2',hue='Class',data=tsne_df, palette=palette, linewidth=0.2, s=30, alpha=1).set_title('t-SNE')


# In[ ]:


#Measure execution time for t-SNE
def tsne_model(X):
    model = TSNE(learning_rate = 10, n_components = 2, random_state = 123, perplexity = 30)
    tsne = model.fit_transform(X)
    return tsne
from timeit import Timer
  
t = Timer(lambda: tsne_model(X_std))
print(t.timeit(number=1))


# In[ ]:


#Visualize data using t-SNE after PCA.
print("Performing T-Distributed Stochastic Neighbor Embedding (tSNE) on PCA ...")
X_reduced = PCA(n_components =50 ).fit_transform(X_std)
model = TSNE(learning_rate = 10, n_components = 2, random_state = 123, perplexity = 30)
tsne_pca = model.fit_transform(X_reduced)
tsne_pca_df = pd.DataFrame(data=tsne_pca, columns=['t-SNE1','t-SNE2']).join(labels)
palette = sns.color_palette("muted", n_colors=5)
sns.set_style("white")
sns.scatterplot(x='t-SNE1',y='t-SNE2',hue='Class',data=tsne_pca_df, palette=palette, linewidth=0.2, s=30, alpha=1).set_title('t-SNE after PCA')


# In[ ]:


#Measure execution time for t-SNE after PCA
def tsne_model_pca(X):
    X_reduced = PCA(n_components =50 ).fit_transform(X)
    model = TSNE(learning_rate = 10, n_components = 2, random_state = 123, perplexity = 30)
    tsne_pca = model.fit_transform(X_reduced)
    return tsne_pca
from timeit import Timer
  
t = Timer(lambda: tsne_model_pca(X_std))
print(t.timeit(number=1))


# In[ ]:


#Visualize data using UMAP.
print("Performing Uniform Manifold Approximation and Projection (UMAP) ...")
model = UMAP(n_neighbors = 30, min_dist = 0.4, n_components = 2)
umap = model.fit_transform(X_std)
umap_df = pd.DataFrame(data=umap, columns=['UMAP1','UMAP2']).join(labels)
palette = sns.color_palette("muted", n_colors=5)
sns.set_style("white")
sns.scatterplot(x='UMAP1',y='UMAP2',hue='Class',data=umap_df, palette=palette, linewidth=0.2, s=30, alpha=1).set_title('UMAP')


# In[ ]:


#Measure execution time for UMAP
def umap(X):
    X_reduced = PCA(n_components =50 ).fit_transform(X)
    model = TSNE(learning_rate = 10, n_components = 2, random_state = 123, perplexity = 30)
    tsne_pca = model.fit_transform(X)
    return tsne_pca
from timeit import Timer
  
t = Timer(lambda: tsne_model_pca(X_std))
print(t.timeit(number=1))


# In[ ]:


kmeans_labels = cluster.KMeans(n_clusters=10).fit_predict(X)


# In[ ]:


plt.scatter(umap[:, 0], umap[:, 1], c=kmeans_labels, s=15, cmap='Spectral')


# In[ ]:


dbscan_labels=cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit_predict(X_reduced)


# In[ ]:


clustered = (dbscan_labels >= 0)
plt.scatter(umap[~clustered, 0],
            umap[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=15,
            alpha=0.5)
plt.scatter(umap[clustered, 0],
            umap[clustered, 1],
            c=dbscan_labels[clustered],
            s=15,
            cmap='Spectral')


# In[ ]:


#umap enhanced clustering
clusterable_embedding = UMAP(
    n_neighbors=30,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(X_std)


# In[ ]:


dbscan_umap=cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None).fit_predict(clusterable_embedding)


# In[ ]:


clustered_umap = (dbscan_labels >= 0)
plt.scatter(umap[~clustered, 0],
            umap[~clustered, 1],
            c=(0.5, 0.5, 0.5),
            s=15,
            alpha=0.5)
plt.scatter(umap[clustered, 0],
            umap[clustered, 1],
            c=dbscan_labels[clustered],
            s=15,
            cmap='Spectral')

