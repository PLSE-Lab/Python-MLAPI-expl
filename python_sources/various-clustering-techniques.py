#!/usr/bin/env python
# coding: utf-8

# We will use various clustering techniques on the world happiness data. Below are the various clustering techniques we will use.
# 
# 1.KMeans
# 
# 2.MeanShift
# 
# 3.MiniBatchKMeans
# 
# 4.SpectralClustering
# 
# 5.DBSCAN
# 
# 6.Affinity Propagation
# 
# 7.Birch
# 
# 8.GaussianMixture

# In[ ]:


get_ipython().run_line_magic('reset', '-f')
import time                   # To time processes
import warnings               # To suppress warnings

import numpy as np            # Data manipulation
import pandas as pd           # Dataframe manipulatio 
import matplotlib.pyplot as plt                   # For graphics

from sklearn import cluster, mixture              # For clustering
from sklearn.preprocessing import StandardScaler  # For scaling dataset

import os                     # For os related operations
import sys 

from sklearn import metrics

import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
# %matplotlib inline
warnings.filterwarnings('ignore')

import seaborn as sns


# In[ ]:


X= pd.read_csv("../input/2017.csv", header = 0)


# In[ ]:


X.columns.values


# In[ ]:


X.shape


# In[ ]:


X.dtypes


# In[ ]:


X_copy = X
X = X.iloc[:, 2: ]


# In[ ]:


## Function for various clusters
def compute_cluster (clType,df= X):
    if clType=='KMeans':
        result = cluster.KMeans(n_clusters= 2).fit_predict(df) 
    elif clType == 'MeanShift' :
        result = cluster.MeanShift(bandwidth=0.2).fit_predict(df)    
    elif clType == 'MiniBatchKMeans':
        result = cluster.MiniBatchKMeans(n_clusters=2).fit_predict(df)        
    elif clType == 'SpectralClustering':
        result = cluster.SpectralClustering(n_clusters=2).fit_predict(df)
    elif clType == 'DBSCAN':
        result = cluster.DBSCAN(eps=0.3).fit_predict(df)
    elif clType == 'Affinity Propagation':
        result = cluster.AffinityPropagation(damping=0.9, preference=-200).fit_predict(df)
    elif clType == 'Birch':
        result = cluster.Birch(n_clusters= 2).fit_predict(df)
    elif clType == 'GaussianMixture' :
        gmm = mixture.GaussianMixture( n_components=2, covariance_type='full')
        gmm.fit(df)
        result = gmm.predict(df)  
    else:
        print("exit")
    
    cl_df.loc[cl_df.Name == clType, 'Silhouette-Coeff'] = metrics.silhouette_score(df, result, metric='euclidean')
    cl_df.loc[cl_df.Name == clType, 'Calinski-Harabaz'] = metrics.calinski_harabaz_score(df, result)
    
    return result


# Standardization of a dataset is a common requirement for many machine learning estimators: they might behave badly if the individual feature do not more or less look like standard normally distributed data (e.g. Gaussian with 0 mean and unit variance).

# In[ ]:


ss = StandardScaler().fit_transform(X)


# In[ ]:


cl_dist = {'Name' : ['KMeans','MeanShift','MiniBatchKMeans','SpectralClustering','DBSCAN','Affinity Propagation','Birch','GaussianMixture']}
cl_df = pd.DataFrame(cl_dist)
cl=pd.Series(['KMeans','MeanShift','MiniBatchKMeans','SpectralClustering','DBSCAN','Affinity Propagation','Birch','GaussianMixture'])


# In[ ]:


X.head(5)


# In[ ]:


for i in range(0,cl.size) :
    result = compute_cluster(clType=cl[i])
    X[cl[i]] = pd.DataFrame(result)


# In[ ]:


cl_df


# The **Silhouette Coefficient** (sklearn.metrics.silhouette_score) is an example of such an evaluation, where a higher Silhouette Coefficient score relates to a model with better defined clusters. The Silhouette Coefficient is defined for each sample and is composed of two scores:
# 
# a: The mean distance between a sample and all other points in the same class.
# b: The mean distance between a sample and all other points in the next nearest cluster.
# The Silhouette Coefficient s for a single sample is then given as:
# s = {b - a}/{max(a, b)}
# 
# The score is bounded between -1 for incorrect clustering and +1 for highly dense clustering. Scores around zero indicate overlapping clusters.
# The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.
# 
# the **Calinski-Harabaz index** (sklearn.metrics.calinski_harabaz_score) can be used to evaluate the model, where a higher Calinski-Harabaz score relates to a model with better defined clusters.
# 
# For k clusters, the Calinski-Harabaz score s is given as the ratio of the between-clusters dispersion mean and the within-cluster dispersion.
# The score is higher when clusters are dense and well separated, which relates to a standard concept of a cluster.

# In[ ]:


rows = 4    # No of rows for the plot
cols = 2    # No of columns for the plot

# 4 X 2 plot
fig,ax = plt.subplots(rows,cols, figsize=(10, 10)) 
x = 0
y = 0
for i in cl:
    ax[x,y].scatter(X.iloc[:, 6], X.iloc[:, 5],  c=X.iloc[:, 12+(x*y)])
    ax[x,y].set_title(i + " Cluster Result")
    y = y + 1
    if y == cols:
        x = x + 1
        y = 0
        plt.subplots_adjust(bottom=-0.5, top=1.5)
plt.show()

