#!/usr/bin/env python
# coding: utf-8

# In[543]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from time import time
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.image import imread
import seaborn as sns
from sklearn.datasets.samples_generator import (make_blobs,
                                                make_circles,
                                                make_moons)
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn import metrics
import matplotlib as mpl
import itertools
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[544]:


get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_context('notebook')
plt.style.use('fivethirtyeight')
from warnings import filterwarnings
filterwarnings('ignore')


# ## 1. Load data 

# In[545]:


iris_data = pd.read_csv("../input/iriscsv/iris.arff.csv")
abalone_data = pd.read_csv("../input/abalone/abalone.csv")


# In[546]:


# Plot the iris data 'sepallength' and 'petalwidth' 
plt.figure(figsize=(6, 6))
iris_labels = iris_data['class']
for i ,label in enumerate(pd.unique(iris_labels)):
    data_plot =iris_data[iris_labels==label]
    plt.scatter(data_plot.iloc[:, 0], data_plot.iloc[:, 3])
plt.xlabel('sepallength')
plt.ylabel('petalwidth')
plt.title('Visualization of raw data');


# ## Visualization and Evaluation

# In[547]:


def plot_clustering(X, label_pred, title=None,centroids=None,xlabel=None,ylabel=None):
    plt.figure(figsize=(6, 4))
    colormap = mpl.cm.Dark2.colors
    for i ,color in zip(range(len(pd.unique(label_pred))),itertools.product(colormap)):
        x = X[label_pred == i]
        plt.scatter(x.iloc[:, 0], x.iloc[:, 1], c = color, marker='.', label='label0')           
    if centroids is not None:
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='*', s=300,c='y', label='centroid')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    plt.xticks([])
    plt.yticks([])
    if title is not None:
        plt.title(title, size=17)
    plt.axis('off')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[548]:


def self_evaluate(X,labels_true,labels):
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    print("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
    print("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
    print("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))


# ## 3. Train and Evaluate

# ### (1) Iris Data 

# In[549]:


# build cluster mode
def self_kmeans(X,label_true,attri=None,clusters_num=3):
    kmeans = KMeans(n_clusters=clusters_num).fit(X)
    centroids = kmeans.cluster_centers_
    
    self_evaluate(X,label_true,kmeans.labels_)
    X_plot = X[attri]
    if attri is not None and len(attri)==2:
        plot_clustering(X_plot,kmeans.labels_,"k-means",kmeans.cluster_centers_,attri[0],attri[1])
    else:
        plot_clustering(X_plot,kmeans.labels_,'k-means',kmeans.cluster_centers_)
    plt.show()
def self_Agglomerative(X,label_true,attri=None,clusters_num=3):
    for linkage in ('ward', 'average', 'complete', 'single'):
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=clusters_num)
        t0 = time()
        clustering.fit(X)
        #print("%s :\t%.2fs" % (linkage, time() - t0))
        self_evaluate(X,label_true,clustering.labels_)
        X_plot = X[attri]
        if attri is not None and len(attri)==2:
            plot_clustering(X_plot, clustering.labels_, "%s linkage" % linkage,xlabel=attri[0],ylabel=attri[1])
        else:
            plot_clustering(X_plot, clustering.labels_, "%s linkage" % linkage)
        plt.show()
def self_DBSCAN(X,label_true,attri=None,min_samples=3,eps=1.0):
    db = DBSCAN(min_samples=min_samples,eps=eps).fit(X)
    self_evaluate(X,label_true,db.labels_)
    X_plot = X[attri]
    if attri is not None and len(attri)==2:
        plot_clustering(X_plot,db.labels_,'DBSCAN',xlabel=attri[0],ylabel=attri[1])
    else:
        plot_clustering(X_plot,db.labels_,'DBSCAN')
    plt.show()


# In[550]:


# select 'sepallength' and 'petalwidth'
class_mapping={}
classes = iris_data["class"].unique()
for i,thisclass in zip(range(len(classes)),classes):
    class_mapping[thisclass]=i
classes_label = iris_data['class'].map(class_mapping)

X_columns = iris_data.columns.drop("class")
X_iris = iris_data[X_columns]
attributes = ['sepallength','petalwidth']
self_kmeans(X_iris,classes_label,attributes,3)


# In[551]:


self_Agglomerative(X_iris,classes_label,attri=attributes,clusters_num=3)


# In[552]:


self_DBSCAN(X_iris,classes_label,attri=attributes,min_samples=5,eps=0.65)


# ## Abalone data

# In[553]:


class_mapping={}
classes = abalone_data["class"].unique()
for i,thisclass in zip(range(len(classes)),classes):
    class_mapping[thisclass]=i
classes_label = abalone_data['class'].map(class_mapping)

X_columns = abalone_data.columns.drop(["class","Sex"])
X_abalone = abalone_data[X_columns]
attributes = ["Height",'Length']
self_kmeans(X_abalone,classes_label,attributes,clusters_num=5)


# In[554]:


self_Agglomerative(X_abalone,classes_label,attri=attributes,clusters_num=20)


# In[557]:


self_DBSCAN(X_abalone,classes_label,attri=attributes,min_samples=5,eps=0.1)

