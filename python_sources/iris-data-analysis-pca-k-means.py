#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# **Import libraries to look at iris data**

# In[ ]:


import numpy as np #linear algebra
import pandas as pd #data preprocessing, csv file I/O (e.g. pd.read_csv)
import seaborn as sns
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# loading the data

# In[ ]:


data = pd.read_csv("../input/Iris.csv", index_col = 0)
data.head(5)


# Different statisstics by species

# In[ ]:


print("Iris-setosa")
data1 = data[data["Species"] == "Iris-setosa"]
data1.describe()


# In[ ]:


print("Iris-versicolor")
data2 = data[data["Species"] == "Iris-versicolor"]
data2.describe()


# In[ ]:


print("Iris-virginica")
data3 = data[data["Species"] == "Iris-virginica"]
data3.describe()


# **Correlation between variables**

# In[ ]:


X = data.iloc[:, 0:4]
f, ax = plt.subplots(figsize=(10, 8))
corr = X.corr()
print(corr)
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), 
            cmap=sns.diverging_palette(220, 10, as_cmap=True), square=True, ax=ax)


# **Kernel density estimations**

# In[ ]:


sns.pairplot(data, hue = "Species", size=3, diag_kind="kde")


# **Principal Component Analysis**

# > PCA is a method which allows you to identify the "Directions" in which the most of the variations in the data are present

# It is a statistical procedure that uses an orthogonal transformation to convert set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables

# In[ ]:


PCA = PCA(n_components = 4)
PCA.fit(X)
print("explained variance ratio: %s"
      % str(PCA.explained_variance_ratio_))
plt.plot(np.r_[[0], np.cumsum(PCA.explained_variance_ratio_)])
plt.xlim(0,5)
plt.xlabel("number of components")
plt.ylabel("cumulative explained variance");


# **The 98% of the variance is expalined, approxmately, by the first two components.
# 
# **Lets see the species represented by this components! **

# In[ ]:


y = data["Species"]
y = y.map({"Iris-setosa" : 0, "Iris-virginica" : 1, "Iris-versicolor" : 2})

X_r = PCA.transform(X)
plt.scatter(X_r[y == 0, 0], X_r[y == 0, 1], alpha=.8, color = 'blue',
              label="Iris-setosa")
plt.scatter(X_r[y == 1, 0], X_r[y == 1, 1], alpha=.8, color = "green",
              label = "Iris-virginica")
plt.scatter(X_r[y == 2, 0], X_r[y == 2, 1], alpha =.8, color = 'orange', 
              label = "Iris-versicolor")
plt.legend(loc='best', shadow=False, scatterpoints=1)
plt.xlabel("First component")
plt.ylabel("Second component")
plt.title("PCA of IRIS datset")


# **What do the components mean? Let's see the correlation between variables and components**

# In[ ]:


components = pd.DataFrame(PCA.components_, columns = X.columns, index=[1, 2, 3, 4])
components


# *The variables PetalLengthCm is highly correlated with the first components(92.4% variance)*

# **K-Means**

# *The K-means unsupervised algorithm involves randomly selecting K initial centroids where K is a user defined number of desired clusters.
# 
# The problem is computationally difficult (NP-hard); however, there are efficient heuristic algorithms that are commonly employed and converge quickly to a local optimum.
# 
# We apply differenet algorithms chaging the parameters number of clusters (n) and random_seed (s). We save the best models, taking those that minimize the objective function.*

# In[ ]:


list_nclusters = range(2, 10)
list_init_seed = range(1, 100)
total = len(list_nclusters)
inertia = np.zeros(total)
ncluster = np.zeros(total)
nseed = np.zeros(total)
inertia_aux = np.inf
count = -1

for n in list_nclusters:
    count = count + 1
    for s in list_init_seed:
        cl = KMeans(n_clusters = n, random_state =s)
        cl.fit(X_r[:, 0:2])
        if  (cl.inertia_ < inertia_aux) :
            inertia[count] = cl.inertia_
            inertia_aux = cl.inertia_
            ncluster[count] = n
            nseed[count] = s


# let's see the best models!!!

# In[ ]:


matrix = np.matrix(np.c_[inertia, ncluster, nseed])
models = pd.DataFrame(data = matrix, columns = ['Inertia', 'Number of clusters', 'random seed'])
models


# **now , we're going to select the number of clusters, let's see the objective function best value!**

# In[ ]:


models.plot(kind = 'scatter', x = 1, y=0)


# Three and four clusters would be good.
# 
# we create different plots in order to explore the clusters

# In[ ]:


c1 = KMeans(n_clusters = 3, random_state = 1)
c1.fit(X_r[:, 0:2])
cluster = cl.predict(X_r[:, 0:2])
plt.scatter(X_r[cluster ==0, 0], X_r[cluster == 0, 1], alpha =.8, color = 'orange', label = "Cluster 1")
plt.scatter(X_r[cluster ==1, 0], X_r[cluster == 1, 1], alpha =.8, color = 'blue', label = "Cluster 2")
plt.scatter(X_r[cluster ==2, 0], X_r[cluster == 2, 1], alpha =.8, color = 'green', label = "Cluster 3")
plt.legend(loc="best", shadow=False, scatterpoints =1)
plt.xlabel("First component")
plt.ylabel('second component')
plt.title("PCA of IRIS dataset")


# Kernal density with three clusters

# In[ ]:


c = pd.DataFrame(cluster, columns = ['Cluster'], index = range(1, len(cluster) + 1))
c.head(n=5)
c["Cluster"] = c["Cluster"].map({0: 'Cluster 1', 1:'Cluster 2', 2: "Cluster 3"})
sns.pairplot(pd.concat([data, c], axis = 1), hue = "Cluster", size=3, diag_kind= "kde")


# > The sam eplots with four clusters!

# In[ ]:


c1 = KMeans(n_clusters= 4, random_state =2)
c1.fit(X_r[:, 0:2])
cluster = cl.predict(X_r[:, 0:2])
plt.scatter(X_r[cluster == 0, 0], X_r[cluster ==0, 1], alpha =.8, color = 'orange', label ="cluster 1")
plt.scatter(X_r[cluster == 1, 0], X_r[cluster ==1, 1], alpha =.8, color = 'blue', label ="cluster 2")
plt.scatter(X_r[cluster == 2, 0], X_r[cluster ==2, 1], alpha =.8, color = 'red', label ="cluster 3")
plt.scatter(X_r[cluster == 3, 0], X_r[cluster ==3, 1], alpha =.8, color = 'green', label ="cluster 4")
plt.legend(loc='best', shadow = False, scatterpoints=1)
plt.xlabel("First component")
plt.ylabel("Second component")
plt.title("PCA of IRIS dataset")


# In[ ]:


c = pd.DataFrame(cluster, columns = ['Cluster'], index = range(1, len(cluster) + 1))
c.head(n=5)
c['Cluster'] = c['Cluster'].map({0: 'Cluster 1', 1:'Cluster 2', 2:'Cluster 3', 3:'Cluster 4'})
sns.pairplot(pd.concat([data, c], axis = 1), hue = "Cluster", size=3, diag_kind="kde")

