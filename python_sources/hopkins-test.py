#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.preprocessing import scale
from pyclustertend import hopkins ## the hopkins test


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt



heart_df = pd.read_csv("../input/heart.csv")


# In this Kernel, I would like to show you an exemple of cluster tendency test. The aim of cluster tendency is to test if clustering is relevant in our dataset.
# A well know test for cluster tendency is the Hopkins Test. It check if observations are randomly distributed in the space or not.
# 
# First, let's quickly look at our data table.

# In[ ]:


X = heart_df[heart_df.columns[~heart_df.columns.isin(["target"])]].values
y = heart_df[heart_df.columns[heart_df.columns.isin(["target"])]].values.flatten()


# In[ ]:


hopkins(X, X.shape[0])


# the null hypothesis (no meaningfull cluster) happens when the hopkins test is around 0.5 and the hopkins test tends to 0 when meaningful cluster exists in the space. Usually, we can believe in the existence of clusters when the hopkins score is bellow 0.25.
# 
# Here the value of the hopkins test is quite high but one could think there is cluster in our subspace. **BUT** the hopkins test is highly influenced by outliers, let's try once again with normalised data. 

# In[ ]:


hopkins(scale(X),X.shape[0])


# Here, we can see that the hopkins score is much more higher with normalised data.

# ### Confirmation : 
# 
# We have made some hypothesis during this test : 
# - there is no cluster in our dataset. 
# 
# Now, let's check if they are true using a PCA. 

# In[ ]:


pca = PCA(n_components = 2)

X_pca = pca.fit_transform(scale(X))

labels = heart_df.target.values

cdict = {0 : "green", 1 : "red"}
labl = {0: "healthy", 1 : "sick"}
marker = {0 : "o", 1: "*"}
alpha = {0: 0.5, 1: 0.5}

#fig = plt.figure(figsize=(10, 10))
#ax = fig.add_subplot(111, projection='3d')

fig = plt.figure(figsize=(10, 10))


for l in np.unique(labels):
    ix = np.where(labels == l)
    plt.scatter(X_pca[ix,0],X_pca[ix,1], c = cdict[l], s=40, label = labl[l], marker = marker[l], alpha = alpha[l])

#plt.scatter(X_pca[:,0],X_pca[:,1]);
plt.xlabel("first principal component")
plt.xlabel("second principal component")
plt.title("PCA  : heart diseases")


# As we can see, our hypothesis seems true.
