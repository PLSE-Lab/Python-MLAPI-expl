#!/usr/bin/env python
# coding: utf-8

# # Patterns in football (soccer)
# ## Objective
# Football positions are the basis of different specialized roles which players have to play. For example, a defender must have good positioning in order to quench attacks. A striker needs to be accurate in front of goal, since scoring opportunities are scarce. A playmaker needs to have good vision, and the ability to break the opposition defense with a pass.
# 
# The dataset we will be using has all of these statistics, according to EA studios, ranked from 1 to 99. Most european top league players are included, as well as players from leagues around the world. We want to see if, by applying principal component analysis (PCA) and hierarchichal clustering algorithms, this distinction in roles and abilities is also present in the FIFA 19 statistics.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

print(os.listdir('../input'))


# In[ ]:


# We bring the functions necessary for PCA analysis
from sklearn.decomposition import PCA


# In[ ]:


# Full dataset
data = pd.read_csv('../input/data.csv')
data.head()


# We have *lots* of parameters! The variables we are interested in are in the list called `variables`. They are a mix of the technical qualities each player has, and the physical qualities, such as pace, jumping, etc. that also have an impact on each role and position.

# In[ ]:


# Parameter list
variables = ['Crossing', 'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle', 'GKDiving', 'GKHandling', 'GKKicking', 'GKPositioning', 'GKReflexes']
#variables = ['Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling', 'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration', 'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower', 'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression', 'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure', 'Marking', 'StandingTackle', 'SlidingTackle']

df = data[variables]
df = df.dropna(how='all') # We must drop all NA values to apply PCA
df = df.fillna(df.mean()) # We fill these values with the mean values


# In[ ]:


# Data visualization
df.head()


# **Data transformation:** By standardizing our data, we may be able to improve the convergence of the hierarchical clustering algorithms which we will use.

# In[ ]:


# In order to improve convergence, we standardize our data

stand_df = StandardScaler().fit_transform(df.loc[:,variables].values)


# Now, we can begin with PCA analysis!

# In[ ]:


# Applying PCA analysis
pca = PCA(n_components=16)
pcafit = pca.fit(stand_df)
pcafeatures = pca.transform(stand_df)

features = range(pca.n_components_)

num_comp = 5
var_percent = sum(pca.explained_variance_ratio_[0:num_comp])
print("Total explained variance by the first %i components is %.5f" % (num_comp, var_percent))


# In[ ]:


# Percentage of variance explained per component
plt.bar(features, pca.explained_variance_ratio_)
plt.xlabel('Component number')
plt.ylabel('Percentage of variance')
plt.show()


# And now lets do a preliminary visualization

# In[ ]:


principalComponents = pca.fit_transform(stand_df)

principalDf = pd.DataFrame(data = principalComponents)
principalDf = principalDf[[0, 1, 2, 3, 4]]
principalDf.columns = [['PC1', 'PC2', 'PC3', 'PC4', 'PC5']]
principalDf.head()


# Lets check out if there are any clusters

# In[ ]:


g = sns.PairGrid(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']])
g.map_diag(plt.hist, histtype="step", linewidth=2)
g.map_offdiag(plt.scatter)

plt.show()


# We can see that the first principal component has what at first glance appears to be two clusters. This probably is because we included goalkeepers, which have much lower physical and technical attributes compared to outfield players. Plus, their goalkeeping stats are much better than all other positions.
# 
# ---

# # Hierarchical clustering algorithms applied
# 
# In order to get a sense of how many clusters we should have, we now apply our two hierarchical clustering algorithms. We begin with the functions from `scipy.cluster.hierarchy`, which contain clustering algorithms that separate these groups according to multiple criteria.
# 
# The two criteria which we are interested in are *single* and *complete*. 
# 
# ## Single method

# In[ ]:


from sklearn import preprocessing
from scipy.cluster.hierarchy import dendrogram, linkage


# In[ ]:


import sys
sys.setrecursionlimit(10000)

dist_sin = linkage(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],method='single')
plt.figure(figsize=(20,8))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("Dendogram: single method",fontsize=20)
plt.show()


# We see that this dendogram is *really* large, so we will truncate it to the first 100 clusters:

# In[ ]:


dist_sin = linkage(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],method='single')
plt.figure(figsize=(20,8))
dendrogram(dist_sin, leaf_rotation=90, p=100, truncate_mode='lastp')
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("Dendogram: truncated single method",fontsize=20)
plt.show()


# ## Complete method
# We can now apply the complete method of the clustering function and compare our results:

# In[ ]:


dist_sin = linkage(principalDf.loc[:,['PC1', 'PC2', 'PC3', 'PC4', 'PC5']],method='complete')
plt.figure(figsize=(20,8))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("Dendogram: complete method",fontsize=20)
plt.show()


# This dendogram makes much more sense, this shows us the importance of applying different clustering methods in order to get the best possible grouping for our data.

# In[ ]:




