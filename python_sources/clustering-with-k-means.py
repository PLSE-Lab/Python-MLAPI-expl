#!/usr/bin/env python
# coding: utf-8

# # Clustering with K-Means

# ## Data
# 
# #### - Features :
# 
# - A: Area
# - P: Perimeter
# - C: Compactness
# - LK: Length of Kernel
# - WK: Width of Kernel
# - A_Coef: Asymmetry Coefficient
# - LKG: Length of Kernel Groove
# 
# #### - Target :
# 
# - Target class : 0, 1, 2

# ## Importing libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.metrics import classification_report


# ## Loading the data

# In[ ]:


seed = pd.read_csv('../input/Seed_Data.csv')


# ## EDA (Exploratory Data Analysis)

# In[ ]:


seed.shape


# In[ ]:


seed.head()


# In[ ]:


seed.describe()


# In[ ]:


seed.info()


# We can check that there is no null data.

# ## Data Visualization

# In[ ]:


sns.pairplot(seed.drop('target', axis=1))


# In[ ]:


sns.lmplot(x='A', y='C', data=seed, hue='target', aspect=1.5, fit_reg=False)


# In[ ]:


sns.lmplot(x='A', y='A_Coef', data=seed, hue='target', aspect=1.5, fit_reg=False)


# In[ ]:


sns.lmplot(x='LK', y='C', data=seed, hue='target', aspect=1.5, fit_reg=False)


# ## Elbow Curve to check elbow point

# In[ ]:


dist = {}

for k in range(1, 10):
    km = KMeans(n_clusters=k).fit(seed.drop('target', axis=1))
    
    dist[k] = km.inertia_  # Sum of squared distances of samples to their closest cluster center.

dist


# In[ ]:


plt.plot(list(dist.keys()), list(dist.values()), marker='H', markersize=10)
plt.xlabel('Number of k')
plt.ylabel('Distances')
plt.title('Elbow Curve')


# Elbow point is 3. Now, we can decide number of clusters should be 3.

# ## Fitting model with sklearn

# In[ ]:


km = KMeans(n_clusters=3, random_state=42)


# In[ ]:


km_pred = km.fit_predict(seed.drop('target', axis=1))
km_pred


# In[ ]:


np.array(seed['target'])


# In[ ]:


# Change the labels of prediction

for i in range(len(km_pred)):
    if km_pred[i] == 1:
        km_pred[i] = 0
    elif km_pred[i] == 0:
        km_pred[i] = 1

km_pred


# In[ ]:


print('Accuracy of clustering : ' + str(round(sum(km_pred == seed['target']) / seed.shape[0] * 100, 2)) + '%')


# In[ ]:


print(classification_report(seed['target'], km_pred, target_names=['1', '2', '3'], digits=4))


# ## Showing the centroids

# In[ ]:


centers = km.cluster_centers_
centers


# In[ ]:


seed['klabels'] = km.labels_
seed.head()


# In[ ]:


seed.klabels.value_counts()


# In[ ]:


seed.target.value_counts()


# In[ ]:


f, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 8))

ax1.set_title('K-Means (k=3)')
ax1.scatter(x=seed['A'], y=seed['A_Coef'], c=seed['klabels'])
ax1.scatter(x=centers[:, 0], y=centers[:, 5], c='r', s=300, alpha=0.5)

ax2.set_title('Original')
ax2.scatter(x=seed['A'], y=seed['A_Coef'], c=seed['target'])

