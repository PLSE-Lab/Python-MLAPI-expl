#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/ccdata/CC GENERAL.csv')
data.head()


# # Data Preprocessing

# In[ ]:


data.shape


# In[ ]:


data.describe()


# #### As you can see, There are many outliers. But, we can't simply drop the outliers as they may contain useful information. So, we'll treat them as extreme values

# In[ ]:


data.isnull().sum()


# #### Data contains some null Values. 
# 
# #### Treating them by substituting with mean.

# In[ ]:


data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean(),inplace=True)
data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean(),inplace=True)


# In[ ]:


data.isnull().sum()


# In[ ]:


data.columns


# #### Dropping the 'CUST_ID' columns as it is not needed.

# In[ ]:


data.drop('CUST_ID',axis=1,inplace=True)


# In[ ]:


data.columns


# # Scaling the data
# 
# We scale the data because it helps to normalise the data within a particular range and every feature transforms to a common scale.

# In[ ]:


from scipy.stats import zscore


# In[ ]:


data_scaled=data.apply(zscore)
data_scaled.head()


# # Clustering

# ### K means

# In[ ]:


cluster_range = range(1,15)
cluster_errors=[]
for i in cluster_range:
    clusters=KMeans(i)
    clusters.fit(data_scaled)
    labels=clusters.labels_
    centroids=clusters.cluster_centers_,3
    cluster_errors.append(clusters.inertia_)
clusters_df=pd.DataFrame({'num_clusters':cluster_range,'cluster_errors':cluster_errors})
clusters_df


# In[ ]:


f,ax=plt.subplots(figsize=(15,6))
plt.plot(clusters_df.num_clusters,clusters_df.cluster_errors,marker='o')
plt.show()


# #### Choosing k=4 as after k=4 , the cluster errors are almost similar and also, the slope of the line is almot constant as well.

# In[ ]:


kmean= KMeans(4)
kmean.fit(data_scaled)
labels=kmean.labels_


# In[ ]:


clusters=pd.concat([data, pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()


# ## Interpretation of clusters

# In[ ]:


for c in clusters:
    grid= sns.FacetGrid(clusters, col='cluster')
    grid.map(plt.hist, c)


# In[ ]:


clusters.groupby('cluster').mean()


# #### Cluster 0 : Youth / New Working professionals
# Low balance but the balance gets updated frequently ie. more no. of transactions. No of purchases from the account are also quite large and majority of the purchases are done either in one go or in installments but not by paying cash in advance.
# 
# #### Cluster 1 : Retired Professionals/Pensioners
# Comparitively high balance but the balance does not get updated frequently ie. less no. of transactions. No. of purchases from account are quite low and very low purchases in one go or in installments. Majority of purchases being done by paying cash in advance. Purchase frequency is also quite low.
# 
# #### Cluster 2 : Industrialist
# Balance is very high and it gets updated very frequently as well. No. of purchases are comparitively less and almost all the purchases are done with cash in advance. Purchase frequency is also quite low.
# 
# #### Cluster 3 : High level Businessmen
# Balance is very high and it gets updated very frequently as well. no. of purchases are extremely high and majority of their purchases are done either in one-go or in installments. Purchase frequency also very high indicating purchasing happening at high frequency. Also, these have the highest credit limit.

# ### Applying PCA
# 
# #### We apply PCA to transform data to 2 dimensions for visualization. We won't be able to visualize the data in 17 dimensions so reducing the dimensions with PCA.
# 
# #### PCA transforms a large set of variables into a smaller one that still contains most of the information in the large set.Reducing the number of variables of a data.
# 

# In[ ]:


pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data_scaled)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf.head(2)


# In[ ]:


finalDf = pd.concat([principalDf, pd.DataFrame({'cluster':labels})], axis = 1)
finalDf.head()


# In[ ]:



plt.figure(figsize=(15,10))
ax = sns.scatterplot(x="principal component 1", y="principal component 2", hue="cluster", data=finalDf,palette=['red','blue','green','yellow'])
plt.show()


# ### Agglomerative / Hierarchichal Clustering

# In[ ]:


data_scaled.head()


# In[ ]:


from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram,linkage


# In[ ]:


Z=linkage(data_scaled,method="ward")


# In[ ]:


plt.figure(figsize=(15,10))
dendrogram(Z,leaf_rotation=90,p=5,color_threshold=20,leaf_font_size=10,truncate_mode='level')
plt.axhline(y=125, color='r', linestyle='--')
plt.show()


# #### If we draw a horizontal line that passes through longest distance without a horizontal line, It intersects 4 vertical lines
# 
# #### So, Optimal cluster = 4

# In[ ]:


model=AgglomerativeClustering(n_clusters=4,affinity='euclidean',linkage='ward')


# In[ ]:


model.fit(data_scaled)


# In[ ]:


model.labels_


# In[ ]:


clusters_agg=pd.concat([data, pd.DataFrame({'cluster':model.labels_})], axis=1)
clusters_agg.head()


# In[ ]:


clusters_agg.groupby('cluster').mean()


# #### Almost similar result as kmeans clustering
# 
# #### Here, Cluster 1 and 2 remains the same but, cluster 0 and 3 are interchanged.

# ### PCA

# In[ ]:


finalDf_agg = pd.concat([principalDf, pd.DataFrame({'cluster':model.labels_})], axis = 1)
finalDf_agg.head()


# In[ ]:


plt.figure(figsize=(15,10))
ax = sns.scatterplot(x="principal component 1", y="principal component 2", hue="cluster", data=finalDf_agg,palette=['red','blue','green','yellow'])
plt.show()


# ### Same clusters as Kmeans

# In[ ]:




