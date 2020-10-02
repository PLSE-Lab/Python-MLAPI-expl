#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set()


# ## Load and clean data

# In[ ]:


stats = pd.read_csv('/kaggle/input/201819-nba-players-stats/2018-19 NBA Players Stats.csv')
stats.head()


# In[ ]:


stats.shape


# In[ ]:


stats.describe()


# In[ ]:


drop = ['PLAYER','TEAM','AGE','GP','W','L','MIN']
stats_train = stats.drop(columns=drop)
stats_train.head()


# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
stats_scaled=pd.DataFrame(scaler.fit_transform(stats_train))
stats_scaled.columns=stats_train.columns


# ## PCA

# In[ ]:


from sklearn.decomposition import PCA
n_components=stats_train.shape[1]
pca = PCA(n_components=n_components, random_state=123)
pca.fit(stats_scaled)


# In[ ]:


explained_variance_ratio = pca.explained_variance_ratio_ 
cum_explained_variance_ratio = np.cumsum(pca.explained_variance_ratio_)
lst = []
for i in range (0, n_components):
    lst.append([i+1, round(explained_variance_ratio[i],6), cum_explained_variance_ratio[i]])

pca_predictor = pd.DataFrame(lst)
pca_predictor.columns = ['Component', 'Explained Variance', 'Cumulative Explained Variance']
pca_predictor


# In[ ]:


plt.subplots(figsize=(10,8))

plt.bar(pca_predictor['Component'],pca_predictor['Explained Variance'], align='center', alpha=0.5, 
       label='individual explained variance')
plt.step(pca_predictor['Component'],pca_predictor['Cumulative Explained Variance'], where='mid',
         label='cumulative explained variance')
plt.xlabel('Principal components')
plt.ylabel('Explained variance ratio');
plt.legend(loc='best')


# In[ ]:


pca = PCA(n_components=10)
pca.fit(stats_scaled)


# In[ ]:


stats_transformed = pca.fit_transform(stats_scaled)
stats_transformed.shape


# ## K-Means

# In[ ]:


from sklearn.cluster import KMeans

n_clusters = 8
kmeans = KMeans(n_clusters=n_clusters, random_state=123)
kmeans.fit(stats_transformed)


# In[ ]:


cluster_labels = kmeans.labels_
stats['cluster'] = cluster_labels


# In[ ]:


ax=plt.subplots(figsize=(10,5))
ax=sns.countplot(cluster_labels)
title="Histogram of Cluster Counts"
ax.set_title(title, fontsize=12)
plt.show()


# ## Comparison of clusters' stats

# In[ ]:


def cluster_stats(columns):
    output = pd.DataFrame({'cluster':[ i for i in range(n_clusters)]})
    for column in columns:
        lst = []
        for i in range(n_clusters):
            mean = stats[stats['cluster'] == i].describe()[column]['mean']
            lst.append([i, round(mean,2)])
        df = pd.DataFrame(lst)
        df.columns = ['cluster', column]
        output = pd.merge(output, df, on='cluster', how='outer')
    return output


# In[ ]:


columns = stats_train.columns
cluster_stats(columns)


# In[ ]:


# starter center 
stats[stats['cluster']==4]


# In[ ]:


# starter guard 
stats[stats['cluster']==7]

