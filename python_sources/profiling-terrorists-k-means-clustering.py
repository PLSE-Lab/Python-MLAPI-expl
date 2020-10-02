#!/usr/bin/env python
# coding: utf-8

# ## Profiling Terrorists: K-means Clustering
# By: Ian Chu Te
# 
# In this notebook, we identify six key geographic-behavioral profiles of terrorists based on data from the *Global Terrorism Database*. We cluster the terrorist attacks using <a href="https://en.wikipedia.org/wiki/K-means_clustering">K-means Clustering</a> on the geographic location of the attack, number of victims, nationality of the perpetrator, target type, weapon type and attack type.
# 
# Geographic location plays a very significant role in the clustering step - 83% of cluster labels coinside with the original region/continent labels. However, the region alone does not determine the cluster boundaries as some regions fit the profile of neighboring regions much better than their official region.
# 
# After the clustering step, we show the characteristics of each geographic-behavioral profile and point out what sets them apart from the others.

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import silhouette_score
from sklearn.preprocessing import scale, robust_scale

import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('coolwarm')
sns.set_color_codes('bright')


# ### 1. Feature Engineering
# 
# We begin by loading the Global Terrorism Database dataset.

# In[ ]:


data = pd.read_csv('../input/globalterrorismdb_0617dist.csv', encoding='latin1')


# We remove massive attacks as they could skew our clusters. We could tackle massive attacks in a separate case study.

# In[ ]:


# outlier removal - remove massive terrorist attacks
data = data[data['nkill'] <= 4].reset_index(drop=True)
data = data[data['nwound'] <= 7].reset_index(drop=True)


# Afterwards, we look for variables with high enough variance that might help us identify our clusters.

# In[ ]:


c = data.count().sort_values().drop([
    'eventid', 'country', 'iyear', 'natlty1', 'longitude', 'latitude', 'targsubtype1'])
_ = data[c[c > 100000].keys()].var().sort_values().plot.barh()


# In[ ]:


features = [
    'longitude',
    'latitude',
    
    'nwound',
    'nkill',
    
    'natlty1_txt',
    'targtype1_txt',
    'targsubtype1_txt',
    'weaptype1_txt',
    'attacktype1_txt',
]

X = pd.get_dummies(data[features])
X = X.T[X.var() > 0.05].T.fillna(0)
X = X.fillna(0)

print('Shape:', X.shape)
X.head()


# ### 2. K-Means Clustering
# 
# We now do K-means Clustering on our identified variables of interest. Firstly, we find the optimal *k* via the <a**** href="https://en.wikipedia.org/wiki/Elbow_method_(clustering)">elbow method</a> which we find to be equal to six.

# In[ ]:


scores = {}
for k in range(2, 11):
    print(k, end=', ')
    scores[k] = KMeans(n_clusters=k).fit(X).score(X)
_ = pd.Series(scores).plot.bar()


# We then get the cluster labels and compute the silhouette score of our clustering. A score of ~60% would suffice in our case.

# In[ ]:


data['Cluster'] = KMeans(n_clusters=6).fit_predict(X) + 1
print('Silhouette Score:', silhouette_score(X, data['Cluster'], sample_size=10000) * 10000 // 1 / 100, '%')


# ### 3. Profiling
# 
# Now, we profile each cluster. We name each cluster by each dominant region lying within it. Then we look for interesting mean values that stand out in our heatmap (after scaling the variables).

# In[ ]:


names = data.groupby('Cluster')['region_txt'].describe()['top'].values
data['ClusterName'] = data['Cluster'].apply(lambda c: names[c - 1])

numerical = data.dtypes[data.dtypes != 'object'].keys()
exclude = [
    'eventid', 'Cluster', 'region', 'country', 'iyear', 
    'natlty1', 'natlty2', 'natlty3', 'imonth', 'iday',
    'guncertain1', 'guncertain2', 'guncertain3'
] + [col for col in numerical if 'type' in col or 'mode' in col or 'ransom' in col]
X_profiling = data[numerical.drop(exclude)].fillna(0)
X_profiling = pd.DataFrame(scale(X_profiling), columns=X_profiling.columns)
X_profiling['ClusterName'] = data['ClusterName']
_ = sns.heatmap(X_profiling.groupby('ClusterName').mean().drop(['longitude', 'latitude'], axis=1).T, 
               cmap='coolwarm')


# In[ ]:


ckeys = data['ClusterName'].unique()
ckeys = dict(zip(ckeys, plt.cm.tab10(range(len(ckeys)))))

for i, x in X_profiling.groupby('ClusterName'):
    _ = plt.scatter(x['longitude'], x['latitude'], c=ckeys[i], marker='.', cmap='tab10', label=i)
_ = plt.legend(loc=3)


# In[ ]:


ckeys = data['region_txt'].unique()
ckeys = dict(zip(ckeys, plt.cm.tab20(range(len(ckeys)))))

for i, x in pd.concat([X_profiling, data['region_txt']], axis=1).groupby('region_txt'):
    _ = plt.scatter(x['longitude'], x['latitude'], c=ckeys[i], marker='.', cmap='tab10', label=i)
_ = plt.legend(loc=3)


# In[ ]:


print('Similarity between cluster and region labels:', 
      len(data[data['region_txt'] == data['ClusterName']]) / len(data) * 10000 // 1 / 100, '%')


# In[ ]:


d = pd.get_dummies(data['attacktype1_txt'])
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


d = pd.get_dummies(data['targtype1_txt'])
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


top = data['targsubtype1_txt'].value_counts().head(20).keys().tolist()
d = pd.get_dummies(data['targsubtype1_txt'].apply(lambda x: x if x in top else None))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


d = pd.get_dummies(data['weaptype1_txt'])
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


top_natls = data['natlty1_txt'].value_counts().head(20).keys()
natl = data['natlty1_txt'].apply(lambda x: x if x in top_natls else None)
d = pd.get_dummies(natl)
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


months = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
d = pd.get_dummies(data['imonth'].apply(lambda x: None if x == 0 else months[int(x)]))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


d = pd.get_dummies(data['iday'].apply(lambda x: None if x == 0 else int(x)))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


d = pd.get_dummies(data['nhours'].apply(lambda x: None if x <= 0 else x))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')


# In[ ]:


d = pd.get_dummies(data['nperps'].apply(lambda x: None if (x <= 0 or x >= 20) else x))
d['ClusterName'] = data['ClusterName']
_ = sns.heatmap(d.groupby('ClusterName').mean().T, cmap='coolwarm')

