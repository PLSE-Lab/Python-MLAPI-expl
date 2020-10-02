#!/usr/bin/env python
# coding: utf-8

# ***Please UpVote if you like the work!!!***

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import zscore
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# # KMeans on Iris Dataset

# In[ ]:


sns.set(style = 'ticks',color_codes=True)


# In[ ]:


df2 = pd.read_csv('../input/matplotlib-datasets/iris_dataset.csv')


# In[ ]:


df2


# In[ ]:


df2.drop('species',axis = 1,inplace = True)


# In[ ]:


df_scaled = df2.apply(zscore)


# In[ ]:


sns.pairplot(df_scaled,diag_kind='kde')


# In[ ]:


clusters_range = range(1,15)
inertia = []
for num_clust in clusters_range:
  model = KMeans(n_clusters = num_clust,random_state = 2)
  model.fit(df_scaled)
  inertia.append(model.inertia_)


# In[ ]:


plt.plot(clusters_range,inertia,marker = 'o')


# As we can see, at k = 3, the inertia values starts reducing at a constant rate. So we select k = 3 and build our kmeans model.

# The total inertia for kmeans at k = 3 is 140.96

# In[ ]:


kmeans = KMeans(n_clusters=3,random_state=2)
kmeans.fit(df_scaled)
df2['class'] = kmeans.labels_
df2


# In[ ]:


df2['class'].value_counts()


# In[ ]:


sns.pairplot(df2,hue = 'class')


# # Agglomerative Clustering on Iris Dataset

# In[ ]:


from sklearn.cluster import AgglomerativeClustering


# In[ ]:


agc = AgglomerativeClustering(n_clusters=3)
agc.fit(df_scaled)


# In[ ]:


df_scaled['agc_class'] = agc.labels_


# In[ ]:


df_scaled['agc_class'].value_counts()


# In[ ]:


df_scaled


# In[ ]:


grps = df_scaled.groupby('agc_class')


# In[ ]:


grp0 = grps.get_group(0)
grp1 = grps.get_group(1)
grp2 = grps.get_group(2)


# In[ ]:


c0 = np.array([grp0['petal_length'].mean(),grp0['petal_width'].mean(),grp0['sepal_length'].mean(),grp0['sepal_width'].mean()])
c1 = np.array([grp1['petal_length'].mean(),grp1['petal_width'].mean(),grp1['sepal_length'].mean(),grp1['sepal_width'].mean()])
c2 = np.array([grp2['petal_length'].mean(),grp2['petal_width'].mean(),grp2['sepal_length'].mean(),grp2['sepal_width'].mean()])


# In[ ]:


df_scaled.columns


# In[ ]:


inertia_0 = np.sum(((grp0['petal_length'] - c0[0])**2) + ((grp0['petal_width'] - c0[1])**2) + ((grp0['sepal_length'] - c0[2])**2) + ((grp0['sepal_width'] - c0[3])**2))
inertia_1 = np.sum(((grp1['petal_length'] - c1[0])**2) + ((grp1['petal_width'] - c1[1])**2) + ((grp1['sepal_length'] - c1[2])**2) + ((grp1['sepal_width'] - c1[3])**2))
inertia_2 = np.sum(((grp2['petal_length'] - c2[0])**2) + ((grp2['petal_width'] - c2[1])**2) + ((grp2['sepal_length'] - c2[2])**2) + ((grp2['sepal_width'] - c2[3])**2))
total_inertia = inertia_0 + inertia_1 + inertia_2
total_inertia


# The total inertia for agglomerative clustering at k = 3 is 150.12 whereas for kmeans clustering its 140.96
# 
# Hence we can conclude that for iris dataset kmeans is better clustering option as compared to agglomerative clustering as inertia is low for kmeans.

# In[ ]:


df_scaled.drop('agc_class',axis = 1,inplace = True)


# In[ ]:


from scipy.cluster.hierarchy import dendrogram,linkage
plt.figure(figsize=(10,5))
plt.xlabel('sample index')
plt.ylabel('distance')
z = linkage(df_scaled,method='ward')
dendrogram(z,leaf_rotation=90,p = 5,color_threshold=10,leaf_font_size=10,truncate_mode='level')
plt.tight_layout()


# ***Please UpVote if you like the work!!!***
