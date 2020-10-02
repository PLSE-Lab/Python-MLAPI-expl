#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


# In[ ]:


top50 = pd.read_csv('../input/top50spotify2019/top50.csv', encoding = "ISO-8859-1")
top50.head()


# In[ ]:


top50.corr().style.background_gradient()


# In[ ]:


dt = top50.drop(['Artist.Name', 'Genre'], axis=1)


# In[ ]:


dt = dt.drop(['Track.Name'], axis=1)


# In[ ]:


dt.head()


# In[ ]:


sns.clustermap(dt.corr())


# In[ ]:


sns.clustermap(dt.corr(), cmap='viridis')


# In[ ]:


sns.clustermap(dt.corr(), cmap='mako')


# In[ ]:


sns.clustermap(dt.corr(), cmap='winter', annot=True)


# In[ ]:


my_palette = dict(zip(dt, ["orange","yellow","brown"]))


# In[ ]:


sns.clustermap(dt.corr(), metric="correlation", cmap="Blues")


# In[ ]:


Z = linkage(dt, 'ward')


# In[ ]:


plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('sample index')
plt.ylabel('distance (Ward)')
dendrogram(Z, labels=dt.index, leaf_rotation=90)


# In[ ]:


# Usual boxplot
ax = sns.boxplot(x=dt['Danceability'], y=dt['Beats.Per.Minute'], data=dt)
ax = sns.swarmplot(x=dt['Danceability'], y=dt['Beats.Per.Minute'], data=dt, color="grey")


# # Work in progress.
