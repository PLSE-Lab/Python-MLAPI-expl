#!/usr/bin/env python
# coding: utf-8

# In[2]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[23]:


df = pd.read_csv("../input/states_all.csv", index_col=0)
df.head()


# In[37]:


df.tail()


# In[36]:


df.dropna(axis=0, inplace=True)


# In[38]:


df.groupby('STATE')[['TOTAL_REVENUE', 'GRADES_ALL_G']].mean()


# In[25]:


new_df = df[['TOTAL_REVENUE', 'GRADES_ALL_G']].copy()
new_df.head()


# In[26]:


new_df.dropna(axis=0, inplace=True)


# In[27]:


new_df.tail()


# In[28]:


new_df.info()


# In[29]:


new_df.plot.scatter(x='TOTAL_REVENUE', y='GRADES_ALL_G')


# In[39]:


new_df2 = df.groupby('STATE')[['TOTAL_REVENUE', 'GRADES_ALL_G']].mean()
new_df2.head()


# In[40]:


new_df2.plot.scatter(x='TOTAL_REVENUE', y='GRADES_ALL_G')


# # Dendrogram

# In[49]:


from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[45]:


sc = StandardScaler()


# In[50]:


base = sc.fit_transform(new_df2)
plt.scatter(x=base[:,0], y=base[:,1])


# In[51]:


dendrograma = dendrogram(linkage(base, method='ward'))


# In[62]:


hc =  AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')


# In[64]:


clusteres = hc.fit_predict(base)
clusteres


# In[65]:


new_df2['cluster'] = clusteres


# In[66]:


plt.scatter(x=new_df2.TOTAL_REVENUE.values, y=new_df2.GRADES_ALL_G, c=new_df2.cluster)


# In[67]:


new_df2[new_df2.cluster == 1]


# In[69]:


new_df2[new_df2.cluster == 2]


# # KMeans

# In[70]:


from sklearn.cluster import KMeans


# In[ ]:




