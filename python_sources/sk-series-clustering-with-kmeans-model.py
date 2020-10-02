#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.cluster import KMeans
import numpy as np

X = np.array([[1,2],[1,4],[1,0],[4,2],[4,4],[4,0]])
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)


# In[ ]:


kmeans.labels_


# In[ ]:


kmeans.predict([[0,0],[4,4]])
kmeans.cluster_centers_

