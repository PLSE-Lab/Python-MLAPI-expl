#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv("../input/CC GENERAL.csv")
df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.corr()


# In[ ]:


df.isna().sum()


# In[ ]:


df=df.fillna(df.median())


# In[ ]:


df.isna().sum()


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


df.drop('CUST_ID',axis=1,inplace=True)


# In[ ]:


df.info()


# In[ ]:


sse_ = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k).fit(df)
    sse_.append([k, kmeans.inertia_])


# In[ ]:


sse_


# ### Elbow method

# In[ ]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1])


# ### Silhouette Analysis

# In[ ]:


from sklearn.metrics import silhouette_score


# In[ ]:


sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(df)
    sse_.append([k, silhouette_score(df, kmeans.labels_)])


# In[ ]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);


# ### Create an instance of a K Means mode

# In[ ]:


kmeans=KMeans(n_clusters=8)


# In[ ]:


kmeans.fit(df)


# In[ ]:


kmeans.cluster_centers_


# In[ ]:


y_kmeans = kmeans.predict(df)


# In[ ]:


y_kmeans


# In[ ]:


df["cluster"] = y_kmeans


# In[ ]:


df.head()


# ### As here we don't have Y label, we cannot find the score
