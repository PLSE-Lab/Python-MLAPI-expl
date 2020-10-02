#!/usr/bin/env python
# coding: utf-8

# # [Abalone](http://archive.ics.uci.edu/ml/datasets/Abalone)
# 
# 8 features
# 
# Applying PCA. PCA - 80% of variance.
# 
# 
# 
# 
# 
# 
# 

# 

# ***

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import seaborn as sns
sns.set()


# In[2]:


df1 = pd.read_csv("../input/abalone.csv", header=1)
df1.head(5)


# In[3]:


df1.drop(['M'], axis=1, inplace=True)
 


# In[4]:


X = df1.iloc[:, 1:]


# In[5]:


X.head()


# In[6]:


from sklearn import preprocessing


# In[7]:


from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc = StandardScaler()


# 
# minmax_scAaler = preprocessing.MinMaxScaler(feature_range =(0,1))
# data_minmax  = minmax_scaler.fit_transform(X)

# In[8]:


scaled_data = preprocessing.scale(X)
minmax_scaler = preprocessing.MinMaxScaler(feature_range =(0,1))
data_minmax  = minmax_scaler.fit_transform(X)


# In[9]:


from sklearn.decomposition import PCA


# In[10]:


pca = PCA(n_components=None)
X_sc = sc.fit_transform(X)
pca.fit(data_minmax)
np.cumsum(pca.explained_variance_ratio_)


# In[11]:


plt.plot(np.cumsum(pca.explained_variance_ratio_)*100.)
plt.xlabel('number of components')
plt.ylabel('cummulative explained variance');


# In[12]:


from sklearn.cluster import KMeans


# In[13]:


kmeans = KMeans(n_clusters=5) #n_clusters=3,5
kmeans.fit(X)


# In[14]:


y_kmeans = kmeans.predict(X)


# In[15]:


from sklearn.metrics import silhouette_score


# In[16]:


sse_ = []
for k in range(2, 8):
    kmeans = KMeans(n_clusters=k).fit(X)
    sse_.append([k, silhouette_score(X, kmeans.labels_)])


# In[17]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);


# In[ ]:




