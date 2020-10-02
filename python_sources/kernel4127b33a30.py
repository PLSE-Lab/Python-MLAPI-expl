#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv("../input/ccdata/CC GENERAL.csv")
df.head()


# In[ ]:


#to add values to missing data
missing = df.isna().sum()
print(missing)
df = df.fillna( df.median() )
missing = df.isna().sum()
print(missing)


# In[ ]:


df.describe()


# In[ ]:


#we will first cluster the data on the basis of their transaction number and purchase amount


# In[ ]:


df_new = df.filter(['PURCHASES', 'PURCHASES_TRX', 'PURCHASES_FREQUENCY'], axis=1)
df_new.head()


# In[ ]:


#to standarize the data
from sklearn.preprocessing import StandardScaler
scaled_features = StandardScaler().fit_transform(df_new.values)
scaled_features_df = pd.DataFrame(scaled_features, index=df_new.index, columns=df_new.columns)


# In[ ]:


scaled_features_df.describe()


# In[ ]:


vals = scaled_features_df.values
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300) 
y_pred = kmeans.fit_predict( vals )
labels=kmeans.labels_


# In[ ]:


clusters=pd.concat([df, pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()


# In[ ]:


'''Here type 0 cluster represents low number of transactions and low amount spent on transaction, type represents significant number of transactions and amout spent 1, we will take data points where we have type 1 cluster'''
cluster_1_df = df[clusters['cluster'] == 1] 


# In[ ]:


cluster_1_df.head()


# In[ ]:


#clustering based on purchase mode


# In[ ]:


purchase_cluster = cluster_1_df.filter(['INSTALLMENTS_PURCHASES', 'ONEOFF_PURCHASES', 'ONEOFF_PURCHASES_FREQUENCY', 'PURCHASES_INSTALLMENTS_FREQUENCY'], axis=1)


# In[ ]:


purchase_cluster.describe()


# In[ ]:


purchase_cluster_scaled_features = StandardScaler().fit_transform(purchase_cluster.values)
purchase_cluster_df = pd.DataFrame(purchase_cluster_scaled_features, index=purchase_cluster.index, columns=purchase_cluster.columns)
purchase_cluster_df.describe()


# In[ ]:


vals = purchase_cluster_df.values


# In[ ]:


kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10, max_iter=300) 
y_pred = kmeans.fit_predict( vals )
labels=kmeans.labels_


# In[ ]:


clusters=pd.concat([purchase_cluster.reset_index(), pd.DataFrame({'cluster':labels})], axis=1)
clusters.head()


# In[ ]:


#groupby cluster, here cluster 0 --- more transaction on installments, and cluster 1-----more transaction with oneoff purchase
gk = clusters.groupby('cluster') 
gk.count() 

