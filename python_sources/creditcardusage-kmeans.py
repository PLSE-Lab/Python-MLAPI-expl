#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Necessary Library
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import pandas_profiling

from sklearn.cluster import KMeans 
import pylab as pl
from sklearn import cluster

import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


# In[ ]:


#load the data to dataframe
data = pd.read_csv("../input/card-usage/CreditCardUsage.csv")


# # Preprocessing

# In[ ]:


pandas_profiling.ProfileReport(data)


# In[ ]:


data.duplicated().sum()


# In[ ]:


data.drop_duplicates(inplace = True)


# In[ ]:


data.isna().sum()


# In[ ]:


data['CREDIT_LIMIT']=data['CREDIT_LIMIT'].fillna(data['CREDIT_LIMIT'].mean())


# In[ ]:


data['MINIMUM_PAYMENTS']=data['MINIMUM_PAYMENTS'].fillna(data['MINIMUM_PAYMENTS'].mean())


# In[ ]:


data.isna().sum()


# In[ ]:


data.drop(labels=['CUST_ID','ONEOFF_PURCHASES'],axis=1,inplace=True)


# In[ ]:


pandas_profiling.ProfileReport(data)


# In[ ]:


data.corr()


# In[ ]:


sns.heatmap(data.corr())


# In[ ]:


data.head()


# In[ ]:


data.shape


# In[ ]:


data.info()


# In[ ]:


data.describe().T


# In[ ]:


from sklearn.preprocessing import StandardScaler
X = data.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# In[ ]:


clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Score')
pl.title('Elbow Curve')
pl.show()


# In[ ]:


kn = KMeans(n_clusters=6)
kn.fit(X)


# In[ ]:


kn.inertia_


# In[ ]:


Nc = range(1, 20)
kmeans = [KMeans(n_clusters=i) for i in Nc]
kmeans
score = [kmeans[i].fit(X).inertia_ for i in range(len(kmeans))]
score
pl.plot(Nc,score)
pl.xlabel('Number of Clusters')
pl.ylabel('Sum of within sum square')
pl.title('Elbow Curve')
pl.show()


# In[ ]:


data['label']=kn.labels_


# In[ ]:


sns.set_palette("RdBu_r", 6)
sns.scatterplot(data['BALANCE'],data['PURCHASES'],hue=data['label'],palette="RdBu_r")


# In[ ]:


sns.set_palette('Set1')
sns.scatterplot(data['PURCHASES'],data['PAYMENTS'],hue=data['label'],palette='Set2')

