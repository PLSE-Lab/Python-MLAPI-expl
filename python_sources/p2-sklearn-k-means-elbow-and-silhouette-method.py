#!/usr/bin/env python
# coding: utf-8

# # Import Libraries

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import Data & Rename Columns

# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


df = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
df.head()


# In[ ]:


df.rename(columns={'Annual Income (k$)' : 'Income', 'Spending Score (1-100)' : 'Spending_Score'}, inplace = True)
df.head()


# In[ ]:


df_Short = df[['Income','Spending_Score']]
df_Short.head()


# # Elbow Method to Indetify Clusters

# In[ ]:


import sklearn.cluster as cluster


# ## Run Cluster Analysis 12 times
# 
# - We run the Cluster Analysis using Cluster as 1 till 12. Also, we store the WSS Scores. The WSS score will be used to create the Elbow Plot
# - WSS = Within-Cluster-Sum of Squared

# In[ ]:


K=range(1,12)
wss = []
for k in K:
    kmeans=cluster.KMeans(n_clusters=k,init="k-means++")
    kmeans=kmeans.fit(df_Short)
    wss_iter = kmeans.inertia_
    wss.append(wss_iter)


# ## We Store the Number of clusters along with their WSS Scores in a DataFrame

# In[ ]:


mycenters = pd.DataFrame({'Clusters' : K, 'WSS' : wss})
mycenters


# # Plot Elbow Plot

# In[ ]:


sns.scatterplot(x = 'Clusters', y = 'WSS', data = mycenters, marker="+")
# We get 5 Clusters


# ## -- 5 Clusters Identified as per Elbow Method

# # Silhouette Method to Indentify Clusters

# In[ ]:


import sklearn.metrics as metrics


# In[ ]:


for i in range(3,13):
    labels=cluster.KMeans(n_clusters=i,init="k-means++",random_state=200).fit(df_Short).labels_
    print ("Silhouette score for k(clusters) = "+str(i)+" is "
           +str(metrics.silhouette_score(df_Short,labels,metric="euclidean",sample_size=1000,random_state=200)))


# ## -- Max Silhouette Score as k = 5, Hence 5 Clusters is the right option

# # Perform K-Mean Clustering with 5 Clusters

# In[ ]:


# We will use 2 Variables for this example
kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")
kmeans = kmeans.fit(df[['Spending_Score','Income']])


# # Plot Clusters on Chart

# In[ ]:


df['Clusters'] = kmeans.labels_


# In[ ]:


kmeans.labels_


# In[ ]:


sns.scatterplot(x="Spending_Score", y="Income",hue = 'Clusters',  data=df)


# # END
