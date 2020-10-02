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


# # Check Descriptive Statistics

# In[ ]:


df.describe()


# In[ ]:


#Plot Age, Income and Spending Score Correlation
sns.pairplot(df[['Age','Income', 'Spending_Score']])


# # Perform K-Mean Clustering with 5 Clusters

# In[ ]:


import sklearn.cluster as cluster


# In[ ]:


# We will use 2 Variables for this example
kmeans = cluster.KMeans(n_clusters=5 ,init="k-means++")
kmeans = kmeans.fit(df[['Spending_Score','Income']])


# In[ ]:


kmeans.cluster_centers_


# # Attach Clusters to the Original Data 

# In[ ]:


df['Clusters'] = kmeans.labels_


# In[ ]:


df.head()


# In[ ]:


df['Clusters'].value_counts()


# # Export Data with Clusters

# In[ ]:


df.to_csv('mallClusters.csv', index = False)


# # Plot Cluster on Chart 

# In[ ]:


sns.scatterplot(x="Spending_Score", y="Income",hue = 'Clusters',  data=df)


# ### In the next video, we will create Elbow Plot and Silhoutte Score

# # END
