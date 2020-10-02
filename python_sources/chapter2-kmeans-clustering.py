#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#import the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns  #Python library for Vidualization
import matplotlib.pyplot as plt


# In[ ]:


#Import the dataset
dataset = pd.read_csv('../input/Mall_Customers.csv')
dataset.head(10) #Printing first 10 rows of the dataset


# In[ ]:


### Feature sleection for the model
#Considering only 2 features (Annual income and Spending Score) and no Label available
customers=dataset[['Annual Income (k$)','Spending Score (1-100)']]


# In[ ]:


customers.head()


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


#Model Build
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
customers['Cluster']= kmeansmodel.fit_predict(customers.values).astype('object')


# In[ ]:


# plot original customers
sns.FacetGrid(customers, size=5)    .map(plt.scatter, "Annual Income (k$)", "Spending Score (1-100)")    .add_legend()


# In[ ]:


# plot customers with Clusters
sns.FacetGrid(customers, hue="Cluster", size=5)    .map(plt.scatter, "Annual Income (k$)", "Spending Score (1-100)")    .add_legend()

