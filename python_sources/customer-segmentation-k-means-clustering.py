#!/usr/bin/env python
# coding: utf-8

# import libraries

# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Reading the dataset.
# 1. Made the index column as index and dropped it as column
# 2. Removed the gender column as it's binary data, so will bias the model to cluster along it's own dimenstion.

# In[48]:


cust=pd.read_csv("../input/Mall_Customers.csv")
cust.columns=["id","gender","age","income","score"]
cust.index=cust.id.values
cust=cust.drop(["id","gender"],axis=1)
cust.head()


# Statistical overview of the data

# In[49]:


cust.describe()


# Plotting the data to see if we can manually decide the number of cluster centers

# In[50]:


fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(cust.age,cust.income,cust.score)


# It's almost clear from the data that there are 3 clusters:
# 1. with low age and high income
# 2. with low age and low income
# 3. with high age and low income

# Using sklearn standard kmeans method to verify our hypothesis

# In[51]:


from sklearn.cluster import KMeans
kmean=KMeans(n_clusters=3).fit_predict(cust.values)


# In[52]:


fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(cust.age,cust.income,cust.score,c=kmean)


# More or less the clusters are formed as similar to our hypothesis. Although there is some discrepancies between customers with low income-low age and low income-high age.

# In[ ]:




