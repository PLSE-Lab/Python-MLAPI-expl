#!/usr/bin/env python
# coding: utf-8

# In[60]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns # data visualization
import matplotlib.pyplot as plt # for adjusting graph
from sklearn.cluster import KMeans # clustering model

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[61]:


#Import data
df = pd.read_csv('../input/Mall_Customers.csv')
df.head()


# In[62]:


#Check the shape
df.shape


# In[63]:


#Look at some simple statistics
df.describe()


# In[64]:


#Check if there is null data and object type are correct
df.info()


# In[65]:


#Histogram of Annual Income (k$)
sns.distplot(df['Annual Income (k$)'],kde = False,rug = True)
plt.title('Histogram of Annual Income (k$)')
plt.ylabel('Count')


# In[67]:


sns.distplot(df['Spending Score (1-100)'],kde = False,rug = True)
plt.title('Histogram of Spending Score (1-100)')
plt.ylabel('Count')


# In[68]:


sns.distplot(df['Age'],kde = False,rug = True)
plt.title('Histogram of Age')
plt.ylabel('Count')


# In[66]:


#Plot each feature together in scatter plot
sns.pairplot(df.iloc[:,1:],kind='reg')


# In[69]:


#Prepare data and convert catagorical data to dummies variable
x = df.iloc[:,1:]
x = pd.get_dummies(x)
x.head()


# In[70]:


#Create model and fit with data
kmean = KMeans(n_clusters = 5 , random_state = 0).fit(x)
kmean.labels_


# In[71]:


#Add the results back
df['Cluster'] = kmean.labels_ + 1
df.head()


# In[72]:


#Groupby cluster and get average value from each cluster 
df.iloc[:,1:].groupby('Cluster').mean()


# From the results above you will see the characteristic of each cluster and we can describe them like:
# 
# **Cluster 1** is customers with old age, low annual income and low spending score
# 
# **Cluster 2** is customers with average age, high annual income and high spending score
# 
# **Cluster 3** is customers with old age, average annual income and average spending score
# 
# **Cluster 4** is customers with old age, high annual income and low spending score
# 
# **Cluster 5** is customers with young age, low annual income and high spending score
# 
# with all these cluster we can further create a marketing strategy which is suitable for customer in each cluster more than one size fits all strategy.

# 

# In[ ]:





# In[ ]:




