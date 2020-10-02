#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[2]:


df = pd.read_csv('../input/Mall_Customers.csv')


# In[3]:


df.head()


# In[54]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
sns.heatmap(df.corr())


# Here we first get an overview of the dataset and understand it before we make any form of clustering

# In[4]:


df.shape


# We now check if there are any null values that are present in the dataset first

# In[5]:


df.info()


# Before that we have to map the values in gender to either a 1 or 0, we can do this using onehotencoder

# In[6]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[7]:


df_tf = df
df_tf.Gender = le.fit_transform(df_tf['Gender'])


# In[8]:


df_tf.head()


# We will now install some useful packages for us to process the data

# In[9]:


from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


# In[10]:


scaler = MinMaxScaler()


# In[11]:


df_tf['Age'] = scaler.fit_transform(df_tf[['Age']])


# In[12]:


df_tf['Annual Income (k$)'] = scaler.fit_transform(df_tf[['Annual Income (k$)']])
df_tf['Spending Score (1-100)'] = scaler.fit_transform(df_tf[['Spending Score (1-100)']])


# In[13]:


df_tf.head()


# In[59]:


X = df_tf.drop(['CustomerID','Spending Score (1-100)'], axis='columns')


# In[60]:


X.head()


# Now is where we start to implelment the actual algorithm, we will iterate the data over a range of K values and then fit them into the KMeans algorithm. We will then append the sum of squred errors using 'inertia_' method and then plot the sse. 

# In[61]:


k_rng = range(1,10)
sse = []
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X)
    sse.append(km.inertia_)


# In[62]:


plt.plot(k_rng,sse)
plt.xlabel('K values')
plt.ylabel('Sum of Squared Errors')


# We can tell that the elbow method's optimal elbow value is either 2 or 4, let's plot both and compare

# In[63]:


km = KMeans(n_clusters=4)


# In[64]:


y_4 = km.fit_predict(X)
y_4


# In[65]:


X['cluster'] = y_4


# In[66]:


X.head()


# Separating the data by their clusters

# In[72]:


cluster0 = X.loc[X.cluster==0]
cluster1 = X.loc[X.cluster==1]
cluster2 = X.loc[X.cluster==2]
cluster3 = X.loc[X.cluster==3]


# In[73]:


cluster1.head()


# In[74]:


from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Since from the correlation plot earlier,we 

# In[76]:


fig = pyplot.figure()
ax = Axes3D(fig)
ax.set_xlabel('Gender')
ax.set_ylabel('Age')
ax.set_zlabel('Annual Income')
ax.scatter(cluster0['Gender'],cluster0['Age'],cluster0['Annual Income (k$)'],color='red')
ax.scatter(cluster1['Gender'],cluster1['Age'],cluster1['Annual Income (k$)'],color='blue')
ax.scatter(cluster2['Gender'],cluster2['Age'],cluster2['Annual Income (k$)'],color='green')
ax.scatter(cluster3['Gender'],cluster3['Age'],cluster3['Annual Income (k$)'],color='yellow')

