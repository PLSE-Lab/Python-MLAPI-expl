#!/usr/bin/env python
# coding: utf-8

# In[17]:


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


# The dataset contains records on offers sent to customers and transaction level data. The transactions data shows which offer customers responded to, and what the customer ended up buying. The data is contained in an Excel workbook containing two worksheets. Each worksheet contains a different dataset.

# ### The offers 

# In[13]:


df_wine_offers = pd.read_excel("../input/WineKMC.xlsx", sheetname=0)
df_wine_offers.columns = ["offer_id", "campaign", "varietal", "min_qty", "discount", "origin", "past_peak"]
df_wine_offers.head()


# ### The transcations

# In[20]:


df_wine_transactions = pd.read_excel("../input/WineKMC.xlsx", sheetname=1)
df_wine_transactions.columns = ["customer_name", "offer_id"]
df_wine_transactions['n'] = 1
df_wine_transactions.head()


# In order to learn more about how our customers behave, so we can use their behavior (whether or not they purchased something based on an offer) as a way to group similar minded customers together. We can then study those groups to look for patterns and trends which can help us formulate future offers.So we need a way to compare customers. To do this, we're going to create a matrix that contains each customer and a 0/1 indicator for whether or not they responded to a given offer.
# 

# In[21]:


# merge the two dataframes
my_df = pd.merge(df_wine_offers, df_wine_transactions)
my_df.head()


# In[22]:


# create a matrix 
matrix = my_df.pivot_table(index=['customer_name'], columns=['offer_id'], values='n',fill_value=0)
matrix.head(5)


# ### K-Means Clustering
# 
# For how does K-Means Clustering work. we need to maximize the distance between centroids and minimize the distance between data points and the respective centroid for the cluster they are in. A very intuitive way for deciding the optimal number of clusters for our dataset is the Elbow method... which works as follows.
# 

# In[24]:


# run a first KMeans clustering algorithm as required in the exercise
from sklearn.cluster import KMeans
cluster = KMeans(n_clusters=5)
matrix['cluster'] = cluster.fit_predict(matrix[matrix.columns[2:]])
matrix.cluster.value_counts()


# In[25]:


# the sum of squared error 
ssd = []
K = range(2,11)
for cluster_i in K:
    kmeans = KMeans(n_clusters=cluster_i)
    kmodel = kmeans.fit(matrix[matrix.columns[2:]])
    ssd.append(kmodel.inertia_)


# Plot graphics that will help the decision process. 

# In[27]:


import sklearn
import matplotlib.pyplot as plt
import seaborn as sns


# In[28]:


plt.plot(K, ssd, 'bx-')
plt.xlabel('nr_clusters')
plt.ylabel('ssd')
plt.title('Elbow method for chosing the best number of clusters')
plt.show()


# As the graphics above shows, in the x axis we have the number of clusters and in the y-axis we have sum-of-squares error. So in this graphics we see an elbow between four and six clusters (x-axis) which suggests that the optimal number of clusters is K = 5 

# ## Visualizing Clusters using PCA
# 
# But how can we visualize clusters when each data point has 32 dimensions? Principal Component Analysis (PCA) will help us reduce the dimensionality of our data from 32 to only two dimensions. This way we can visualize the clusters.

# In[29]:


# prepare the x_cols first.
x_cols = matrix.columns[:-1]
x_cols


# In[30]:


#run PCA

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:,0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:,1]
matrix = matrix.reset_index()


# In[31]:


matrix.head(5)


# ### Now we have two columns x, y gnerated from PCA that we can use to visualize the clusyters.

# In[32]:


customer_clusters = matrix[['customer_name', 'cluster', 'x', 'y']]
customer_clusters.head()
# Get current size of figure 
fig_size = plt.rcParams["figure.figsize"]
print(fig_size)

# And reset figure width to 12 and height to 9
fig_size[0] = 12
fig_size[1] = 9
plt.rcParams["figure.figsize"] = fig_size

plt.scatter(customer_clusters['x'], customer_clusters['y'], c = customer_clusters['cluster'])


# ### Here we go!
# #### More than "five clusters" there seem to be only three (well defined) clusters of customers! The green, the black and the yellow-darkgreen.magenta.
