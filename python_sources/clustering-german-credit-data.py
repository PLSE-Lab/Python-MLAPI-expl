#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# ## 1. Business Understanding

# #### 5 C's of Credit
# The five Cs of credit is a system used by lenders to gauge the creditworthiness of potential borrowers. Creditworthiness is how a lender determines that you will default on your debt obligations, or how worthy you are to receive new credit. Your creditworthiness is what creditors look at before they approve any new credit to you. The five Cs of credit are character, capacity, capital, collateral, and conditions.
# 1. Character : Credit history of the customer
# 2. Capacity : Assesses borrower's debt-to-income ratio
# 3. Capital :Assesses borrower's seriousness level
# 4. Collateral : It gives the lender the assurance that if the borrower defaults on the loan, the lender can get something back by repossessing the collateral
# 5. Conditions : Conditions are the external variables that can affect credit and credit quality. This refers to national, international and local economy, the industry and the bank itself.
# 
# We will try to find the creditworthiness of the customer on the German Credit Dataset.
# 
# Source :https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1872804

# ## 2. Load dataset and quick look

# ### 1.1. Load Dataset

# In[ ]:


df = pd.read_csv('/kaggle/input/german-credit/german_credit_data.csv')


# ### 1.2. Quick look

# In[ ]:


df.shape


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# ## 3. Missing values identification and handling

# In[ ]:


df.isnull().sum()


# In[ ]:


numerical = ['Credit amount','Age','Duration']
categorical = ['Sex','Job','Housing','Saving accounts','Checking account','Purpose']
unused = ['Unnamed: 0']


# In[ ]:


df = df.drop(columns = unused)
df.shape


# In[ ]:


for cat in categorical:
    df[cat] = df[cat].fillna(df[cat].mode().values[0])


# In[ ]:


df.isnull().sum()


# ## 4. Visualize

# In[ ]:


sns.pairplot(df)


# Above are the pairplot of all the numerical features.

# In[ ]:


fig = plt.figure(figsize = (20,15))
axes = 320
for cat in categorical:
    axes += 1
    fig.add_subplot(axes)
    sns.countplot(data = df, x = cat)
    plt.xticks(rotation=30)
plt.show()


# Above are the bar plot of all the categorical feature. From the bar plot above, we can get some insight. That are:
# 1. The amount of men are greater than women
# 2. Most of the customer are skilled on their job
# 3. Most of the customer have their own house
# 4. Most of the customer have little saving account
# 5. Most of the customer hav little checking account
# 6. Most of the customer use credit for car

# In[ ]:


#create correlation
corr = df.corr(method = 'pearson')

#convert correlation to numpy array
mask = np.array(corr)

#to mask the repetitive value for each pair
mask[np.tril_indices_from(mask)] = False
fig, ax = plt.subplots(figsize = (15,12))
fig.set_size_inches(15,15)
sns.heatmap(corr, mask = mask, vmax = 0.9, square = True, annot = True)


# From the heatmap above we can see that the best correlation is between credit amount and duration. But so far we will still use all the numeric features for the clustering.

# In[ ]:


df_cluster = pd.DataFrame()
df_cluster['Credit amount'] = df['Credit amount']
df_cluster['Age'] = df['Age']
df_cluster['Duration'] = df['Duration']
df_cluster['Job'] = df['Job']
df_cluster.head()


# In[ ]:


fig = plt.figure(figsize = (15,10))
axes = 220
for num in numerical:
    axes += 1
    fig.add_subplot(axes)
    sns.boxplot(data = df, x = num)
plt.show()


# From the figures above we can see that there are still some outliers on the numerical features.

# In[ ]:


fig, (ax1, ax2, ax3, ax4) = plt.subplots(4,1, figsize=(8,8))
sns.distplot(df["Age"], ax=ax1)
sns.distplot(df["Credit amount"], ax=ax2)
sns.distplot(df["Duration"], ax=ax3)
sns.distplot(df["Job"], ax=ax4)
plt.tight_layout()
plt.legend()


# From the figure above, we can see that distributions are right-skewed.

# ## 5. Feature Engineering

# ### 5.1. Log Transform

# We can use logarithmic transformation to reduce the outliers and distribution skewness.

# In[ ]:


df_cluster_log = np.log(df_cluster[['Age', 'Credit amount','Duration']])

fig, (ax1, ax2, ax3) = plt.subplots(3,1, figsize=(8,8))
sns.distplot(df_cluster_log["Age"], ax=ax1)
sns.distplot(df_cluster_log["Credit amount"], ax=ax2)
sns.distplot(df_cluster_log["Duration"], ax=ax3)
plt.tight_layout()


# We can see that the skewness of the distribution is eliminatied.

# ### 5.2. Fit & Transform

# In[ ]:


df_cluster_log.head()


# In[ ]:


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
cluster_scaled = scaler.fit_transform(df_cluster_log)


# ## 6. Models

# We will make a clustering model based on the features we already choose. We will try three clustering models, those are:
# 1. K-Means
# 2. Hierarchical Agglomerative Clustering
# 3. DBSCAN

# ### 6.1. K-Means

# First, we use Elbow Method to determine the optimal k value for the k-means.

# In[ ]:


from sklearn.cluster import KMeans

Sum_of_squared_distances = []
K = range(1,15)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(cluster_scaled)
    Sum_of_squared_distances.append(km.inertia_)
plt.figure(figsize=(20,5))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# From the figure above we can see that the most optimal values are 3. So we choose 3 as the k values of the k-means model.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D

model = KMeans(n_clusters=3)
model.fit(cluster_scaled)
kmeans_labels = model.labels_

fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")

ax.scatter3D(df_cluster['Age'],df_cluster['Credit amount'],df_cluster['Duration'],c=kmeans_labels, cmap='rainbow')

xLabel = ax.set_xlabel('Age', linespacing=3.2)
yLabel = ax.set_ylabel('Credit Amount', linespacing=3.1)
zLabel = ax.set_zlabel('Duration', linespacing=3.4)
print("K-Means")


# From the figure above we could see that the cluster segmented well.

# In[ ]:


df_clustered_kmeans = df_cluster.assign(Cluster=kmeans_labels)
grouped_kmeans = df_clustered_kmeans.groupby(['Cluster']).mean().round(1)
grouped_kmeans


# The table above shows the centroid of each clusters that could determine the clusters rule. These are:<br>
# <br>
# Cluster 0 : Higher credit amount, middle-aged, long duration customers<br>
# Cluster 1 : Lower credit amount, young, short duration customers<br>
# Cluster 2 : Lower credit amount, old, short duration customers

# ### 6.2. Hierarchical Agglomerative Clustering

# On this model, to determine the n_clusters we can use dendogram. 

# In[ ]:


import scipy.cluster.hierarchy as sch

plt.figure(figsize=(20,10))
dendrogram = sch.dendrogram(sch.linkage(cluster_scaled, method='ward'))


# From the dendogram above, we can see that the most optimal n_clusters is 4.

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
model = AgglomerativeClustering(n_clusters=4)
model.fit(cluster_scaled)
hac_labels = model.labels_

fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")

ax.scatter3D(df_cluster['Age'],df_cluster['Credit amount'],df_cluster['Duration'],c=hac_labels, cmap='rainbow')

xLabel = ax.set_xlabel('Age', linespacing=3.2)
yLabel = ax.set_ylabel('Credit Amount', linespacing=3.1)
zLabel = ax.set_zlabel('Duration', linespacing=3.4)
print("Hierarchical Agglomerative Clustering")


# In[ ]:


df_clustered_hac = df_cluster.assign(Cluster=hac_labels)
grouped_hac = df_clustered_hac.groupby(['Cluster']).mean().round(1)
grouped_hac


# The table above shows the centroid of each clusters that could determine the clusters rule. These are:<br>
# <br>
# Cluster 0 : Higher credit amount, old, long duration customers<br>
# Cluster 1 : Lower credit amount, young, long duration customers<br>
# Cluster 2 : Lower credit amount, old, short duration customers<br>
# Cluster 3 : Lower credit amount, young, short duration customers

# ### 6.3. DBSCAN

# In[ ]:


from sklearn.cluster import DBSCAN

model = DBSCAN()
model.fit(cluster_scaled)
dbs_labels = model.labels_

fig = plt.figure(num=None, figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
ax = plt.axes(projection="3d")

ax.scatter3D(df_cluster['Age'],df_cluster['Credit amount'],df_cluster['Duration'],c=dbs_labels, cmap='rainbow')

xLabel = ax.set_xlabel('Age', linespacing=3.2)
yLabel = ax.set_ylabel('Credit Amount', linespacing=3.1)
zLabel = ax.set_zlabel('Duration', linespacing=3.4)
print("DBSCAN")


# From the figure above we can see that DBSCAN is not suitable for this dataset.

# ## 7. Result Analysis

# From these models, we can choose the most well segmented model, that is k-means. We use the clusters from the that model to analyze the dataset.

# In[ ]:


grouped_kmeans


# Looking back from the centroid of the clusters, let's see the "returning power" of each of the centroid by dividing the Credit amount with the duration. The higher the "returning power".

# In[ ]:


df_clustered = df.assign(Cluster=kmeans_labels)


# In[ ]:


df_clustered.head()


# In[ ]:


fig = plt.figure(figsize = (20,15))
axes = 320
for cat in categorical:
    axes += 1
    fig.add_subplot(axes)
    sns.countplot(data = df_clustered, hue=df_clustered['Cluster'], x = cat)
    plt.xticks(rotation=30)
plt.show()


# Above are figures of the clusters distribution on each categorical feature.

# ## 8. Summaries

# 1. After comparing three kind of clustering models, we decide to use k-means as the model 
# 2. The data divided into three clusters
# 3. The three clusters can be used to determine the creditworthiness of the German Credit potential borrowers
# 4. Each of the cluster have their own characteristics

# In[ ]:




