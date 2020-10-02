#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist, pdist
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # Data cleaning and exploration

# In[ ]:


data = pd.read_csv('../input/unsupervised-learning-on-country-data/Country-data.csv')
data_dict = pd.read_csv('../input/unsupervised-learning-on-country-data/data-dictionary.csv')
print(data.dtypes)
print(data.head(3))
print(data_dict)


# In[ ]:


# Detect number of NaN values in the table
NAN = [(col,data[col].isna().sum()) for col in data]
NAN = pd.DataFrame(NAN, columns=['Column_Name', 'Num_of_NaN'])
NAN


# In[ ]:


# Check if there is replication in the data
print('The dataset contains repeated country:', data.duplicated().any())
# General discription of the table
data.describe()


# In[ ]:


# separate country column and the rest variables
data.index=data.iloc[:,0]
data_1=data.drop(["country"], axis=1)


# In[ ]:


# find the correlation between variables
corr = data_1.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(10, 220, n=20),
    square=True)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right')


# In[ ]:


# plot the pair scatter plots of all factors
sns.pairplot(data_1)


# From the correlation map we see that 
# - **child_mort** has a strong inverse correlation with **income**(net income per person), **life_expect**(expected years of living) and **gdpp**(the GDP per capita), a moderate correlation with **inflation**(annual growth rate of the total GDP) and **total_fert**(fertility rate). 
# - **income** is correlated with **exports** and **gdpp**.
# - **exports** is correlated with **imports**
# - **life_expec** is correlated with **income** and **gdpp**, inversly correlated with **total_fert**

# The variables can be didived into economic metrics(exports, imports, income, inflation, gdpp, health), health metrics(child_mort, life_expec, total_fer).
# 

# - It's also noticed that there are obvious outliers in the income, gdpp, total_fert scatter plots.

# In[ ]:


# find the outliers
print(data.query('child_mort>100 & income>20000')) #outlier in income/gdpp plot
# data.query('child_mort>100 & gdpp>17000')
print(data.query('inflation>100')) #outlier in inflation plot
print(data.query('child_mort>200')) #outlier in total_fer plot


# # Clustering the countries

# In[ ]:


# clustering the samples based on health factors(child_mort, life_expec, total_fer)
data_clustering_1=data[['child_mort', 'life_expec', 'total_fer']]

# plot sum of squared distances (elbow method)
K = range(1,10)
data_km = [KMeans(n_clusters=k).fit(data_clustering_1) for k in K]
ssd = [data_km[k].inertia_ for k in range(len(data_km))]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, ssd, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow for KMeans clustering')
plt.show()


# Take k=3

# In[ ]:


kmeans = KMeans(n_clusters=3)
model = kmeans.fit(data_clustering_1)
pred = model.labels_
data_1['cluster_km'] = pred
data_1.head()


# # Visualize k-means clustering results

# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
cluster_0 = data_1.query('cluster_km==0')
cluster_1 = data_1.query('cluster_km==1')
cluster_2 = data_1.query('cluster_km==2')
plt_0 = ax.scatter(cluster_0['child_mort'], cluster_0['life_expec'], cluster_0['total_fer'], c='c', s=10)
plt_1 = ax.scatter(cluster_1['child_mort'], cluster_1['life_expec'], cluster_1['total_fer'], c='r', s=10)
plt_2 = ax.scatter(cluster_2['child_mort'], cluster_2['life_expec'], cluster_2['total_fer'], c='g', s=10)
ax.set_xlabel('child_mort')
ax.set_ylabel('life_expec')
ax.set_zlabel('total_fer')
plt.tight_layout()
plt.show()


# We can see from the plot that cluster 2 are high child_mort, high total_fer and low life_expec, indicating poor social health, so we want to further cluster these data points based on their economic conditions.

# In[ ]:


# Find out the number data points in cluster 2
print(cluster_2.count())


# In[ ]:


# clustering the samples based on economic factors(child_mort, life_expec, total_fer)
data_clustering_2=cluster_2[['income', 'gdpp', 'health']]

# plot sum of squared distances (elbow method)
K = range(1,10)
data_km = [KMeans(n_clusters=k).fit(data_clustering_2) for k in K]
ssd = [data_km[k].inertia_ for k in range(len(data_km))]
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(K, ssd, 'b*-')
plt.grid(True)
plt.xlabel('Number of clusters')
plt.ylabel('Sum of squared distances')
plt.title('Elbow for KMeans clustering')
plt.show()


# Take k=3

# In[ ]:


kmeans = KMeans(n_clusters=3)
model = kmeans.fit(data_clustering_2)
pred = model.labels_
cluster_2['cluster_km'] = pred
cluster_2.head()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
cluster_2_0 = cluster_2.query('cluster_km==0')
cluster_2_1 = cluster_2.query('cluster_km==1')
cluster_2_2 = cluster_2.query('cluster_km==2')
plt_0 = ax.scatter(cluster_2_0['income'], cluster_2_0['gdpp'], cluster_2_0['health'], c='c', s=10, label='cluster0')
plt_1 = ax.scatter(cluster_2_1['income'], cluster_2_1['gdpp'], cluster_2_1['health'], c='r', s=10, label='cluster1')
plt_2 = ax.scatter(cluster_2_2['income'], cluster_2_2['gdpp'], cluster_2_2['health'], c='g', s=10, label='cluster2')
ax.set_xlabel('income')
ax.set_ylabel('gdpp')
ax.set_zlabel('health')
ax.legend()
plt.tight_layout()
plt.show()


# From the plot we see that cluster 0 of the second k-means contains countries with low income and low gdpp. The health expenses vary. 

# In[ ]:


# sort cluster0 by gdpp, income, health ascendingly
print(cluster_2_0.sort_values(['income','gdpp','health'],ascending=True))


# The unsupervised learning results suggest that the aid should be provided to the countries at the top of the list(Congo, Niger, Central African Republic, etc.).
