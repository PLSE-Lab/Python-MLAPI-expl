#!/usr/bin/env python
# coding: utf-8

# An investment banking manager has the details of Earnings across different industries. He wants to look at the current details of performance and conclude which segments can be clustered together so that he can then assign Relationship Managers accordingly. The dataset has the Earnings and performance data across different industries.
# 
# **Problem Statement**
# 
# An investment banking manager has the details of earnings across different industries.
# Based on performance, which segments do I cluster together to assign relationship managers? 

# In[ ]:


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


# In[ ]:


# Importing packages

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# Importing the dataset and changinng certain column names.

CS_1 = pd.read_csv('../input/CS_1.csv')
CS_1 = CS_1.rename(index=str, columns={"Industry Name": "Industry_Name", "Number of firms": "Number_of_firms", 
                                "EV/EBIT (1-t)": "(EV/EBIT)(1-t)"})

# Displaying the top records of the dataset

CS_1.head()


# In[ ]:


# Checking for no. of null values in each column

print(CS_1.isna().sum())


# In[ ]:


# Figuring out the categorical columns and numerical columns

CS_1.info()


# In[ ]:


# Separating the Industry Name from the dataset and storing in y variable for labling

y = CS_1.Industry_Name.copy()
y.value_counts()


# In[ ]:


# Labling the Industry names using LabelEncoder

labelEncoder = LabelEncoder()
labelEncoder.fit(y)
y = labelEncoder.transform(y)
y


# In[ ]:


# Separating the Numerical variables for initiating the clustering

X = CS_1.iloc[:,1:6].copy()
X.head()


# In[ ]:


# Instantiating the k-means and fitting X

kmeans = KMeans(n_clusters=3, max_iter=300, random_state=42)
kmeans.fit(X)


# In[ ]:


# Labels of the initial clustering

labels = kmeans.labels_
labels


# In[ ]:


# Check the stats of the data

X.describe()


# In[ ]:


# MinMax Scaling - giving equal weight to all the features

scaler = MinMaxScaler()
X['Number_of_firms'] = scaler.fit_transform(X['Number_of_firms'].values.reshape(-1,1))
X['EV/EBITDAR&D'] = scaler.fit_transform(X['EV/EBITDAR&D'].values.reshape(-1,1))
X['EV/EBITDA'] = scaler.fit_transform(X['EV/EBITDA'].values.reshape(-1,1))
X['EV/EBIT'] = scaler.fit_transform(X['EV/EBIT'].values.reshape(-1,1))
X['(EV/EBIT)(1-t)'] = scaler.fit_transform(X['(EV/EBIT)(1-t)'].values.reshape(-1,1))


# In[ ]:


# Check the stats of the data after scaling

X.describe()


# In[ ]:


# Re-running clustering on scaled data

kmeans = KMeans(n_clusters=3, max_iter=300, random_state=42)
kmeans.fit(X)


# In[ ]:


# Labels of the scaled clustering

labels = kmeans.labels_
labels


# In[ ]:


# Identifying the optimum number of clusters using the elbow method

plt.plot([KMeans(n_clusters=k).fit(X).inertia_ for k in range(1,10)])


# In[ ]:


# More verbose code

sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=100).fit(X)
    X["clusters"] = kmeans.labels_
    sse[k] = kmeans.inertia_ #Inertia: Sum of distances of samples to their closest cluster center


# In[ ]:


# Plotting SSE against Number of clusters

plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show


# In[ ]:


# SSE values

sse


# In[ ]:


# From seeing the change in graph and values of SSE against Number of clusters, there is no significant change after 4
# Re-running clustering with n_clusters=4

kmeans = KMeans(n_clusters=4, max_iter=300, random_state=42)
kmeans.fit(X)


# In[ ]:


# Labels after optimum clusters

labels = kmeans.labels_
labels


# In[ ]:


# Grouping and displaying the 4 clustered industries

invbank_clust1 = pd.DataFrame(columns=['Industry Name','Cluster'])
invbank_clust1['Industry Name'] = CS_1.Industry_Name.values
invbank_clust1['Cluster'] = kmeans.fit(X).labels_
invbank_clust1.sort_values(by='Cluster', ascending=True)


# ***The industries can be clustered into 4 different clusters to assign Relationship Managers accordingly.***
