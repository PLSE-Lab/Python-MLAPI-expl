#!/usr/bin/env python
# coding: utf-8

# This is a simple example of KMeans clustering on Mall customers dataset for segregating the people depending upon their Annual income and Spending Score.

# # Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn import preprocessing

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Reading and understanding Data

# In[ ]:


raw_data = pd.read_csv('/kaggle/input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
raw_data.head()


# In[ ]:


data = raw_data.set_index(['CustomerID'])


# In[ ]:


mapping = { "Gender" : {"Male":0, "Female":1}} 
data.replace(mapping, inplace=True)


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.nunique()


# ### This provides us the number of unique values present in our data

# In[ ]:


data.isnull().sum()


# ### This shows we have no null values in our data hence can now move forward with plotting

# # Pair plotting of data

# In[ ]:


sns.pairplot(data)


# In[ ]:


# pair plot with Gender

sns.pairplot(data, hue = 'Gender', kind='reg')


# ### By above plots we can see Annual Income and Spending score are forming clusters, Age and Spending score also form rough cluster and linear regressions are not a good way for this set of data

# # Correlation Matrix

# ### I'll form corrlation matrix using 2-3 ways and you can pick the one you think is better for you

# In[ ]:


data.corr()


# ### Correlation Matrix using Matplotlib

# In[ ]:


plt.matshow(data.corr())
plt.colorbar()


# ### Correlation matrix using SNS

# In[ ]:


corr = data.corr()
sns.heatmap(corr, cmap='YlGnBu', linewidths='0.1')


# ### SNS Cluster Map

# In[ ]:


sns.clustermap(corr)


# In[ ]:


corr.style.background_gradient(cmap='coolwarm')


# ### By looking at Correlation matrix we can see that Age and Spending score are more linearly correlated than any other feature and Annual Income and Spending score are not at all linearly correlated 

# # Building Elbow method graph for identifying number of clusters

# In[ ]:


## Building The Elbow Method Graph with full data

wcss = []
for i in range(1,len(data)):
    kmeans = KMeans(i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
wcss


# In[ ]:


plt.plot(range(1,len(data)), wcss)


# ### Looking at this we can see major curve is around 20, so we'll plot Elbow Curve again for range 1 to 20

# In[ ]:


wcss = []
for i in range(1,20):
    kmeans = KMeans(i)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)
wcss


# In[ ]:


plt.plot(range(1,20), wcss)


# ## We'll Plot for k = 4,5,6 and check which value is best

# In[ ]:


kmeans = KMeans(4)
kmeans.fit(data)
clusters = data.copy()
clusters['prediction'] = kmeans.fit_predict(data)
clusters[:15]


# In[ ]:


plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'], c= clusters['prediction'], cmap='rainbow')
plt.ylabel('Annual Income (k$)')
plt.xlabel('Spending Score (1-100)')


# ### Here the cyan colour cluster could be divided into 2 parts

# ### There certainly can be another cluster for lower values of Spending Score so we'll try with k = 5

# In[ ]:


kmeans = KMeans(5)
kmeans.fit(data)
clusters = data.copy()
clusters['prediction'] = kmeans.fit_predict(data)
plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'], c= clusters['prediction'], cmap='rainbow')
plt.ylabel('Annual Income (k$)')
plt.xlabel('Spending Score (1-100)')


# In[ ]:


kmeans = KMeans(6)
kmeans.fit(data)
clusters = data.copy()
clusters['prediction'] = kmeans.fit_predict(data)
plt.scatter(data['Spending Score (1-100)'],data['Annual Income (k$)'], c= clusters['prediction'], cmap='rainbow')
plt.ylabel('Annual Income (k$)')
plt.xlabel('Spending Score (1-100)')


# ### This divided the middle cluster into further parts which is not required so K=5 looks like the best option hence we'll stop here as k=5 is the best approach

# # We can look at this cluster graph and infer that the dataset is divided into 5 clusters
# 
# ## 1. Low income - Low Spending Score
# ## 2. Low Income - High Spending Score
# ## 3. Moderate Income - Moderate Spending Score
# ## 4. High Income - Low Spending Score
# ## 5. High Income - High Spending Score
# 
# ### There are a few Outliers as well in the data and we can choose either to neglect those or keep them for now as this is a very small dataset so the chances of overfitting are high
