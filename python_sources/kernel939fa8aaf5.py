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
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df = pd.read_csv('../input/Mall_Customers.csv')


# In[ ]:


df.head()


# In[ ]:


df.rename(columns={"Annual Income (k$)":"AIncome","Spending Score (1-100)":"Score"},inplace=True)


# In[ ]:


df.head()


# In[ ]:


df.shape


# In[ ]:


df.isna().sum()


# In[ ]:


sns.countplot(data=df, x='Gender')


# In[ ]:


plt.hist(data=df,x='Age',bins=[10,20,30,40,50,60,70,80],color='Green')
plt.xlabel('Age')


# In[ ]:


plt.hist(data=df,x='AIncome',bins=[10,20,30,40,50,60,70,80,90,100,110,120,130,140],color='Grey')


# # Modelling our data

# In[ ]:


X = df.drop(columns=['CustomerID', 'Gender', 'AIncome'])
X.head()


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans = KMeans(n_clusters=4).fit(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# In[ ]:


x=df['Age']
y=df['Score']

plt.scatter(x,y,c=labels)
plt.scatter(centroids[:,0],centroids[:,1],color='red')
plt.xlabel('Age')
plt.ylabel('Spending Score')


# In[ ]:


X2 = df.drop(columns=['CustomerID','Gender','Age'])
X2.head()


# In[ ]:


kmeans2 = KMeans(n_clusters=4).fit(X2)
labels2 = kmeans2.labels_
centroid2 = kmeans2.cluster_centers_


# In[ ]:


x2 = df['AIncome']
y2 = df['Score']

plt.scatter(x2,y2,c=labels2)
plt.scatter(centroid2[:,0],centroid2[:,1],color='red')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')


# In[ ]:




