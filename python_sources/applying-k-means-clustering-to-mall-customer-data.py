#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Trying to segment unstructured data with K-Means clustering

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

df=pd.read_csv('../input/Mall_Customers.csv', index_col=0)


# In[ ]:


df.head(5)


# In[ ]:


df.describe()


# ##Exploratory Data Visualisations

# In[ ]:


#scatterplot of Annual Income Vs Spending Score against Gender as hue
sns.set_style('whitegrid')
sns.lmplot("Annual Income (k$)", "Spending Score (1-100)", data=df, hue='Gender', palette='coolwarm',size=6, aspect=1, fit_reg=False)


# In[ ]:


#Distribution of Spendng Score by Gender in a histogram
sns.set_style='darkgrid'
dg=sns.FacetGrid(df,hue='Gender', palette='coolwarm', size=6, aspect=2)
dg=dg.map(plt.hist, 'Spending Score (1-100)', bins=30, alpha=0.7)


# # Error Checks on Age and Score

# In[ ]:


df[df['Age']>99]


# In[ ]:


df[df['Spending Score (1-100)']>100]


# # K-Means Cluster creation

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans=KMeans(n_clusters=3) # Gender has 2 values only


# In[ ]:


kmeans.fit(df.drop('Gender', axis=1))


# In[ ]:


kmeans.cluster_centers_


# # Evaluate and assign Clusters

# In[ ]:


def assign(cluster):
    if cluster<20:
        return 1
    elif cluster>40:
        return 3
    else:
        return 2


# In[ ]:


df['Cluster'] = df['Age'].apply(assign)


# In[ ]:


df.tail()


# In[ ]:


from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(df['Age'], kmeans.labels_))
print(classification_report(df['Age'], kmeans.labels_))


# In[ ]:




