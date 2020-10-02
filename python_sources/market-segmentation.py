#!/usr/bin/env python
# coding: utf-8

# ### Import the relevant libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ### Load the data

# In[ ]:


data = pd.read_csv('/kaggle/input/3.12. Example.csv')
data.head()


# About the data:
# 
# Satisfaction is self-reported (on a scale of 1 to 10, 10 is the highest satisfcation).
# Satisfaction is discrete variable
# 
# Loyalty is measured based on the number of purhases per year and some other factors. It is continuous data type. Range is from -2.5 to +2.5, as the data is already standardized

# ### Plot the data

# In[ ]:


plt.scatter(data['Satisfaction'], data['Loyalty'])
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()


# ### Select the features

# In[ ]:


x = data.copy()


# ### Clustering

# In[ ]:


kmeans = KMeans(2)
kmeans.fit(x)


# ### Clustering results

# In[ ]:


clusters = x.copy()
clusters['cluster_pred'] = kmeans.fit_predict(x)


# In[ ]:


plt.scatter(clusters['Satisfaction'], clusters['Loyalty'], c=clusters['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()


# From above plot, we can see that at the satisfaction value of 6, almost there is a vertical separation line, this happened because kmeans has just considered satisfaction as independent variable, and left loyalty, as the values are not normalized, so the next step is to standardize the satisfaction (as loyalty is already standardized)

# ### Standardize the variable

# In[ ]:


from sklearn import preprocessing
x_scaled = preprocessing.scale(x)
x_scaled


# ### Elbow Method

# In[ ]:


wcss = []
for i in range(1, 10):
    kmeans = KMeans(i)
    kmeans.fit(x_scaled)
    wcss.append(kmeans.inertia_)
    
wcss


# In[ ]:


number_cluster = range(1, 10)
plt.plot(number_cluster, wcss)
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()


# As from above, we can take 2, 3, 4, and 5 as k values. It is not clear which one is the best, so we are gonna explore them below

# ### Explore clustering solutions and select the number of clusters

# In[ ]:


# Just change the parameter passed to the KMeans(2, 3, 4, 5)
kmeans_new = KMeans(4)
kmeans_new.fit(x_scaled)
clusters_new =  x.copy()
clusters_new['cluster_pred'] = kmeans_new.fit_predict(x_scaled)
clusters_new.head()


# In[ ]:


plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
plt.xlabel('Satisfaction')
plt.ylabel('Loyalty')
plt.show()

