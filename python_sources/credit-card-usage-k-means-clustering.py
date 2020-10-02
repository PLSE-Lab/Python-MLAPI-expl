#!/usr/bin/env python
# coding: utf-8

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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import preprocessing as pp
from sklearn.cluster import KMeans
import random 
from sklearn.datasets.samples_generator import make_blobs 
import pylab as pl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


card = pd.read_csv("../input/CreditCardUsage.csv")


# In[ ]:


card.head(5)


# In[ ]:


card.describe().T


# In[ ]:


card.isna().sum()


# In[ ]:


mean_value=card['CREDIT_LIMIT'].mean()
card['CREDIT_LIMIT']=card['CREDIT_LIMIT'].fillna(mean_value)


# In[ ]:


mean_value=card['MINIMUM_PAYMENTS'].mean()
card['MINIMUM_PAYMENTS']=card['MINIMUM_PAYMENTS'].fillna(mean_value)


# In[ ]:


card.corr()


# In[ ]:


card.cov()


# In[ ]:


a = card.corr()


# In[ ]:


plt.figure(figsize=(20,10))
sns.heatmap(a,vmin=-1,vmax=1,center=0,annot=True)


# In[ ]:


df = card.drop('CUST_ID', axis=1)
df.head(3)


# In[ ]:


from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# **K Means Modelling**

# In[ ]:


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=6,random_state=0)
kmeans.fit(df)


# In[ ]:


kmeans.labels_


# In[ ]:


Sum_of_squared_distances = []
K = range(1,21)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    Sum_of_squared_distances.append(km.inertia_)
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum_of_squared_distances')
plt.title('Elbow Method For Optimal k')
plt.show()


# 
# 
# The Elbow curve depicts sum of squared distances for each point from its respective centroid. Our goal is to check for a K value that has minimum sum of square distance.
# 

# **Silhouette Coefficient**

# In[ ]:


from sklearn.metrics import silhouette_score, silhouette_samples

for n_clusters in range(2,21):
    km = KMeans (n_clusters=n_clusters)
    preds = km.fit_predict(df)
    centers = km.cluster_centers_

    score = silhouette_score(df, preds, metric='euclidean')
    print ("For n_clusters = {}, silhouette score is {}".format(n_clusters, score))


# 
# 
# **K=3 has maximum Silhoutte score. Let us visualize Silhouette score for each cluster at k=3.**
# 

# In[ ]:


from yellowbrick.cluster import SilhouetteVisualizer

# Instantiate the clustering model and visualizer
km = KMeans (n_clusters=3)
visualizer = SilhouetteVisualizer(km)

visualizer.fit(df) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data


# In[ ]:


from yellowbrick.cluster import KElbowVisualizer
# Instantiate the clustering model and visualizer
km = KMeans (n_clusters=3)
visualizer = KElbowVisualizer(
    km, k=(2,21),metric ='silhouette', timings=False
)

visualizer.fit(df) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data


# In[ ]:


km_sample = KMeans(n_clusters=4)
km_sample.fit(df)


# In[ ]:


labels_sample = km_sample.labels_


# In[ ]:


df['label'] = labels_sample


# In[ ]:


sns.set_palette('Set2')
sns.scatterplot(df['BALANCE'],df['PURCHASES'],hue=df['label'],palette='Set1')


# label 0: Low balance and low purchases - Fine group
# 
# label 1: Low to moderate balance and high purchases - Carefree group
# 
# label 2: Moderate balance and moderate purchases - choosy group
# 
# label 3: Moderate to high balance and low purchases - Saving group**
# 
