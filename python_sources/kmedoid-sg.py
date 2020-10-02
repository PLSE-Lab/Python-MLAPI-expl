#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# * [<font size=4>Question 1:How to  fit kMedoids?</font>](#1)
# * [<font size=5>Question 2: How to calculate Silhouette score for a cluster?</font>](#2)   
# * [<font size=4>Question 3: How to use Silhouette score for finding optimal number of cluster?</font>](#3)   
# * [<font size=5>Question 4: How to calculate purity?</font>](#4)  
# * [<font size=6>Question 5: How extreme observations effect k-Medoids compared to K-Means</font>](#5)  

# ### Installing sklearn Extra

# In[ ]:


get_ipython().system('pip install https://github.com/scikit-learn-contrib/scikit-learn-extra/archive/master.zip')


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids


# # Question 1: How to fit kMedoids <a id="1"></a>

# ### loading Data

# In[ ]:


from sklearn import datasets
myiris = datasets.load_iris()
x = myiris.data
y = myiris.target


# In[ ]:


y.shape


# In[ ]:


y


# ### Scaling and Fitting KMedoids

# In[ ]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(x)
x_scaled = scaler.transform(x)
kMedoids = KMedoids(n_clusters = 3, random_state = 0)
kMedoids.fit(x_scaled)
y_kmed = kMedoids.fit_predict(x_scaled)


# In[ ]:


y_kmed


# In[ ]:


kMedoids.inertia_


# # Question 2: How to use Silhouette to evaluate cluster <a id="2"></a>

# ### Silhouette Width

# ![image.png](attachment:image.png)

# 
# ![image.png](attachment:image.png)

# In[ ]:


from sklearn.metrics import silhouette_samples, silhouette_score
kMedoids = KMedoids(n_clusters = 3, random_state = 0)
kMedoids.fit(x_scaled)
y_kmed = kMedoids.fit_predict(x_scaled)
silhouette_avg = silhouette_score(x_scaled, y_kmed)
print(silhouette_avg)


# In[ ]:


sample_silhouette_values = silhouette_samples(x_scaled, y_kmed)
for i in range(3):
    ith_cluster_silhouette_values = sample_silhouette_values[y_kmed == i]
    print(np.mean(ith_cluster_silhouette_values))


# # Question 3: How to use Silhouette Width to find number of cluster <a id="3"></a>

# In[ ]:


sw = []

for i in range(2, 11):
    kMedoids = KMedoids(n_clusters = i, random_state = 0)
    kMedoids.fit(x_scaled)
    y_kmed = kMedoids.fit_predict(x_scaled)
    silhouette_avg = silhouette_score(x_scaled, y_kmed)
    sw.append(silhouette_avg)
    


# In[ ]:


plt.plot(range(2, 11), sw)
plt.title('Silhoute Score')
plt.xlabel('Number of clusters')
plt.ylabel('SW')      #within cluster sum of squares
plt.show()


# # Question 4: How to compute Purity <a id="4"></a>

# ### Purity

# ![image.png](attachment:image.png)

# ![image.png](attachment:image.png)

# In[ ]:


from sklearn import metrics

def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix) 


# # Question 5: How extreme values effect k-Medoid compared to k-Means <a id="5"></a>

# ### KMeans on Iris

# In[ ]:


kmeans = KMeans(n_clusters = 3, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x_scaled)
purity_score(y,y_kmeans)


# ### Visualization

# In[ ]:


plt.scatter(x_scaled[y_kmeans == 0, 0], x_scaled[y_kmeans == 0, 1], s = 100, c = 'red', label = 'C1')
plt.scatter(x_scaled[y_kmeans == 1, 0], x_scaled[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'C2')
plt.scatter(x_scaled[y_kmeans == 2, 0], x_scaled[y_kmeans == 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()


# In[ ]:


kmedoids = KMedoids(n_clusters=3, random_state=0).fit(x_scaled)
y_kmed = kmedoids.fit_predict(x_scaled)
purity_score(y,y_kmed)


# In[ ]:


plt.scatter(x_scaled[y_kmed == 0, 0], x_scaled[y_kmed == 0, 1], s = 100, c = 'red', label = 'C1')
plt.scatter(x_scaled[y_kmed == 1, 0], x_scaled[y_kmed == 1, 1], s = 100, c = 'blue', label = 'C2')
plt.scatter(x_scaled[y_kmed == 2, 0], x_scaled[y_kmed == 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(kmedoids.cluster_centers_[:, 0], kmedoids.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()


# ### Adding Extreme Values

# In[ ]:


x


# In[ ]:


import numpy as np
m=np.append(x,[[10,10,10,10],[15,15,15,15],[12,12,12,12]],axis=0)
m.shape
y=np.append(y,[2,2,2])
y


# In[ ]:


m.shape


# In[ ]:


scaler = StandardScaler().fit(m)
x_scaled = scaler.transform(m)


# In[ ]:


kmeans = KMeans(n_clusters = 3, init = 'random', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(x_scaled)
purity_score(y,y_kmeans)


# In[ ]:


kmedoids = KMedoids(n_clusters=3, random_state=0).fit(x_scaled)
y_kmed = kmedoids.fit_predict(x_scaled)
purity_score(y,y_kmed)


# In[ ]:


plt.scatter(x_scaled[y_kmeans == 0, 0], x_scaled[y_kmeans == 0, 1], s = 100, c = 'red', label = 'C1')
plt.scatter(x_scaled[y_kmeans  == 1, 0], x_scaled[y_kmeans  == 1, 1], s = 100, c = 'blue', label = 'C2')
plt.scatter(x_scaled[y_kmeans  == 2, 0], x_scaled[y_kmeans== 2, 1], s = 100, c = 'green', label = 'C3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:,1], s = 100, c = 'yellow', label = 'Centroids')
plt.legend()


# In[ ]:


data = [['k-Means', 0.81], ['k-Means with Outliers', 0.67], ['k-Medoid', 0.84],['K Medoid with outliers', 0.86]] 
df = pd.DataFrame(data, columns = ['Method', 'Purity']) 


# In[ ]:


df.plot.bar(x='Method',y='Purity',title='Cluster Quality')

