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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


data=pd.read_csv("../input/camera_dataset.csv")
data.head()


# Here, we start by the correlation heat map, to find the correlation between attributes. The matrix used here, uses Pearson's correlation matrix, where the value for correlation lies between -1 and 1 , negative values suggesting a negative correlation between the attributes ( indirectly proportional) and positive values suggesting a positive corelation between attributes (directly proportional). 

# In[ ]:


df_corr=data.corr(method='pearson')
fig, ax = plt.subplots(figsize=(8,8))
sns.heatmap(df_corr, annot=True, ax=ax)
plt.title('Correlation for camera attributes')
plt.show()


# In[ ]:


fig= plt.figure(figsize=(15,15))
ax2=fig.add_subplot(331)
plt.scatter(data['Low resolution'], data['Max resolution'])
plt.title('Low ressolution & High Resolution')
plt.xlabel('Low Resolution')
plt.ylabel('Max Resolution')
ax2=fig.add_subplot(332)
plt.scatter(data["Effective pixels"], data['Max resolution'])
plt.title('Effective pixels & Max resolution ')
plt.xlabel('Effective pixels')
plt.ylabel('Max resolution')
ax2=fig.add_subplot(333)
plt.scatter(data['Release date'],data['Effective pixels'])
plt.title('Release date & Effective pixels')
plt.xlabel('Release date')
plt.ylabel('Effective pixels')
plt.show()


# In[ ]:


data.fillna(0, inplace=True)


# In[ ]:


from sklearn.cluster import KMeans
import sklearn.metrics as sm


# In[ ]:


chosen=['Release date','Max resolution','Low resolution','Effective pixels','Zoom wide (W)','Zoom tele (T)','Normal focus range','Macro focus range','Storage included','Weight (inc. batteries)','Dimensions','Price']
X=data[chosen].values
model = KMeans(n_clusters=2)
model.fit(X)


# In[ ]:


colormap = np.array(['pink','purple'])
plt.scatter(data['Zoom wide (W)'], data['Low resolution'],c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()


# In[ ]:


plt.scatter(data['Zoom wide (W)'], data['Weight (inc. batteries)'],c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()


# In[ ]:


fig2 = plt.figure(figsize=(15, 15))
ax6 = fig2.add_subplot(331)
plt.scatter(data['Effective pixels'], data['Max resolution'],c=colormap[model.labels_], s=40)
plt.title('effective pixels and max resolution')
plt.xlabel('Effective pixels')
plt.ylabel('Max resolution')
ax6 = fig2.add_subplot(332)
plt.scatter(data['Effective pixels'], data['Low resolution'],c=colormap[model.labels_], s=40)
plt.title('effective pixels and low resolution')
plt.xlabel('Effective pixels')
plt.ylabel('Low resolution')
ax6=fig2.add_subplot(333)
plt.scatter(data['Max resolution'], data['Low resolution'],c=colormap[model.labels_], s=40)
plt.title('max resolution and low resolution')
plt.xlabel('Max resolution')
plt.ylabel('Low resolution')
ax6=fig2.add_subplot(334)
plt.scatter(data['Zoom wide (W)'], data['Weight (inc. batteries)'],c=colormap[model.labels_], s=40)
plt.title('Zoom and weight')
plt.xlabel('Zoom')
plt.ylabel('weight')
ax6=fig2.add_subplot(335)
plt.scatter(data['Effective pixels'], data['Release date'],c=colormap[model.labels_], s=40)
plt.title('Effective pixels and release date')
plt.xlabel('Effective pixels')
plt.ylabel('Release date')
ax6=fig2.add_subplot(336)
plt.scatter(data['Max resolution'], data['Release date'],c=colormap[model.labels_], s=40)
plt.xlabel('Max resolution')
plt.ylabel('Release date')
ax6=fig2.add_subplot(337)
plt.scatter(data['Low resolution'], data['Release date'],c=colormap[model.labels_], s=40)
plt.xlabel('Low resolution')
plt.ylabel('Release date')
ax6=fig2.add_subplot(337)
plt.scatter(data['Weight (inc. batteries)'], data['Dimensions'],c=colormap[model.labels_], s=40)
plt.xlabel('Weight')
plt.ylabel('Dimensions')
plt.show()


# In[ ]:


data1=data.drop(['Model','Release date'],axis=1)


# **We plot the mean distance to the centroid as a function of K, leading to an elbow point, where the rate of decreasesharply shifts, giving a rough estimate to determine K **

# In[ ]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(data1)
    data1["clusters"] = kmeans.labels_
    #print(data["clusters"])
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("Distortion")
plt.show()


# **Since PCA is effected by scale, we need to scale features  in the data before applying PCA.**

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


x = StandardScaler().fit_transform(X)
print(x)


# covariance matrix  (similar to correlation matrix)

# In[ ]:


print('NumPy covariance matrix: \n%s' %np.cov(x.T))


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
principalDf


# In[ ]:


pca = PCA().fit(X)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# **Since, we observe that only 2 components account for over 95% of the data, it means, that we will be able to recover most of the essential characteristics with only 2 components.**

# **Next, we find the explained variance ratio. This is tells how much information can be attributed to each of the principal component. Since, we are converting a 13 dimensional data to 2 dimensions, we will loose some of the variance by doing so.**

# In[ ]:


print(pca.explained_variance_ratio_)


# **Above we find, that the first component contains about 62.3% of the variance and second about 29.2%, together about 91.5% of the information.**

# In[ ]:



plt.scatter(principalDf['principal component 1'],principalDf['principal component 2'],s=30,c='goldenrod',alpha=0.5)
plt.title('plotting both variables')
plt.xlabel('Principal component 1')
plt.ylabel('Principal component 2')
plt.show()


# In[ ]:


model = KMeans(n_clusters=3)
model.fit(principalDf)


# In[ ]:


colormap = np.array(['blue','red','yellow','orange','purple'])
plt.scatter(principalDf['principal component 1'], principalDf['principal component 2'],c=colormap[model.labels_], s=40)
plt.title('K Mean Classification')
plt.show()


# In[ ]:




