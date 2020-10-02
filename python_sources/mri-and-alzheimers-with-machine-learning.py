#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 

import os
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


data=pd.read_csv("../input/mri-and-alzheimers/oasis_cross-sectional.csv")


# In[ ]:


data.head()


# In[ ]:


data.tail()


# In[ ]:


data.Hand.unique()


# In[ ]:


data.drop(["Hand","Delay"],axis=1,inplace=True)


# In[ ]:


data.columns=["id","gender","age","education","socio_economic_status","mini_mental_state_examination","clinical_dementia_rating",
           "estimated_total_intracranial_volume","normalize_whole_brain_volume","atlas_scaling_factor"]


# In[ ]:


data.gender = [1 if each == "F" else 0 for each in  data.gender]


# In[ ]:


data.info()


# In[ ]:


data.isnull().sum()


# In[ ]:


def impute_median(series):
    return series.fillna(series.median())


# In[ ]:


data.education =data["education"].transform(impute_median)
data.socio_economic_status =data["socio_economic_status"].transform(impute_median)
data.mini_mental_state_examination =data["mini_mental_state_examination"].transform(impute_median)
data.clinical_dementia_rating =data["clinical_dementia_rating"].transform(impute_median)


# In[ ]:


data.head()


# In[ ]:


#visualize the correlation
plt.figure(figsize=(15,10))
sns.heatmap(data.iloc[:,0:10].corr(), annot=True,fmt=".0%")
plt.show()


# In[ ]:



plt.scatter(data['age'],data['atlas_scaling_factor'],color="red",label="Bad")
plt.xlabel('age')
plt.ylabel('atlas_scaling_factor')
plt.show()


# In[ ]:


# KMeans Clustering
data2 = data.loc[:,['age','atlas_scaling_factor']]
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters = 2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data['age'],data['atlas_scaling_factor'],c = labels)
plt.xlabel('age')
plt.ylabel('atlas_scaling_factor')
plt.show()


# In[ ]:


data["class"] = ["Normal" if (each > 0.8) & (each < 1.2) else "Abnormal" for each in  data.atlas_scaling_factor]
color_list = ['red' if i=='Abnormal' else 'green' for i in data.loc[:,'class']]


# In[ ]:


# cross tabulation table
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# In[ ]:


# inertia
inertia_list = np.empty(8)
for i in range(1,8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0,8),inertia_list,'-o')
plt.xlabel('Number of cluster')
plt.ylabel('Inertia')
plt.show()


# In[ ]:


data3 = data.drop('class',axis = 1)


# In[ ]:


data3 = pd.get_dummies(data3)


# In[ ]:


from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
scalar = StandardScaler()
kmeans = KMeans(n_clusters = 2)
pipe = make_pipeline(scalar,kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({'labels':labels,"class":data['class']})
ct = pd.crosstab(df['labels'],df['class'])
print(ct)


# In[ ]:


from scipy.cluster.hierarchy import linkage,dendrogram

merg = linkage(data3.iloc[0:50,:],method = 'single')
dendrogram(merg, leaf_rotation = 90, leaf_font_size = 6)
plt.show()


# In[ ]:


data2.head()


# In[ ]:


from sklearn.manifold import TSNE
model = TSNE(learning_rate=120)
transformed = model.fit_transform(data2)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list )
plt.xlabel('age')
plt.xlabel('atlas_scaling_factor')
plt.show()


# In[ ]:


data3.head()


# In[ ]:


# PCA
from sklearn.decomposition import PCA
model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print('Principle components: ',model.components_)


# In[ ]:


# PCA variance
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler,pca)
pipeline.fit(data3)

plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()


# In[ ]:


# apply PCA
pca = PCA(n_components = 2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:,0]
y = transformed[:,1]
plt.scatter(x,y,c = color_list)
plt.show()

