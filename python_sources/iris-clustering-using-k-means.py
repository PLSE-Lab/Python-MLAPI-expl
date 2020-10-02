#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
data.head()


# we are taking petal length and petal width for clustering 

# In[ ]:


petal=data.drop(['sepal_length','sepal_width','species'],1)
petal


# In[ ]:


plt.scatter(petal['petal_length'],petal['petal_width'])


# thats how petal data is scattered 
# by using k means algoritham we are clustering the petal data

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kmeans=KMeans(n_clusters=3)
kmeans


# In[ ]:


kmeans.fit(petal)


# In[ ]:


pred=kmeans.predict(petal)
pred


# In[ ]:


centroids=kmeans.cluster_centers_
centroids


# In[ ]:


petal['CLUSTERING']=pred
petal


# visualize the data after clustering

# In[ ]:


petal1=petal[petal.CLUSTERING==0]
petal2=petal[petal.CLUSTERING==1]
petal3=petal[petal.CLUSTERING==2]

plt.scatter(petal1['petal_length'],petal1['petal_width'],color='green')
plt.scatter(petal2['petal_length'],petal2['petal_width'],color='red')
plt.scatter(petal3['petal_length'],petal3['petal_width'],color='yellow')
plt.scatter(centroids[:,0],centroids[:,1],color='blue',marker='*')


plt.xlabel('petal_length')
plt.ylabel('petal_width')


# by using the elbow method lets check the n_clusters value 

# In[ ]:


distortions=[]
K=range(1,10)
for k in K:
    kmeans=KMeans(n_clusters=k)
    kmeans.fit(petal)
    distortions.append(kmeans.inertia_)
    
distortions


# In[ ]:


plt.plot(K,distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method showing the optimal k')
plt.show()


# the above graph shows that elbow formed at k=3 so hence we took n_clusters=3 is  good 

# model has prepared for clustering the data now lets take the sepal data also to cluster 

# In[ ]:


data['species'].value_counts()


# In[ ]:


fulldata=data.drop('species',1)
fulldata.head()


# In[ ]:


KMNS=KMeans(n_clusters=3)
KMNS


# In[ ]:


KMNS.fit(fulldata)


# In[ ]:


prediction=KMNS.predict(fulldata)
prediction


# In[ ]:


data['predicted']=prediction
data


# In[ ]:


centroids=KMNS.cluster_centers_
centroids


# In[ ]:


data1=data.copy()
data1["species"]=data1["species"].map({'Iris-versicolor':0,'Iris-setosa':1,'Iris-virginica':2}).astype(int)
data1['predicted']=prediction
data1


# In[ ]:


data['predicted']=prediction
data


# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(data1['species'],prediction)


# Iris-versicolor 48 are correctly predicted  2  wrongly predicted .50 data of Iris-setosa are wcorrectly predicted. 50 data of Iris-virginica 36 are correctly predicted whereas 14 are wrongly predicted .
# 
# 

# In[ ]:




