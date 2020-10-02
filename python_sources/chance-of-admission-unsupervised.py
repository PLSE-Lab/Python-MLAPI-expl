#!/usr/bin/env python
# coding: utf-8

# # **Importing Libraries**

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# # **Loading and Visualizing Data**

# In[ ]:


corona_pd=pd.read_csv("../input/graduate-admissions/Admission_Predict_Ver1.1.csv")
corona_pd.head()


# In[ ]:


#Returns the  meta data of the dataset.
corona_pd.info()


# In[ ]:


#Returns the information like mean,max,min,etc., of the dataset.
corona_pd.describe()


# In[ ]:


#To remove the columns of the DataFrame in memory.
corona_pd.drop(["Serial No.","Chance of Admit "],axis=1,inplace=True)
corona_pd.head()


# In[ ]:


#Returns the sum of null values under each column.
corona_pd.isnull().sum()


# In[ ]:


#To check whether the row contains duplicate values or not.
corona_pd.duplicated()


# In[ ]:


#To plot a correlation matrix between features.
f,ax= plt.subplots(figsize=(10,10))
sns.heatmap(corona_pd.corr(),annot=True)


# You can see from the above visualization that each feature has a effect on every other features.

# # **Elbow Method**
# Used to find optimal number of clusters.

# In[ ]:


#To scale the values along columns.
scaler= StandardScaler()
corona_pd_scaled=scaler.fit_transform(corona_pd)


# In[ ]:


#To get the Within Cluster Sum of Squares(WCSS) for each cluster count to find the optimal K value(i.e cluster count).
scores=[]
for i in range(1,20):
    corona_means=KMeans(n_clusters=i)
    corona_means.fit(corona_pd_scaled)
    scores.append(corona_means.inertia_)


# In[ ]:


#Plotting the values obtained to get the optimal K-value.
plt.plot(scores,"-rx")


# At point 3 ,the graph looks like a elbow. So we choose this as our K value.

# # **K-MEANS Implementation**

# In[ ]:


#Applying K-means algorithm with the obtained K value.
corona_means=KMeans(n_clusters=3)
corona_means.fit(corona_pd_scaled)


# In[ ]:


#Returns an array with cluster labels to which it belongs.
labels=corona_means.labels_


# In[ ]:


#Creating a Dataframe with cluster centres(The example which is taken as center for each cluster)-If you are not familiar ,learn about k-means through the link given at last.
corona_pd_m=pd.DataFrame(corona_means.cluster_centers_,columns=corona_pd.columns)
corona_pd_m.head()


# In[ ]:


#Inverting the scaled values to original values to get a better view.
corona_cluster=scaler.inverse_transform(corona_pd_m)
corona_cluster=pd.DataFrame(corona_cluster,columns=corona_pd.columns)
corona_cluster.head()


# It's clear from the above table that the students at cluster 1 will get the admission for sure , the students at cluster 2 have intermediate chance of getting admission and the students at cluster 0 have almost no chance of getting admission.

# In[ ]:


#Concatenating the cluster labels.
corona_cluster=pd.concat([corona_pd,pd.DataFrame({"Cluster":labels})],axis=1)
corona_cluster.head()


# # **Principal Component Analysis (PCA)**
# Used to perform dimentionality reduction to have a better view of clusters of examples.

# In[ ]:


#Implementing pca with 3 components i.e 3d plot
corona_pca=PCA(n_components=3)
principal_comp=corona_pca.fit_transform(corona_pd_scaled)


# In[ ]:


principal_comp=pd.DataFrame(principal_comp,columns=['pca1','pca2','pca3'])
principal_comp.head()


# In[ ]:


principal_comp=pd.concat([principal_comp,pd.DataFrame({"Cluster":labels})],axis=1)


# In[ ]:


principal_comp.sample(5)


# In[ ]:


#Plotting the 2d-plot.
plt.figure(figsize=(10,10))
ax=sns.scatterplot(x='pca1',y='pca2',hue="Cluster",data=principal_comp ,palette=['red','green','blue'])
plt.show()


# You can understand the same from the above visualization which I mentioned earlier.

# In[ ]:


#Plotting the 3d-plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
sc=ax.scatter(xs=principal_comp['pca2'],ys=principal_comp['pca3'],zs=principal_comp['pca1'],c=principal_comp['Cluster'],marker='o')
plt.colorbar(sc)
plt.show()


# You can have a even better view from the above plot.

# For K-Means algorithm - Refer this [link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwiInYrt2cDqAhUbyzgGHaiTBHAQwqsBMAF6BAgKEAQ&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3DhDmNF9JG3lo&usg=AOvVaw0uqMZBuHXA-UDOHz-ymSfK)

# For PCA - Refer this [link](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjB1NOK28DqAhUjguYKHUNxDxwQwqsBMAB6BAgKEAQ&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3Drng04VJxUt4&usg=AOvVaw14_fwoAfo-0sWFezc-7qiy)

# Thank You!
