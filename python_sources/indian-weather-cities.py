#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


os.listdir("../input/zip_file/zip_file")


# In[ ]:


PATH ="../input/zip_file/zip_file"


# In[ ]:


file_data_list=pd.DataFrame()
classes_names_list=[]
target_column=[]
target_column = np.ones((42711,), dtype='int64')
dataset_size=[]


# In[ ]:


data_dir_list=os.listdir(PATH)
print(data_dir_list)


# In[ ]:


len(target_column)


# In[ ]:


data_dir_list=os.listdir(PATH)

for dataset in data_dir_list:
    classes_names_list.append(dataset) 
    print ('Loading data from {} folder\n'.format(dataset))
    temp=pd.read_csv(PATH+'/'+dataset)
    print("length of the dataset = ",temp.shape[0])
    file_data_list=file_data_list.append(temp, ignore_index = True)
    dataset_size.append(temp.shape[0])
for i in range(len(data_dir_list)):
    target_column[(dataset_size[i]*i):(dataset_size[i]*(i+1))]=i
    print(i)


# In[ ]:


dataset_size


# In[ ]:


file_data_list.shape


# In[ ]:


value,counts=np.unique(target_column,return_counts=True)
np.asarray((value, counts)).T


# In[ ]:


file_data_list.head()


# In[ ]:


file_data_list['date_time'] = pd.to_datetime(file_data_list.date_time,format='%d-%m-%Y') 


# In[ ]:


file_data_list['month']=file_data_list.date_time.dt.month


# In[ ]:


file_data_list['target_column']=target_column


# In[ ]:


np.unique(file_data_list['target_column'])


# In[ ]:


X=pd.DataFrame()
X['cities']=np.unique(file_data_list['target_column'])


# In[ ]:


X['tempC']=file_data_list.groupby(('target_column')).mean()['tempC']


# In[ ]:


X['sunHour']=file_data_list.groupby(('target_column')).mean()['sunHour']


# In[ ]:


X['precipMM']=file_data_list.groupby(('target_column')).mean()['precipMM']


# In[ ]:


X.drop(columns='cities',inplace=True)


# In[ ]:


X.head()


# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


wcss=[]
for i in range(1, 11):
   kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
   kmeans.fit(X)
   wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# In[ ]:


kmeans.cluster_centers_


# In[ ]:


kmeans.cluster_centers_[:,0]


# In[ ]:


X['y_kmeans']=y_kmeans


# In[ ]:


X.head()


# In[ ]:


y_kmeans.shape


# In[ ]:


import seaborn as sns


# In[ ]:


sns.scatterplot(X.tempC,X.sunHour,hue=X.y_kmeans,)


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


centers =kmeans.cluster_centers_[:,0:2]
centers


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(X.tempC,X.sunHour,c=X.y_kmeans,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.show()


# In[ ]:


kmeans.cluster_centers_[:,0:2]


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X.iloc[:,0:3])
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal_component_1', 'principal_component_2'])


# In[ ]:


finalDf = pd.concat([principalDf, X[['y_kmeans']]], axis = 1)


# In[ ]:


finalDf.head()


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(finalDf.principal_component_1,finalDf.principal_component_2,c=finalDf.y_kmeans,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.show()


# In[ ]:


from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(X.iloc[:,0:3])
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal_component_1', 'principal_component_2'])


# In[ ]:


principalDf.head()


# In[ ]:


wcss=[]
for i in range(1, 11):
   kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
   kmeans.fit(principalDf)
   wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[ ]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(finalDf)


# In[ ]:


centers =kmeans.cluster_centers_[:,0:2]
centers


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(finalDf.principal_component_1,finalDf.principal_component_2,c=y_kmeans,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.show()


# In[ ]:


cities=[]
for j in range(0,len(os.listdir(PATH))):
    newstring=str()
    for i in range(0,len(os.listdir(PATH)[j])-4):
        newstring=newstring+os.listdir(PATH)[j][i]
    cities.append(newstring)


# In[ ]:


cities


# In[ ]:


fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(finalDf.principal_component_1,finalDf.principal_component_2,c=y_kmeans,s=50)
for i,j in centers:
    ax.scatter(i,j,s=50,c='red',marker='+')
for i, txt in enumerate(cities):
    ax.annotate(txt, (finalDf.principal_component_1[i], finalDf.principal_component_2[i]))
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.colorbar(scatter)

fig.show()

