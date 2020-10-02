#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly as py
from sklearn.manifold import TSNE
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
py.offline.init_notebook_mode(connected = True)


import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Data Exploration

# In[ ]:


data=pd.read_csv('../input/Mall_Customers.csv')


# In[ ]:


data.shape


# In[ ]:


data.head()


# In[ ]:


data.describe()


# In[ ]:


data.isnull().sum()


# As we could see there are no null values in data so we doesnt need to clean the data.

# In[ ]:


## Data Visualisation


# ## Data Visualisation

# In[ ]:


plt.style.use('fivethirtyeight')


# In[ ]:


plt.figure(1 , figsize = (15 , 6))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    n += 1
    plt.subplot(1 , 3 , n)
    plt.subplots_adjust(hspace =5, wspace = 0.5)
    sns.distplot(data[x] , bins = 40)
    plt.title('Distplot of {}'.format(x))
plt.show()


# In[ ]:


plt.figure(1 , figsize = (15 , 5))
sns.countplot(y = 'Gender' , data = data)
plt.show()


# In[ ]:


plt.figure(1 , figsize = (15 , 7))
n = 0 
for x in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
    for y in ['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']:
        n += 1
        plt.subplot(3 , 3 , n)
        plt.subplots_adjust(hspace = 0.5 , wspace = 0.5)
        sns.regplot(x = x , y = y , data = data)
        plt.ylabel(y.split()[0]+' '+y.split()[1] if len(y.split()) > 1 else y )
plt.show()


# In[ ]:


plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Age' , y = 'Annual Income (k$)' , data = data[data['Gender'] == gender] ,
                s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Age'), plt.ylabel('Annual Income (k$)') 
plt.title('Age vs Annual Income w.r.t Gender')
plt.legend()
plt.show()


# In[ ]:


plt.figure(1 , figsize = (15 , 6))
for gender in ['Male' , 'Female']:
    plt.scatter(x = 'Annual Income (k$)',y = 'Spending Score (1-100)' ,
                data = data[data['Gender'] == gender] ,s = 200 , alpha = 0.5 , label = gender)
plt.xlabel('Annual Income (k$)'), plt.ylabel('Spending Score (1-100)') 
plt.title('Annual Income vs Spending Score w.r.t Gender')
plt.legend()
plt.show()


# ## K-Means Clustering

# In[ ]:


data_1=data[['Age' , 'Annual Income (k$)' , 'Spending Score (1-100)']]


# In[ ]:


data_1.describe()


# In[ ]:


data_1.corr()


# As we could see that we have no strongly correlared features we can use standar algorithms with using dimensionality reduction.
# 

# Before we feed the data to the K-Means algorithm we would use standard scaler to scale the data.

# In[ ]:


X=data_1.values
#print(x)
scaler=StandardScaler()
X=scaler.fit_transform(X)


# In[ ]:


from sklearn.metrics import silhouette_samples, silhouette_score
x = list(range(2, 12))
y_std = []
for n_clusters in x:
    print("n_clusters =", n_clusters)
    
    kmeans = KMeans(init='k-means++', n_clusters = n_clusters, n_init=10,max_iter=300)
    kmeans.fit(X)
    clusters = kmeans.predict(X)
    silhouette_avg = silhouette_score(X, clusters)
    y_std.append(silhouette_avg)
    print("The average silhouette_score is :", silhouette_avg, "with Std Scaling")


# As we could see the maximum silhoutee score is at number of clusters is equal to 6 so we would try taking 6 clusters and try it.

# In[ ]:


kmeans=KMeans(n_clusters = 6 ,init='k-means++', n_init = 10 ,max_iter=300, 
                        tol=0.0001,  random_state= 111  , algorithm='elkan') 
kmeans.fit(X)
clusters=kmeans.predict(X)
#print(clusters)


# ## Interpretting and Visualizing the clusters

# In[ ]:


data_1['clusters']=clusters
data_1.head()


# In[ ]:


trace1 = go.Scatter3d(
    x= data_1['Age'],
    y= data_1['Spending Score (1-100)'],
    z= data_1['Annual Income (k$)'],
    mode='markers',
     marker=dict(
        color = data_1['clusters'], 
        size= 20,
        line=dict(
            color= data_1['clusters'],
            width= 12
        ),
        opacity=0.8
     )
)
data = [trace1]
layout = go.Layout(
#     margin=dict(
#         l=0,
#         r=0,
#         b=0,
#         t=0
#     )
    title= 'Clusters',
    scene = dict(
            xaxis = dict(title  = 'Age'),
            yaxis = dict(title  = 'Spending Score'),
            zaxis = dict(title  = 'Annual Income')
    )
)
fig = go.Figure(data=data, layout=layout)
py.offline.iplot(fig)


# Visualizing clusters in 2D using T-SNE

# In[ ]:


tsne = TSNE(n_components=2)
proj = tsne.fit_transform(X)

plt.figure(figsize=(10,10))
plt.scatter(proj[:,0], proj[:,1], c=clusters)
plt.title("Visualization of the clustering with TSNE", fontsize="25")


# ## Interpretting the cluster using the cluster centriods.

# ### Cluster 0

# In[ ]:


cluster_0=data_1[data_1['clusters']==0]


# In[ ]:


cluster_0.describe()


# By comparing the mean values of the clusters we could see that these customers are with medium annual income and medium spending score but there age is high.

# ### Cluster 1

# In[ ]:


cluster_1=data_1[data_1['clusters']==1]
cluster_1.describe()


# By comparing the mean values of the clusters we could see that these customers are with medium annual income and medium spending score but there age is low.

# ### Cluster 2
# 

# In[ ]:


cluster_2=data_1[data_1['clusters']==2]
cluster_2.describe()


# By comparing the mean values of the clusters we could see that these customers are with high annual income and high spending score and they are middle aged customers.

# ### Cluster 3

# In[ ]:


cluster_3=data_1[data_1['clusters']==3]
cluster_3.describe()


# By comparing the mean values of the clusters we could see that these customers are with high annual income and low spending score and they are middle aged customers.

# ### Cluster 4

# In[ ]:


cluster_4=data_1[data_1['clusters']==4]
cluster_4.describe()


# By comparing the mean values of the clusters we could see that these customers are with low annual income and low spending score and they are middle aged customers.

# ### Cluster 5

# In[ ]:


cluster_5=data_1[data_1['clusters']==5]
cluster_5.describe()


# By comparing the mean values of the clusters we could see that these customers are with low annual income and high spending score but there age is low.

# # Classifying the clusters using Support Vector Machine

# In[ ]:


X_1 = data_1.drop(columns=['clusters']) 
y = data_1['clusters'].values


# In[ ]:


scaler=StandardScaler()
X_scaled = scaler.fit_transform(X_1)


# Making a training and testing Split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size = 0.2,random_state = 0, stratify=y)


# Checking with Dummy Classifier First

# In[ ]:


from sklearn.dummy import DummyClassifier


# In[ ]:


dummy_biased = DummyClassifier(random_state=0)

dummy_biased.fit(X_train, y_train)


# In[ ]:


print("Baseline accuracy", (dummy_biased.score(X_test, y_test))*100)


# Checking the Accuracy of Logistic Regression

# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


lr = LogisticRegression(multi_class='auto')
lr.fit(X_train, y_train)
print("LogisticRegression_classifier accuracy:", (lr.score(X_test, y_test))*100)


# Using Linear Support Vector Machine

# In[ ]:


from sklearn.svm import LinearSVC

svc = LinearSVC()
svc.fit(X_train, y_train)
print("LinearSVC accuracy:", (svc.score(X_test, y_test))*100)


# Checking if using kernel trick the accuracy could be improved.

# In[ ]:


svc_rbf=SVC(kernel='rbf',gamma='auto')
svc_rbf.fit(X_train,y_train)
print("svc_rbf accuracy:", (svc_rbf.score(X_test, y_test))*100)

