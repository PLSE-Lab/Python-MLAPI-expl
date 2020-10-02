#!/usr/bin/env python
# coding: utf-8

# In this excercise we will use the KMeans Clustering algorithm to predict the output of unlabelled data(IRIS Dataset).
# Here, first to explain the algorithm we will take only the "sepal_length" and "sepal_width" and then we will predict to which clusters or group(i.e Output) these data belongs to. After that we will take the entire data to predict the clusters or group they belongs to.
# 
# First we will load the necessary library for this excercise.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


# In[ ]:


data=pd.read_csv('../input/iris-flower-dataset/IRIS.csv')
data.head()


# In[ ]:


sepal=data.drop(['petal_length','petal_width','species'],1)
sepal.head()


# As you can see, only "sepal_length" and "sepal_width" are taken from the entire dataset using the drop function.

# Now, we will use scatterplot to plot the input data so as to visualize how these data are scattered.

# In[ ]:


plt.scatter(sepal['sepal_length'],sepal['sepal_width'])


# Now using KMeans algorithm we will predict to which group of cluster these data belongs to.
# Note: Here we have taken n_clusters=3 but we will later find out how to find the optimal value of n_clusters.

# In[ ]:


km=KMeans(n_clusters=3)
km


# In[ ]:


y_pred=km.fit_predict(sepal)
y_pred


# These are the output or you can say assigned value of clusters for each input data.
# For example for the first row of data (i.e sepal_length=5.1 and sepal_width=3.5) it is assigned to the 1st cluster(i.e "0" cluster).

# In[ ]:


sepal['cluster']=y_pred
sepal


# Here is the complete Dataframe of input and output(clustered group).

# Now we will visualize the entire data so as to define to which clusters the data belongs to using different colors for different clusters.
# Here for cluster 0 the color is Green, for cluster 1 the color is Red,
# and for cluster 2 the color is Yellow.

# In[ ]:


sep1=sepal[sepal.cluster==0]
sep2=sepal[sepal.cluster==1]
sep3=sepal[sepal.cluster==2]

plt.scatter(sep1['sepal_length'],sep1['sepal_width'],color='green')
plt.scatter(sep2['sepal_length'],sep2['sepal_width'],color='red')
plt.scatter(sep3['sepal_length'],sep3['sepal_width'],color='yellow')

plt.xlabel('sepal_length')
plt.ylabel('sepal_width')


# As the data is grouped into different clusters, you may want to know the centre of each clusters. Or you can say the value of centroid in KMeans, so that we can plot it against the data.

# In[ ]:


centroid=km.cluster_centers_
centroid


# In[ ]:


plt.scatter(sep1['sepal_length'],sep1['sepal_width'],color='green')
plt.scatter(sep2['sepal_length'],sep2['sepal_width'],color='red')
plt.scatter(sep3['sepal_length'],sep3['sepal_width'],color='yellow')
plt.scatter(centroid[:,0],centroid[:,1],color='blue',marker='*')
plt.xlabel('sepal_length')
plt.ylabel('sepal_width')


# Here, the blue star mark represents the centre of each clusters or group.

# As we have discussed earlier that we have to find the optimal value of K or n_clusters, there is one method called Elbow method which is a perfect fit for this optimization.
# Here Elbow method uses the sum of squared error of the input data. i.e total sum of squared error of each point from the centre of each group. After that we can find the optimal n_clusters from the "elbow" of the graph which shows number of clusters having low sum of squared error.

# In[ ]:


k_rng=range(1,10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(sepal)
    sse.append(km.inertia_)
    
sse


# In[ ]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# Here, the "elbow" of the graph can be seen at K=3. Hence, we can take the value of n_clusters=3. 
# Note: By default the value of n_clusters is 8 so you can decide the optimal value after visualizing the graph.

# Now we will take the entire data,(i.e sepal_length,sepal_width,petal_length and petal_width) and then predict the classes they belongs to. 

# In[ ]:


data['species'].value_counts()


# Here we can see we have only three classes as output(i,e Iris-versicolor, Iris-virginica and Iris-setosa)

# In[ ]:


new_data=data.drop('species',1)
new_data.head()


# Now we will find the optimal K value(or value of n_clusters) although we know that would be 3 as output has 3 classes.

# In[ ]:


k_rng=range(1,10)
sse=[]
for k in k_rng:
    km=KMeans(n_clusters=k)
    km.fit(new_data)
    sse.append(km.inertia_)
    
sse


# In[ ]:


plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)


# In[ ]:


km=KMeans(n_clusters=3)
km


# In[ ]:


prediction=km.fit_predict(new_data)
prediction


# In[ ]:


data['predicted']=prediction
data


# Here the predicted clusters are shown with the output but to compare these values we will later transform them into their respective class name. But first to check the accuracy of the prediction value we will use confusion matrix. 

# In[ ]:


centroids=km.cluster_centers_
centroids


# In[ ]:


data1=data.copy()
data1["species"]=data1["species"].map({'Iris-versicolor':0,'Iris-setosa':1,'Iris-virginica':2}).astype(int)
data1['predicted']=prediction
data1


# Here the class name are converted to respective integer value(i.e 0-''Iris-versicolor'',1-'Iris-setosa',2-'Iris-virginica') to check the confusion matrix

# In[ ]:


from sklearn.metrics import confusion_matrix
confusion_matrix(data1['species'],prediction)


# From the confusion matrix we can see that out of all 50 data of class Iris-versicolor 48 are correctly predicted whereas 2 are wrongly predicted as Iris-virginica. For 50 data of Iris-setosa all are wcorrectly predicted. And for 50 data of Iris-virginica 36 are correctly predicted whereas 14 are wrongly predicted as Iris-versicolor.

# In[ ]:


data['predicted']=prediction
data.head()


# In[ ]:



data["predicted"]=data["predicted"].map({0:'Iris-versicolor',1:'Iris-setosa',2:'Iris-virginica'})
data


# In[ ]:




