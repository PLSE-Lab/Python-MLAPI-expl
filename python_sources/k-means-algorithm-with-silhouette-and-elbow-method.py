#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats
import pylab
import seaborn as sns


# In[ ]:


data_path="../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv"


# In[ ]:


data=pd.read_csv(data_path)


# In[ ]:


data.head()


# In[ ]:


gender_dataframe=pd.get_dummies(data.Gender)


# In[ ]:


gender_dataframe.head(2)


# In[ ]:


data.drop(["Gender"],axis=1,inplace=True)


# In[ ]:


data.insert(loc=2,column="Male",value=gender_dataframe.Male)


# In[ ]:


data.head()


#  **Data Analysis**

# In[ ]:


data.isnull().sum()##checking the null value


# In[ ]:


data.describe()


# In[ ]:





# Data Distribution Plot

# In[ ]:


plt.hist(data.Age,color="#41b098")
plt.title("Age Distribution throughout the dataset")
data["Age"].plot.kde()
plt.show()


# In[ ]:


sns.countplot(data.Age)
plt.rcParams['figure.figsize'] = (20, 10)
plt.title("Distribution of the Age")
plt.show()


# Now lets check how good this data is normally distributed using "scipy.stats"

# **QQ-Plot**

# In[ ]:


scipy.stats.probplot(data.Age,plot=pylab)
pylab.show()


# In[ ]:


from pandas.plotting import andrews_curves


# In[ ]:


plt.hist(data["Annual Income (k$)"],color="#41b098")
plt.title("Anual Income Distribution throughout the dataset")
data["Annual Income (k$)"].plot.kde()
plt.show()


# In[ ]:


sns.countplot(data["Annual Income (k$)"])
plt.rcParams['figure.figsize'] = (20, 7)
plt.title("Distribution of the Salary")
plt.show()


# In[ ]:


scipy.stats.probplot(data["Annual Income (k$)"],plot=pylab)
pylab.show()


# In[ ]:


#identifying the number of the Males and Females
males=data.Male.sum()
females=200-males


# In[ ]:


#creating Pie Chart for the 
color=["#80ff00", "#4245a8"]
plt.pie([males,females],labels=["Males","Females"],explode = (0, 0.1,),shadow=True,startangle=360,radius=1.5,colors=color,autopct='%1.1f%%')
fig=plt.gcf()
plt.show()


# **Pair plot to show how each feature is affected by other features**

# In[ ]:


sns.pairplot(data)
plt.show()


# **Starting clustering the Data into multiple groups**

# In[ ]:


data.head()


# In[ ]:


features=data.iloc[:,1:]


# In[ ]:


features.head()


# Identitfy the optimal number of cluster using Elbow-"Inertia" Method

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


intertia=[]
index_number=[]

for i in range(1,15):
  kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
  kmeans.fit(features)
  intertia.append(kmeans.inertia_)
  index_number.append(i)


# In[ ]:


plt.plot(intertia)
plt.show()


# **It looks like ideal number of cluster is 4, however i am not sure, so lets use silhouette score to identify the optimal number of cluster**

# In[ ]:


from sklearn.metrics import silhouette_score


# In[ ]:


sil_score=[]
for n_clusters in range(2,12):
    clusterer = KMeans (n_clusters=n_clusters).fit(features)
    preds = clusterer.predict(features)
    centers = clusterer.cluster_centers_

    score = silhouette_score (features, preds, metric='euclidean')
    sil_score.append(score)
    


# In[ ]:


plt.plot(sil_score)
plt.title("Silhouette score vs No.Of cluste")
plt.show()


# **As confirmed by the silhouette analysis as well, the optimal number of categories=4, so I am going to use kmeans with cluster=4**

# In[ ]:


kmeans=KMeans(n_clusters=4,max_iter=300)
kmeans.fit(features)


# In[ ]:


class_per_data=kmeans.labels_


# In[ ]:


print("Classes are",class_per_data)

