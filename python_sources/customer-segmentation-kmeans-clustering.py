#!/usr/bin/env python
# coding: utf-8

# # CUSTOMER SEGMENTATION

# ## IMPORT THE NECESSARY LIBRARIES

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


data = pd.read_csv('../input/customer-segmentation-tutorial-in-python/Mall_Customers.csv')
data.head()


# ## CHECK FOR MISSING VALUES IN THE DATA

# In[ ]:


print(data.isnull().sum())


# In[ ]:


data.info()


# # VISUALIZATIONS

# In[ ]:


sns.heatmap(data.corr(),annot = True)


# ## HEATMAP IN COMBINATION WITH CORRELATION 
# - THERE IS A STRONG CORRELATION BETWEEN THE INPUT FEATURES 'CustomerID' & 'Annual Income'
# - HOWEVER THE 'CustomerID' & 'Annual Income' ARE ARRANGED IN ASCENDING ORDER
# - HENCE THEY ARE HAVING A STRONG CORRELATION BETWEEN THEM
# - THUS, WE CAN DROP 'CustomerID' 
# - OTHERWISE THE FEATURES ARE NOT DEPENDENT ON EACH OTHER 

# In[ ]:


data = data.drop(columns = 'CustomerID')
data.head()


# ## PLOTS w.r.t GENDER 

# In[ ]:


df = data.groupby(data['Gender']).sum()
df.head()


# In[ ]:


male = [i for i in data['Gender'] if(i == 'Male')]
female = [i for i in data['Gender'] if(i == 'Female')]
size = []
size.append(len(male)/2)
size.append(len(female)/2)


# ## PERCENTAGE OF MEN & WOMEN IN THE DATA

# In[ ]:


plt.pie(size,labels = ['Male','Female'],autopct='%1.1f%%',startangle = 90,colors = ['Orange','Blue'])


# In[ ]:


sns.countplot(x = data['Gender'],data = data,order = ['Female','Male'])


# - NUMBER OF WOMEN IN THE DATA IS MORE THAN MEN

# ### SPENDING SCORE OF MEN VS WOMEN

# In[ ]:


sns.barplot(x = df.index,y = 'Spending Score (1-100)',data = df)


# In[ ]:


total_spend = df.iloc[0,2] + df.iloc[1,2]
spend = []
spend.append((df.iloc[0,2]/total_spend)*100)
spend.append((df.iloc[1,2]/total_spend)*100)


# In[ ]:


plt.pie(spend,labels = ['Female','Male'],autopct='%1.1f%%',startangle = 90,colors = ['Blue','Orange'])


# ### ANNUAL INCOME OF MEN VS WOMEN

# In[ ]:


sns.barplot(x = df.index,y ='Annual Income (k$)',data = df)


# In[ ]:


total_income = df.iloc[0,1] + df.iloc[1,1]
income = []
income.append((df.iloc[0,1]/total_income)*100)
income.append((df.iloc[1,1]/total_income)*100)


# In[ ]:


plt.pie(income,labels = ['Female','Male'],autopct='%1.1f%%',startangle = 90,colors = ['Blue','Orange'])


# - THE ABOVE PLOTS CLEARLY SHOW THAT WOMEN ACCOUNT MORE IN SPENDING SCORE & TOTAL INCOME
# - THIS IS CLEARLY BECAUSE OF THE MORE NUMBER OF WOMEN

# ## DISTRIBUTION OF DATA w.r.t AGE & GENDER

# ### SPENDING SCORE vs AGE

# In[ ]:


sns.jointplot('Age','Spending Score (1-100)',data=data,kind = 'kde')


# ### ANNUAL INCOME vs AGE

# In[ ]:


sns.jointplot('Age','Annual Income (k$)',data= data,kind = 'kde')


# ### AGE vs GENDER

# In[ ]:


sns.violinplot('Gender','Age',data=data,order= ['Female','Male'])


# ### SPENDING SCORE VS GENDER

# In[ ]:


sns.violinplot('Gender','Spending Score (1-100)',data = data,order = ['Female','Male'])


# ### ANNUAL INCOME vs GENDER

# In[ ]:


sns.violinplot('Gender','Annual Income (k$)',data = data,order = ['Female','Male'])


# ### SPENDING SCORE VS ANNUAL INCOME w.r.t GENDER

# In[ ]:


sns.scatterplot('Annual Income (k$)','Spending Score (1-100)',data = data,hue = 'Gender')


# # SEGMENTATION

# # 2 METHODS TO DECIDE THE NUMBER OF CLUSTERS FOR KMEANS
# ### 1] ELBOW METHOD:-
# - FROM THE GRAPH OF ELBOW METHOD,WE SELECT THE VALUE WHICH RESEMBLES 
# - LIKE AN ELBOW OF A HAND AS THE NUMBER OF CLUSTERS
# 
# ### 2] SILHOUETTE SCORE:-
# - IT IS THE METRIC WHICH TELLS US ABOUT HOW THE CLUSTER 
# - HAS GROUPED WELL WITH POINTS AROUND IT & HOW IT HAS BEEN 
# - DIFFERENTIATED WELL FROM OTHER CLUSTERS.
# - WE SELECT THE NUMBER OF CLUSTERS WHICH HAS THE HIGHEST SILHOUETTE SCORE. 

# ## 1] AGE & SPENDING SCORE

# In[ ]:


d1 = data[['Age','Spending Score (1-100)']].values


# ### A] ELBOW METHOD

# In[ ]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(d1)
    sse[k] = kmeans.inertia_ # Inertia: Sum of distances of samples to their closest cluster center
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# ### B] SILHOUETTE SCORE

# In[ ]:


sil = []
kmax = 10
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(d1)
    labels = kmeans.labels_
    sil.append(silhouette_score(d1, labels, metric = 'euclidean'))
plt.figure()
plt.plot(range(2,kmax+1),sil)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()


# ### CLUSTERS = 4

# In[ ]:


model = KMeans(n_clusters = 4,max_iter = 1000)
model.fit(d1)
cluster = model.cluster_centers_
centroids = np.array(cluster)
labels = model.labels_
plt.scatter(centroids[:,0],centroids[:,1], marker="X", color = 'b')
plt.scatter('Age','Spending Score (1-100)',c=labels,cmap ='rainbow',data = data)
plt.title('SPENDING SCORE VS AGE')
plt.xlabel('AGE')
plt.ylabel('SPENDING SCORE')


# ## 2] ANNUAL INCOME & SPENDING SCORE

# In[ ]:


d2 = data[['Annual Income (k$)' , 'Spending Score (1-100)']].values


# ### A] ELBOW METHOD

# In[ ]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(d2)
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# ### B] SILHOUETTE SCORE

# In[ ]:


sil = []
kmax = 10
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(d2)
    labels = kmeans.labels_
    sil.append(silhouette_score(d2, labels, metric = 'euclidean'))
plt.figure()
plt.plot(range(2,kmax+1),sil)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()


# ### CLUSTERS = 5

# In[ ]:


model = KMeans(n_clusters = 5,max_iter = 1000)
model.fit(d2)
cluster = model.cluster_centers_
centroids = np.array(cluster)
labels = model.labels_
plt.scatter(centroids[:,0],centroids[:,1], marker="X", color = 'b')
plt.scatter('Annual Income (k$)','Spending Score (1-100)',c=labels,cmap ='rainbow',data = data)
plt.title('SPENDING SCORE VS ANNUAL INCOME')
plt.xlabel('ANNUAL INCOME')
plt.ylabel('SPENDING SCORE')


# - ## 3] AGE,ANNUAL INCOME & SPENDING SCORE

# In[ ]:


d3 = data[['Age','Annual Income (k$)','Spending Score (1-100)']].values


# ### A] ELBOW METHOD

# In[ ]:


sse = {}
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, max_iter=1000).fit(d3)
    sse[k] = kmeans.inertia_
plt.figure()
plt.plot(list(sse.keys()), list(sse.values()))
plt.xlabel("Number of cluster")
plt.ylabel("SSE")
plt.show()


# ### B] SILHOUETTE SCORE

# In[ ]:


sil = []
kmax = 10
for k in range(2, kmax+1):
    kmeans = KMeans(n_clusters = k).fit(d3)
    labels = kmeans.labels_
    sil.append(silhouette_score(d3, labels, metric = 'euclidean'))
plt.figure()
plt.plot(range(2,kmax+1),sil)
plt.xlabel("k")
plt.ylabel("Silhouette Score")
plt.show()


# ### CLUSTERS = 6

# In[ ]:


model = KMeans(n_clusters = 6,max_iter = 1000)
model.fit(d3)
cluster = model.cluster_centers_
centroids = np.array(cluster)
labels = model.labels_


# In[ ]:


fig = plt.figure()
ax = Axes3D(fig)
x = np.array(data['Annual Income (k$)'])
y = np.array(data['Spending Score (1-100)'])
z = np.array(data['Age'])
ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],marker="X", color = 'b')
ax.scatter(x,y,z,c=y)
plt.title('SPENDING SCORE VS ANNUAL INCOME VS AGE')
ax.set_xlabel('ANNUAL INCOME')
ax.set_ylabel('SPENDING SCORE')
ax.set_zlabel('AGE')


# # END
