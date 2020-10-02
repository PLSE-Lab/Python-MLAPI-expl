#!/usr/bin/env python
# coding: utf-8

# In this kernel, I am going to build few clustering models. Clustering is used to group data points together into a group which we call as a cluster. Broadly, there are two types of clustering :
# 1. Hard Clustering(each point belongs to only one cluster)
# 2. Soft Clustering(each data point is associated with probabilistic values for each cluster)
# 
# We are going to use some common clustering algorithms like K-Means Clustering , Agglomerative Clustering and Gaussian Mixture.
# K-Means and Agglomerative are hard clustering methods whereas Gaussian Mixture is a soft clustering method.

# In[ ]:


# Load csv into data frame
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df = pd.read_csv('../input/Seed_Data.csv')
df.head() # first 5 rows


# In[ ]:


df.shape #(rows, columns)
# 210 data points


# In[ ]:


df.describe()


# In[ ]:


#  Are the features strongly related ? 
# To know this, take each column as dependent variable and try to predict this column
# from other columns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
cols = df.columns

for col in cols:
    X = df.drop([col], axis=1)
    y = pd.DataFrame(df.loc[:, col])
    X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42)
    reg = LinearRegression()
    reg.fit(X_train,y_train)
    score = reg.score(X_test,y_test)
    print('Score for {} as dependent variable is {}'.format(col,score))


# In[ ]:


# Visualize how each feature is related to another feature
pd.scatter_matrix(df, diagonal='kde', figsize=(16,9))


# Now, I am going to pick two columns whose graph gives us some cluster like distribution i.e. I am not going to choose graphs with linear distribution(P and LK). For example, A and A_Coef have a non-linear distribution. Let's work on that!

# In[ ]:


df_A = df[['A','A_Coef']]
df_A.head()


# In[ ]:


import matplotlib.pyplot as plt
df_A.plot('A','A_Coef',kind='scatter',figsize=(7,5))


# **How to find the number of clusters we need?**
# 
# I am going to use elbow method and draw a graph that can give us the rough idea on the number of clusters that best suits for our data set.

# In[ ]:


from sklearn.cluster import KMeans

no_of_clusters = range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in no_of_clusters]
score = [kmeans[i].fit(df_A).score(df_A) for i in range(len(kmeans))]
plt.plot(no_of_clusters,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# The above graph plots number of clusters against score. We can see that after 3, there isn't much increase in score.

# **K-Means Clustering on A and A_Coef**

# In[ ]:


kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(df_A)
cluster_labels = kmeans.predict(df_A)
kmeans.cluster_centers_
# 3 cluster centres


# In[ ]:


kmeans.labels_ 
# 0, 1, 2


# In[ ]:


plt.figure(figsize=(7,5))
plt.scatter(df_A['A'],df_A['A_Coef'],c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='red')
plt.title('3 Means Clustering')
plt.xlabel('A')
plt.ylabel('A_Coef')
plt.show()


# **Agglomerative Clustering**

# In[ ]:


from sklearn.cluster import AgglomerativeClustering
hac_clustering = AgglomerativeClustering(n_clusters=3).fit(df_A)
hac_clustering


# In[ ]:


plt.figure(figsize=(7,5))
plt.scatter(df_A['A'],df_A['A_Coef'],c=hac_clustering.labels_)
plt.title('3 Means Clustering')
plt.xlabel('A')
plt.ylabel('A_Coef')
plt.show()


# **K- Means Clustering on WK and LKG**

# In[ ]:


df.plot.scatter('WK','LKG')


# In[ ]:


df_LKG_WK = df[['LKG','WK']]
no_of_clusters=range(1,10)
kmeans = [KMeans(n_clusters=i) for i in no_of_clusters]
score = [kmeans[i].fit(df_LKG_WK).score(df_LKG_WK) for i in range(len(kmeans))]
score
plt.plot(no_of_clusters, score)


# In[ ]:


kmeans = KMeans(n_clusters=2,random_state=42)
kmeans.fit(df_LKG_WK)
kmeans.predict(df_LKG_WK)
kmeans.cluster_centers_


# In[ ]:


plt.scatter(df_LKG_WK.iloc[:,0],df_LKG_WK.iloc[:,1],c=kmeans.labels_)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1], color='red')
plt.title('2 Means Clustering')
plt.xlabel('WK')
plt.ylabel('LKG')
plt.show()


# **GaussianMixture(Soft Clustering) on LKG and WK**

# In[ ]:


from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components=2).fit(df_LKG_WK)
gmm_labels = gmm.predict(df_LKG_WK)
plt.scatter(df_LKG_WK.iloc[:,0],df_LKG_WK.iloc[:,1],c=gmm_labels)
plt.title('2 Means Clustering')
plt.xlabel('WK')
plt.ylabel('LKG')
plt.show()


# In[ ]:


# As we know that this soft clustering, we can find the probability
# with which each data point belongs to the two clusters
y_pred = gmm.predict_proba(df_LKG_WK)
y_pred[50] # Probability that that data point at row 50 belongs to cluster 1 is 0.99674903 and cluster 2 is 0.00325097


# In[ ]:


y_pred[100] #cluster 1 - 0.01608574, cluster 2 - 0.98391426

