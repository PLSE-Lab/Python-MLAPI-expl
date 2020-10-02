#!/usr/bin/env python
# coding: utf-8

# ![4-watercolour-political-map-of-the-world-michael-tompsett.jpg](attachment:4-watercolour-political-map-of-the-world-michael-tompsett.jpg)

# **Problem**
# 
# X International is an international humanitarian NGO that is committed to fighting poverty and providing the people of backward countries with basic amenities and relief during the time of disasters and natural calamities. It runs a lot of operational projects from time to time along with advocacy drives to raise awareness as well as for funding purposes.
# 
# After the recent funding programs, they have been able to raise around $ 10 million. Now the CEO of the NGO needs to decide how to use this money strategically and effectively. The significant issues that come while making this decision are mostly related to choosing the countries that are in the direst need of aid.
# 
# > The NGO wants to identify the countries based on the socioeconomic factors **Child Mortality, Income, GDPP**

# ![Approach.JPG](attachment:Approach.JPG)

# ### Importing Libaries

# In[ ]:


import pandas as pd
import numpy as np
import pandas as pd

# For Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

# To Scale our data
from sklearn.preprocessing import scale

# To perform KMeans clustering 
from sklearn.cluster import KMeans

# To perform Hierarchical clustering
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


#importing Dataset already uploaded to ANACONDA cloud
df_original = pd.read_csv('../input/Country-data.csv')
df = pd.read_csv('../input/Country-data.csv')


# In[ ]:


df.head()


# In[ ]:


df.shape


# ### Checking for Missing Data

# In[ ]:


df.info()


# #### `Observation`
# - None of the rows have missing values
# 
# - All variables are int64/float64 excluding country, no categorical variables
# 
# - *No Data Preparation or Dummy variable creation is required for the columns in this dataset*

# ### Data Description

# In[ ]:


#Checking outliers for ciontinuous variables
# Checking outliers at 25%,50%,75%,90%,95% and 99%
df.describe(percentiles=[.25,.50,.75,.90,.95,.99])


# ### Scaling the Data

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scaler = StandardScaler()


# In[ ]:


y= df.pop('country')


# In[ ]:


X = df.copy()


# In[ ]:


x = scaler.fit_transform(X)


# In[ ]:


x


# ### Principal Component Analysis

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca = PCA(svd_solver = 'randomized', random_state=42)


# In[ ]:


pca.fit(x)


# In[ ]:


pca.components_


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


plt.ylabel('Cumulative Variance')
plt.xlabel('Number of Components')
plt.bar(range(1,len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)


# In[ ]:


var_cumu = np.cumsum(pca.explained_variance_ratio_)
var_cumu


# ### Making the Scree Plot

# In[ ]:


plt.plot(range(1,(len(var_cumu)+1)),var_cumu)
plt.title('Scree Plot')
plt.xlabel('Cumulative Variance')
plt.ylabel('Number of Components')


# `The graph starts flattening after 5 components, contributing to almost 95% variance`

# ### Creating a PCA with 5 components

# In[ ]:


from sklearn.decomposition import IncrementalPCA


# In[ ]:


pc5 = PCA(n_components = 5, random_state=42)


# In[ ]:


df_pc5 = pc5.fit_transform(x)


# In[ ]:


df_pc5.shape


# In[ ]:


df_pc5[:10]


# ### Making a dataframe out of it for convinience

# In[ ]:


df5 = pd.DataFrame(df_pc5, columns=['PC1','PC2','PC3','PC4','PC5'])


# In[ ]:


df5.head()


# In[ ]:


df5_final = pd.concat([df5,y], axis=1)


# In[ ]:


df5_final.head()


# ### Let's perfrom Outlier treatment

# In[ ]:


plt.figure(figsize=(20,5))
# For PC1
plt.subplot(1,5,1)
plt.title('PC1')
plt.boxplot(df5_final.PC1)

Q1 = df5_final.PC1.quantile(0.05)
Q3 = df5_final.PC1.quantile(0.95)
IQR = Q3-Q1
df5_final = df5_final[(df5_final.PC1>=Q1) & (df5_final.PC1<=Q3)]

# For PC2
plt.subplot(1,5,2)
plt.title('PC2')
plt.boxplot(df5_final.PC2)

Q1 = df5_final.PC2.quantile(0.05)
Q3 = df5_final.PC2.quantile(0.95)
IQR = Q3-Q1
df5_final = df5_final[(df5_final.PC2>=Q1) & (df5_final.PC2<=Q3)]


# For PC3
plt.subplot(1,5,3)
plt.title('PC3')
plt.boxplot(df5_final.PC3)

Q1 = df5_final.PC3.quantile(0.05)
Q3 = df5_final.PC3.quantile(0.95)
IQR = Q3-Q1
df5_final = df5_final[(df5_final.PC3>=Q1) & (df5_final.PC3<=Q3)]


# For PC4
plt.subplot(1,5,4)
plt.title('PC4')
plt.boxplot(df5_final.PC4)

Q1 = df5_final.PC4.quantile(0.05)
Q3 = df5_final.PC4.quantile(0.95)
IQR = Q3-Q1
df5_final = df5_final[(df5_final.PC4>=Q1) & (df5_final.PC4<=Q3)]


# For PC5
plt.subplot(1,5,5)
plt.title('PC5')
plt.boxplot(df5_final.PC5)

Q1 = df5_final.PC5.quantile(0.05)
Q3 = df5_final.PC5.quantile(0.95)
IQR = Q3-Q1
df5_final = df5_final[(df5_final.PC5>=Q1) & (df5_final.PC5<=Q3)]


# ### Clustering - Calculating the Hopkins statistic

# In[ ]:


### Clustering - Calculating the Hopkins statistic#Calculating the Hopkins statistic
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
from math import isnan
 
def hopkins(X):
    d = X.shape[1]
    #d = len(vars) # columns
    n = len(X) # rows
    m = int(0.1 * n) 
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)
 
    rand_X = sample(range(0, n, 1), m)
 
    ujd = []
    wjd = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X,axis=0),np.amax(X,axis=0),d).reshape(1, -1), 2, return_distance=True)
        ujd.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        wjd.append(w_dist[0][1])
 
    H = sum(ujd) / (sum(ujd) + sum(wjd))
    if isnan(H):
        print(ujd, wjd)
        H = 0
 
    return H


# In[ ]:


hopkins(df5_final.drop('country', axis =1))


# In[ ]:


df6_final = df5_final.drop('country', axis = 1)


# # K-Means Clustering

# ### Look at the silhouette score plot and choose the optimal number of cluster

# In[ ]:


#First we'll do the silhouette score analysis
from sklearn.metrics import silhouette_score

ss = []
for k in range(2,10):
    kmeans = KMeans(n_clusters = k).fit(df6_final)
    ss.append([k,silhouette_score(df6_final, kmeans.labels_)])


# In[ ]:


plt.title('Silhoette')
plt.plot(pd.DataFrame(ss)[0], pd.DataFrame(ss)[1]);


# ###  Look at the Elbow Curve plot and choose the optimal number of cluster

# In[ ]:


#Now let's proceed to the elbow curve method
from sklearn.metrics import silhouette_score

ssd = []
for k in range(1,10):
    model = KMeans(n_clusters = k, max_iter = 50).fit(df6_final)
    ssd.append([model.inertia_])

print(ssd)


# In[ ]:


plt.title('Elbow')
plt.plot(ssd)


# In[ ]:


range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(df6_final)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(df6_final, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))


# ## KMeans with the K we have choosed = 3

# In[ ]:


#Let's perform K means using K=
model_clus2 = KMeans(n_clusters = 3, max_iter = 50, random_state = 50)
model_clus2.fit(df6_final)


# In[ ]:


# Let's add the cluster Ids to the PCs data 

dat_km = pd.concat([df5_final.reset_index().drop('index', axis=1), pd.Series(model_clus2.labels_)], axis =1)


# In[ ]:


dat_km.head()


# In[ ]:


dat_km.columns = ['PC1', 'PC2', 'PC3','PC4','PC5', 'country','ClusterID']
dat_km.head()


# In[ ]:


# Check the count of observation per cluster
dat_km['ClusterID'].value_counts()


# In[ ]:


# Plot the Cluster with respect to the clusters obtained

plt.figure(figsize=(15,4))

plt.subplot(1,3,1)
sns.scatterplot(x='PC1', y ='PC2', hue = 'ClusterID', palette=['green','dodgerblue','red'], legend='full', data = dat_km)
plt.subplot(1,3,2)
sns.scatterplot(x='PC1', y ='PC3', hue = 'ClusterID', palette=['green','dodgerblue','red'], legend='full', data = dat_km)
plt.subplot(1,3,3)
sns.scatterplot(x='PC2', y ='PC3', hue = 'ClusterID', palette=['green','dodgerblue','red'], legend='full', data = dat_km)


# ### Cluster Profiling

# In[ ]:


# Let's merge the original data with the data(ClusterID)
dat5 = pd.merge(df_original, dat_km, how = 'inner', on = 'country')
dat5.head()


# In[ ]:


dat5.shape


# In[ ]:


dat6 = dat5[['country','child_mort', 'income','gdpp', 'PC1', 'PC2', 'ClusterID']]


# In[ ]:


dat6.groupby('ClusterID').count()


# In[ ]:


child_mort = dat6.groupby(['ClusterID']).child_mort.mean()
income = dat6.groupby(['ClusterID']).income.mean()
gdpp = dat6.groupby(['ClusterID']).gdpp.mean()


# In[ ]:


final_df = pd.concat([child_mort, income, gdpp], axis = 1)


# In[ ]:


final_df


# ### K-Means Suggestion
# - Countries in Cluster 2 have an average child mortality 90, 90/1000 child die before the age of 5
# - Countries in Cluster 2 have an average gdpp of 553 which is very low compared to clusters 0 & 1
# - Countries in Cluster 2 have an average income of 1600 which is very low compared to clusters 0 & 1

# In[ ]:


plt.figure(figsize=(20, 8))
plt.subplot(1,3,1)
sns.boxplot(x='ClusterID', y='income', data=dat6)

plt.subplot(1,3,2)
sns.boxplot(x='ClusterID', y='child_mort', data=dat6)

plt.subplot(1,3,3)
sns.boxplot(x='ClusterID', y='gdpp', data=dat6)


# In[ ]:


# List of Countries which need Attention.

dat6[dat6['ClusterID']==2]['country']


# # Hierarchical Clustering

# In[ ]:


rfm_df = df[['child_mort', 'income','gdpp']]

# instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
rfm_df_scaled.shape


# In[ ]:


rfm_df_scaled[:10]


# ### Single Linkage

# In[ ]:


# single linkage
plt.figure(figsize=(20,10))
plt.title('Dendrogram - Single Linkage')
mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
dendrogram(mergings)

plt.show()


# ### Complete linkage

# In[ ]:


# complete linkage
plt.figure(figsize=(20,10))
plt.title('Dendrogram - Complete Linkage')
mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()


# In[ ]:


# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
cluster_labels


# In[ ]:


# assign cluster labels
rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['child_mort', 'income','gdpp']
rfm_df_scaled.head()
rfm_df_scaled['cluster_labels'] = cluster_labels


# In[ ]:


rfm_df_scaled['country'] = df_original['country']
rfm_df_scaled.groupby('cluster_labels').count()


# In[ ]:


# Plot the Cluster with respect to the clusters obtained

plt.figure(figsize=(20,5))

plt.subplot(1,3,1)
sns.scatterplot(x='child_mort', y ='income', hue = 'cluster_labels', palette=['green','dodgerblue','red'], legend='full', data = rfm_df_scaled)
plt.subplot(1,3,2)
sns.scatterplot(x='child_mort', y ='gdpp', hue = 'cluster_labels', palette=['green','dodgerblue','red'], legend='full', data = rfm_df_scaled)
plt.subplot(1,3,3)
sns.scatterplot(x='gdpp', y ='income', hue = 'cluster_labels', palette=['green','dodgerblue','red'], legend='full', data = rfm_df_scaled)


# In[ ]:


rfm_df_scaled[rfm_df_scaled['cluster_labels']==0].head(10)


# In[ ]:


rfm_df_scaled[rfm_df_scaled['cluster_labels']==1].head(10)


# In[ ]:


rfm_df_scaled[rfm_df_scaled['cluster_labels']==2]


# In[ ]:


plt.figure(figsize=(20,8))
plt.subplot(1,3,1)
sns.boxplot(x='cluster_labels', y='income', data=rfm_df_scaled)

plt.subplot(1,3,2)
sns.boxplot(x='cluster_labels', y='child_mort', data=rfm_df_scaled)

plt.subplot(1,3,3)
sns.boxplot(x='cluster_labels', y='gdpp', data=rfm_df_scaled)


# ### Heirarchical Clustring - Suggestion
# - Countries in Cluster 2 have good income,gdpp and low child mortality.
# - Countries in Cluster 1 have income almost similar to Cluster 0 but with high child mortality.
# 
# - **In my analysis countries from Cluster 1 should be considered**

# In[ ]:


rfm_df_scaled[rfm_df_scaled['cluster_labels']==1]['country']


# ### Merging the 2 results and analysing

# In[ ]:


# List of Countries which need Attention.

df_H = rfm_df_scaled[rfm_df_scaled['cluster_labels']==1]
df_H = df_H[['country', 'child_mort','income', 'gdpp']]
df_H.head(30)


# In[ ]:


# List of Countries which need Attention.

df_K = dat6[dat6['ClusterID']==2]
df_K = df_K[['country', 'child_mort','income', 'gdpp']]
df_K.head(30)


# In[ ]:


df_combined = pd.concat([df_K, df_H], join = 'inner' )
df_combined.head(20)

df_combined.head()


# In[ ]:


plt.figure(figsize=(15,15))
plt.subplot(1,3,1)
plt.title('child_mort')
sns.barplot(x="child_mort", y="country", data=df_combined.sort_values(by=['income'], ascending = True))

plt.subplot(1,3,2)
plt.title('income')
sns.barplot(x="income", y="country", data=df_combined.sort_values(by=['income'], ascending = True))

plt.subplot(1,3,3)
plt.title('gdpp')
sns.barplot(x="gdpp", y="country", data=df_combined.sort_values(by=['income'], ascending = True))


#  **From the above Graph Countries which need Attention**
# - Child Mortality : Mozambique, Malawi, Togo, Guinea, Comoros
# - Income: Congo, Dem. Rep.,Niger, Central African Republic, Sierra Leone, Burkina Faso, Haiti
# - GDPP: Congo, Dem. Rep.,Niger, Central African Republic, Sierra Leone, Burkina Faso, Haiti

# In[ ]:


# Bottom 20 Countries which need help
df_combined.sort_values(by=['income'], ascending = True)[:20]['country']


# `End`
