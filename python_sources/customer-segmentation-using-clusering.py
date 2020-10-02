#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/CC GENERAL.csv")


# In[ ]:


data.head()


# In[ ]:


data.info()


# In[ ]:


data.describe()


# # Missing Values

# In[ ]:


data.isna().sum()


# In[ ]:


data = data.fillna(data.mean())
data.isna().sum()


# In[ ]:


data.drop('CUST_ID', axis=1, inplace=True)


# In[ ]:


data.head(2)


# # Data Exploration

# In[ ]:


data.dtypes


# In[ ]:


data.nunique()


# In[ ]:


data[['CASH_ADVANCE_TRX','PURCHASES_TRX','TENURE']].nunique()


# # Corelation Plot

# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(data.corr(),xticklabels=data.columns, yticklabels=data.columns, annot=True)


# In[ ]:


sns.pairplot(data)


# In[ ]:


fig, axes = plt.subplots(ncols=1, nrows=3)
ax0,ax1,ax2 = axes.flatten()

ax0.hist(data['CASH_ADVANCE_TRX'],65,histtype='bar', stacked=True)
ax0.set_title('CASH_ADVANCE_TRX')

ax1.hist(data['PURCHASES_TRX'], 173, histtype='bar', stacked=True)
ax1.set_title('PURCHASES_TRX')

ax2.hist(data['TENURE'],7,histtype='bar', stacked=True)
ax2.set_title('TENURE')

fig.tight_layout()


# # Feature Generation

# In[ ]:


features = data.copy()
list(features)


# In[ ]:


cols = ['BALANCE',
        'PURCHASES',
        'ONEOFF_PURCHASES',
        'INSTALLMENTS_PURCHASES',
        'CASH_ADVANCE',
        'CASH_ADVANCE_TRX',
        'PURCHASES_TRX',
        'CREDIT_LIMIT',
        'PAYMENTS',
        'MINIMUM_PAYMENTS']
features[cols] = np.log(1+features[cols])
features.head()


# In[ ]:


data.head()


# In[ ]:


features.describe()


# In[ ]:


# Determining outliers by boxplot
features.boxplot(rot=90, figsize=(30,10))


# # Clustering using Kmeans

# ### Elbow method to find the appeopreate number of clusters

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


X = np.array(features)
sumOfSqrdDist = []
K = range(1,15)

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans = kmeans.fit(X)
    sumOfSqrdDist.append([k, kmeans.inertia_])
    
plt.plot(K, sumOfSqrdDist, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow method for optimal k')
plt.show();


# In[ ]:


sumOfSqrdDist


# In[ ]:


n_cluster = 11
clustering = KMeans(n_clusters=n_cluster, random_state=42)
cluster_labels = clustering.fit_predict(X)

plt.hist(cluster_labels, bins=range(n_cluster+1))
plt.title("# Customers per cluster")
plt.xlabel("Cluster")
plt.ylabel(" # Customers")
plt.show()

features['CLUSTER_INDEX'] = cluster_labels
data['CLUSTER_INDEX'] = cluster_labels


# In[ ]:


kmeans.cluster_centers_


# ### Dendograms

# In[ ]:


from scipy.cluster.hierarchy import ward,dendrogram,linkage
np.set_printoptions(precision=4,suppress=True)


# In[ ]:


distance = linkage(X,'ward')


# In[ ]:


plt.figure(figsize=(20,10))
plt.title("Hierarchical Clustering Dendogram")
plt.xlabel("Index")
plt.ylabel("Ward's Distance")
dendrogram(distance, leaf_rotation=90, leaf_font_size=9);
plt.axhline(98, c='k')


# ### Clusters by distance

# In[ ]:


from scipy.cluster.hierarchy import fcluster

max_d = 97
clusters = fcluster(distance, max_d, criterion='distance')
clusters


# In[ ]:


k = 11 #K=3
clusters = fcluster(distance, k, criterion='maxclust')

plt.figure(figsize=(10,8))
plt.scatter(X[:,0], X[:,1], c=clusters);


# # Silhoutte_score

# In[ ]:


from sklearn.metrics import silhouette_score

sumOfSquaredErrors = []
for k in range(2,30):
    kmeans = KMeans(n_clusters=k).fit(X)
    sumOfSquaredErrors.append([k,silhouette_score(X, kmeans.labels_)])


# In[ ]:


plt.plot(pd.DataFrame(sumOfSquaredErrors)[0],pd.DataFrame(sumOfSquaredErrors)[1])


# # Evaluation

# In[ ]:


data.head()


# In[ ]:


data['CLUSTER_INDEX'].nunique()


# In[ ]:


from sklearn.metrics import confusion_matrix,classification_report


# In[ ]:


print(classification_report(features['CLUSTER_INDEX'], kmeans.labels_))


# In[ ]:




