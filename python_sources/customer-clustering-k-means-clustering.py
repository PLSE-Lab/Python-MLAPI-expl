#!/usr/bin/env python
# coding: utf-8

# # Customer Segmentation

# ## K- Means Clustering 

# ### Problem Statement:
# To identify different segments of customers on the basis of their shopping behaviour to run targeted marketing campaign.
# 
# ### Data:
# Online retail is a transnational data set which contains all the transactions occurring between 01/12/2010 and 09/12/2011 for a UK-based and registered non-store online retail.The company mainly sells unique all-occasion gifts. Many customers of the company are wholesalers.
# 
# ### Approach:
# We will build a RFM clustering to identify different customers.

# In[ ]:


# supress warnings
import warnings
warnings.filterwarnings('ignore')

#Importing Libraries
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


# ### Data Reading and Understanding

# In[ ]:


retail = pd.read_csv("../input/Online Retail.csv", sep = ',',encoding = "ISO-8859-1", header= 0)
retail.head()


# ### Data Inspection

# In[ ]:


retail.shape


# In[ ]:


retail.describe()


# In[ ]:


retail.info()


# ### Data Cleaning

# In[ ]:


# parse date
retail['InvoiceDate'] = pd.to_datetime(retail['InvoiceDate'], format = "%d-%m-%Y %H:%M")
retail.head()


# ### Data quality check and cleaning

# In[ ]:


# Let's look top 5 rows
retail.head()


# In[ ]:


#Na Handling
retail.isnull().values.any()


# In[ ]:


retail.isnull().sum()*100/retail.shape[0]


# In[ ]:


#dropping the na cells
retail = retail.dropna()


# In[ ]:


retail.isnull().sum()*100/retail.shape[0]


# ## RFM

# ### Extracting R (Recency), F (Frequency), M (Monetary) columns form the data.

# In[ ]:


#RFM implementation

# Extracting amount by multiplying quantity and unit price and saving the data into amount variable.
retail["Amount"]  = retail.Quantity * retail.UnitPrice


# #### Monetary Value

# In[ ]:


# Monetary Function

# Finding total amount spent per customer
monetary = retail.groupby("CustomerID").Amount.sum()
monetary = monetary.reset_index()
monetary.head()


# #### Frequency Value

# In[ ]:


#Frequency function

# Getting the count of orders made by each customer based on customer ID.
frequency = retail.groupby("CustomerID").InvoiceNo.count()
frequency = frequency.reset_index()
frequency.head()


# ##### Merging Amount and Frequency columns

# In[ ]:


#creating master dataset
master = monetary.merge(frequency, on = "CustomerID", how = "inner")
master.head()


# ### Recency Value

# In[ ]:


# Finding max data
maximum = max(retail.InvoiceDate)


# In[ ]:


# Adding one more day to the max data, so that the max date will have 1 as the difference and not zero.
maximum = maximum + pd.DateOffset(days = 1)


# In[ ]:


retail['diff'] = maximum - retail.InvoiceDate
retail.head()


# In[ ]:


#Dataframe merging by recency
recency = retail.groupby('CustomerID').diff.min()
recency = recency.reset_index()
recency.head()


# ### RFM combined DataFrame

# In[ ]:


#Combining all recency, frequency and monetary parameters
RFM = master.merge(recency, on = "CustomerID")
RFM.columns = ['CustomerID','Amount','Frequency','Recency']
RFM.head()


# In[ ]:


RFM.info()


# ### Outlier Treatment

# In[ ]:


# outlier treatment for Amount
fig, axs = plt.subplots(1,3, figsize = (15,5))

sns.boxplot(RFM.Amount, ax = axs[0])
sns.boxplot(RFM.Frequency, ax = axs[1])
sns.boxplot(RFM.Recency.dt.days, ax = axs[2])

plt.tight_layout
plt.show()


# In[ ]:


# outlier treatment for Amount
Q1 = RFM.Amount.quantile(0.25)
Q3 = RFM.Amount.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Amount >= Q1 - 1.5*IQR) & (RFM.Amount <= Q3 + 1.5*IQR)]


# In[ ]:


# outlier treatment for Frequency
Q1 = RFM.Frequency.quantile(0.25)
Q3 = RFM.Frequency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Frequency >= Q1 - 1.5*IQR) & (RFM.Frequency <= Q3 + 1.5*IQR)]


# In[ ]:


# outlier treatment for Recency
Q1 = RFM.Recency.quantile(0.25)
Q3 = RFM.Recency.quantile(0.75)
IQR = Q3 - Q1
RFM = RFM[(RFM.Recency >= Q1 - 1.5*IQR) & (RFM.Recency <= Q3 + 1.5*IQR)]


# In[ ]:


fig, axs = plt.subplots(1,3, figsize = (15,5))

sns.boxplot(RFM.Amount, ax = axs[0])
sns.boxplot(RFM.Frequency, ax = axs[1])
sns.boxplot(RFM.Recency.dt.days, ax = axs[2])

plt.tight_layout
plt.show()


# In[ ]:


RFM.head()


# ### Scaling the RFM data

# In[ ]:


# standardise all parameters
RFM_norm1 = RFM.drop("CustomerID", axis=1)
RFM_norm1.Recency = RFM_norm1.Recency.dt.days

from sklearn.preprocessing import StandardScaler
standard_scaler = StandardScaler()
RFM_norm1 = standard_scaler.fit_transform(RFM_norm1)


# In[ ]:


RFM_norm1 = pd.DataFrame(RFM_norm1)
RFM_norm1.columns = ['Frequency','Amount','Recency']
RFM_norm1.head()


# ## Hopkins Statistics:
# The Hopkins statistic, is a statistic which gives a value which indicates the cluster tendency, in other words: how well the data can be clustered.
# 
# - If the value is between {0.01, ...,0.3}, the data is regularly spaced.
# 
# - If the value is around 0.5, it is random.
# 
# - If the value is between {0.7, ..., 0.99}, it has a high tendency to cluster.

# In[ ]:


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


hopkins(RFM_norm1)


# ## K-Means with some K

# ## Silhouette Analysis
# 
# $$\text{silhouette score}=\frac{p-q}{max(p,q)}$$
# 
# $p$ is the mean distance to the points in the nearest cluster that the data point is not a part of
# 
# $q$ is the mean intra-cluster distance to all the points in its own cluster.
# 
# * The value of the silhouette score range lies between -1 to 1. 
# 
# * A score closer to 1 indicates that the data point is very similar to other data points in the cluster, 
# 
# * A score closer to -1 indicates that the data point is not similar to the data points in its cluster.

# In[ ]:


from sklearn.metrics import silhouette_score
sse_ = []
for k in range(2, 15):
    kmeans = KMeans(n_clusters=k).fit(RFM_norm1)
    sse_.append([k, silhouette_score(RFM_norm1, kmeans.labels_)])


# In[ ]:


plt.plot(pd.DataFrame(sse_)[0], pd.DataFrame(sse_)[1]);


# ## Sum of Squared Distances

# In[ ]:


# sum of squared distances
ssd = []
for num_clusters in list(range(1,21)):
    model_clus = KMeans(n_clusters = num_clusters, max_iter=100)
    model_clus.fit(RFM_norm1)
    ssd.append(model_clus.inertia_)

plt.plot(ssd)


# In[ ]:


# Kmeans with K=5
model_clus5 = KMeans(n_clusters = 5, max_iter=50)
model_clus5.fit(RFM_norm1)


# In[ ]:


# analysis of clusters formed
RFM.index = pd.RangeIndex(len(RFM.index))
RFM_km = pd.concat([RFM, pd.Series(model_clus5.labels_)], axis=1)
RFM_km.columns = ['CustomerID', 'Amount', 'Frequency', 'Recency', 'ClusterID']
RFM_km.head()


# In[ ]:


RFM_km.Recency = RFM_km.Recency.dt.days
km_clusters_amount = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Amount.mean())
km_clusters_frequency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Frequency.mean())
km_clusters_recency = pd.DataFrame(RFM_km.groupby(["ClusterID"]).Recency.mean())


# In[ ]:


df = pd.concat([pd.Series([0,1,2,3,4]), km_clusters_amount, km_clusters_frequency, km_clusters_recency], axis=1)
df.columns = ["ClusterID", "Amount_mean", "Frequency_mean", "Recency_mean"]
df.head()


# In[ ]:


fig, axs = plt.subplots(1,3, figsize = (15,5))

sns.barplot(x=df.ClusterID, y=df.Amount_mean, ax = axs[0])
sns.barplot(x=df.ClusterID, y=df.Frequency_mean, ax = axs[1])
sns.barplot(x=df.ClusterID, y=df.Recency_mean, ax = axs[2])
plt.tight_layout()            
plt.show()


# ## Inferences
# 1. On RFM analysis it was found that Customers with Cluster ID 4 are the best and loyal customers.
# 2. Cluster Id 3 are customers who are not interested in retail store.
# 3. People with cluster ID 0,1,2 needs targeted marketing marketing based on their demographics, buying pattern etc. 
