#!/usr/bin/env python
# coding: utf-8

# This is my first kernel and I am trying to implment all the skills I have learnt during my work on Credit Risk and Credit Card Marketing.
# The approach I have followed is:
# 1.     Do EDA on the Data and understand what variables makes sense to take into clustering.
# 1.     k-mean clustering with certain clusters and then combining them to arrive at n distinct clusters
# 1.     2-D & 3-D visualizations with Plotly to understand how clusters look different
# 1.     Try Agglomerative Clustering which is a type Hierarchical Clustering 
# 1.     Visualize the results on some variable
#     
# 
# **Clustering & its benefits?**
# Clustering is the grouping of objects together so that objects belonging in the same group (cluster) are more similar to each other than those in other groups (clusters). Clustering solves a lot of purpose:
# *     Clustering is useful to find similar groups of stores for retail chains to help an organization with increase of sales in the same cluster
# *     Clustering helps in the test and control strategies in marketing
# *     Clustering helps in disburment of loans of similar types and mitigate risk
# *     Clustering can also help in developing recommendation engines
# *     Clustering is also useful in social network analysis like finding targets on Facebook, Twitter or Instagram
#   
# In here, the objective is to market or rather personalize Credit Cards products/offers to one cluster and diversify on the basis of strategy. It will help to remain focus on each clusters and implement strategies as per their Credit Card Behaviour. In my own experience, I have used clustering a lot both in retail and banking projects
# 
# **Types of clustering**
# There are many types of clustering but in this kernel we will implement two types:
# *     **Non-Hierarchical Clustering**: In this the groups are formed in comparison to the distance between each other. In the non-hierarchical method a position in the measurement is taken as central place and distance is measured from such central point (seed). The distance used is Eucleadian Distance. A popular method is k-means which is based on selecting random central points (Centroids) and then calculating the distance of all other points from these random centroids uptill there is no change in centroids. All the points closer to one centroid will be considered a cluster. In this there is no relationship or a hierarchy considered with the other point
#     
# *     **Hierarchical Clustering**: Hierarchical clustering, as the name suggests is an algorithm that builds hierarchy of clusters. This algorithm starts with all the data points assigned to a cluster of their own. Then two nearest clusters are merged into the same cluster. In the end, this algorithm terminates when there is only a single cluster left. Thus we are left with a hierarchy for each of the cluster
# 
# I will be trying to do clustering with some other techniques when I get time.
# I would appreciate your feedback and please upvote if you like my work

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA


# In[ ]:


data_   = pd.read_csv('../input/ccdata/CC GENERAL.csv')


# In[ ]:


data_.head()


# In[ ]:


data_.describe()


# ### Check for missing values

# In[ ]:


def missings_(data):
    miss      = data.isnull().sum()
    miss_pct  = 100 * data.isnull().sum()/len(data)
    
    miss_pct      = pd.concat([miss,miss_pct], axis=1)
    missings_cols = miss_pct.rename(columns = {0:'Missings', 1: 'Missing pct'})
    missings_cols = missings_cols[missings_cols.iloc[:,1]!=0].sort_values('Missing pct', ascending = False).round(1)
    
    return missings_cols  

missings = missings_(data_)
missings


# In[ ]:


fig, ax = plt.subplots(1,4,figsize =(20,4))
ax0, ax1, ax2, ax3 = ax.flatten()

ax0.hist(data_['BALANCE'], bins = 60, alpha =0.8 )
ax1.hist(data_['ONEOFF_PURCHASES'], bins = 60, color="green" ,alpha =0.8 )
ax2.hist(data_['PURCHASES'], bins = 60, color="red",alpha =0.8 )
ax3.hist(data_['PAYMENTS'], bins = 60, color="orange",alpha =0.8 )

ax0.set_title("BALANCES")
ax1.set_title("ONEOFF_PURCHASES")
ax2.set_title("PURCHASES")
ax3.set_title("PAYMENTS")

plt.show()


# In[ ]:


cols   = data_.columns
fig , ax = plt.subplots(1,4, figsize = (20,8))
ax0, ax1, ax2, ax3 = ax.flatten() 

for i in range(0,4):
    
    X   = data_[cols[i+2]]
    Y   = data_[cols[1]]
    ax[i].plot(X, Y, marker = 'o', linestyle = "None")
    ax[i].set_xlabel(cols[i+2])
    ax[0].set_ylabel(cols[1])


# In[ ]:


fig = plt.figure(figsize = (50,20))
data_sub = data_[(data_['BALANCE_FREQUENCY']>=0.3)]
data_sub['BALANCE_FREQ'] = round(data_['BALANCE_FREQUENCY'],2)

sns.violinplot(y='BALANCE',x='BALANCE_FREQ',data=data_sub)
plt.xlabel('BALANCE FREQ',fontsize=40)
plt.ylabel('BALANCE',fontsize=40)
plt.tick_params(labelsize=30)


# ### Exploring how Purchases behave for these customers
# #### Creating bins to understand the distributions of these variables

# In[ ]:


data_['Balance_decile'] = pd.qcut(data_['BALANCE'], q=10)
data_grp   = data_.groupby('Balance_decile', as_index=False).mean()
data_grp   = data_grp[['Balance_decile', 'PURCHASES', 'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES']]
data_grp_t = pd.melt(data_grp, id_vars = 'Balance_decile')


# In[ ]:


fig = plt.figure(figsize=(20,10))
sns.barplot(x= "Balance_decile" , y = "value", hue = 'variable', data =data_grp_t)
plt.ylabel("Average Purchase Amount", fontsize=20)
plt.xlabel(" Balance Groups", fontsize =20)
plt.tick_params(labelsize=12)
plt.xticks(rotation=45)
plt.show()


# #### Finding average Balance left by Frequency of Purchases

# In[ ]:


data_['freq_purchase_decile'] = pd.qcut(data_['PURCHASES_FREQUENCY'], q=4)
data_bal   = data_.groupby('freq_purchase_decile', as_index=False).mean()
fig = plt.figure(figsize=(10,5))
sns.barplot(x= "freq_purchase_decile" , y = "BALANCE", data =data_bal)
plt.show()


# #### Credit Card Utilization: It is quite useful metric while working on Banking projects or in a Fintech. It tells us basically how much unused balance is left in the credit card of the customer. It is calcualated as CC Utilization = (Credit Card Limit - Balance Used)/Credit Card Limit

# In[ ]:


data_['CREDIT_LIMIT'].fillna(1, inplace=True)
data_['CC_utilisation']     = (data_['CREDIT_LIMIT'] - data_['BALANCE'])/data_['CREDIT_LIMIT']


# In[ ]:


data_['CC_util_decile']     = pd.qcut(data_['CC_utilisation'], q=10)
data_cc_grp                 = data_.groupby('CC_util_decile', as_index=False).mean()
data_cc_grp                 = data_cc_grp[['CC_util_decile', 'PAYMENTS' , 'MINIMUM_PAYMENTS']]
data_cc_grp_t               = pd.melt(data_cc_grp, id_vars = 'CC_util_decile')


# In[ ]:


fig = plt.figure(figsize=(10,5))
sns.barplot(x= "CC_util_decile" , y = "value", hue = "variable" ,data =data_cc_grp_t)
plt.xlabel("Credit Card Utilization")
plt.ylabel("Average Payments")
plt.xticks(rotation=45)
plt.show()


# ### Creating clusters for Marketing and Risk Strategy
# #### Also, we had outliers in our columns so it will be better to bin the variables

# In[ ]:


data_n  = data_.copy()


# In[ ]:


cols = ['BALANCE', 'PURCHASES', 'ONEOFF_PURCHASES','CASH_ADVANCE',
         'CREDIT_LIMIT', 'PAYMENTS']
for c in cols:
    bins = c+'_bin'
    max_ = max(data_n[c])
    data_n[bins] = pd.cut(data_n[c], bins=[0,500,1000,3000,5000,10000,15000,max_],labels = [1,2,3,4,5,6,7], include_lowest= True)


# In[ ]:


cols = ['BALANCE_FREQUENCY', 'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY','PURCHASES_INSTALLMENTS_FREQUENCY',
         'CASH_ADVANCE_FREQUENCY']
for c in cols:
    bins = c+'_bin'
    max_ = max(data_[c])
    data_n[bins] = pd.cut(data_n[c], bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,max_],labels = [1,2,3,4,5,6,7,8,9,10], include_lowest= True)


# In[ ]:


cols = ['CASH_ADVANCE_TRX', 'PURCHASES_TRX']

for c in cols:
    bins = c+'_bin'
    max_ = max(data_[c])
    data_n[bins] = pd.cut(data_n[c], bins=[0,20,40,60,80,100,max_],labels = [1,2,3,4,5,6], include_lowest= True)


# #### Dropping columns before doing k-means clustering

# In[ ]:


data_model  = data_n.drop(['CUST_ID', 'BALANCE', 'BALANCE_FREQUENCY', 'PURCHASES',
       'ONEOFF_PURCHASES', 'INSTALLMENTS_PURCHASES', 'CASH_ADVANCE',
       'PURCHASES_FREQUENCY', 'ONEOFF_PURCHASES_FREQUENCY',
       'PURCHASES_INSTALLMENTS_FREQUENCY', 'CASH_ADVANCE_FREQUENCY',
       'CASH_ADVANCE_TRX', 'PURCHASES_TRX', 'CREDIT_LIMIT', 'PAYMENTS',
       'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT', 'Balance_decile',
       'freq_purchase_decile', 'CC_utilisation', 'TENURE', 'PURCHASES_TRX_bin', 'CASH_ADVANCE_TRX_bin'], axis=1)


# In[ ]:


data_model  = data_model.drop(['CC_util_decile'], axis =1)


# In[ ]:


stand_         = StandardScaler()
data_model_std = stand_.fit_transform(data_model)

random.seed(234)
n_clusters=20
sse=[]
for i in range(1,n_clusters+1):
    kmean= KMeans(i)
    kmean.fit(data_model_std)
    sse.append([i, kmean.inertia_]) 


# In[ ]:


sse


# In[ ]:


plt.plot(pd.DataFrame(sse)[0], pd.DataFrame(sse)[1])
plt.title("Elbow Curve")


# In[ ]:


random.seed(234)
kmean= KMeans(8)
kmean.fit(data_model_std)


# In[ ]:


kmean.cluster_centers_


# In[ ]:


y_kmeans = kmean.predict(data_model_std)
y_kmeans

data_model['Cluster']       = y_kmeans
data_model_std              = pd.DataFrame(data_model_std)
data_model_std['Cluster']   = y_kmeans


# In[ ]:


data_model['Cluster'].value_counts()


# In[ ]:


for c in data_model:
    g   = sns.FacetGrid(data_model, col='Cluster')
    g.map(plt.hist, c, color = "red")


# ### Seeing the clusters we try to regroup them on the basis of their variable distributions
# #### Cluster 1 and 3 could be combined
# #### Cluster 0 and 2 could be combined
# #### Cluster 4 and 5 could be combined

# In[ ]:


data_model["Cluster"].replace({3: 1, 2: 0, 5:4}, inplace=True)
data_model['Cluster'].value_counts()
data_model_std["Cluster"].replace({3: 1, 2: 0, 5:4}, inplace=True)
clusters_   = data_model["Cluster"]


# ### 2-D & 3-D Visualization of clusters using PCA and plotly. Here we will visualize the cluster using seaborn and Plotly library. Plotly allows us to draw some amazing and cutting edge interactive visualizations. We wil be using PCA as we had to get 2 and 3 axis which is a combination of the other variables in the data

# In[ ]:


random.seed(32)
pca = PCA()
pca.fit(data_model_std)


fig = plt.figure(figsize =(12,6))
plt.plot(range(0,12),pca.explained_variance_ratio_.cumsum(), marker ='o', linestyle = "--")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance Explained")


# In[ ]:


pca = PCA(n_components = 3)
pca.fit(data_model_std)


# In[ ]:


scores = pca.transform(data_model_std)


x,y = scores[:,0] , scores[:,1]
df_data = pd.DataFrame({'x': x, 'y':y, 'clusters':clusters_})


# In[ ]:


grouping_ = df_data.groupby('clusters')
fig, ax = plt.subplots(figsize=(20, 13))

names = {0: 'Cluster 1', 
         1: 'Cluster 2', 
         4: 'Cluster 3',
         6: 'Cluster 4',
         7: 'Cluster 5'}

for name, grp in grouping_:
    ax.plot(grp.x, grp.y, marker='o', label = names[name], linestyle='')
    ax.set_aspect('auto')

ax.legend()
plt.show()


# In[ ]:


import plotly
import plotly.graph_objs as go
plotly.offline.init_notebook_mode()


# In[ ]:


x,y,z = scores[:,0] , scores[:,1], scores[:,2]

df_data = pd.DataFrame({'x': x, 'y':y, 'z':z, 'clusters':clusters_})


# In[ ]:


# Visualize cluster shapes in 3d.

cluster1=df_data.loc[df_data['clusters'] == 0]
cluster2=df_data.loc[df_data['clusters'] == 1]
cluster3=df_data.loc[df_data['clusters'] == 4]
cluster4=df_data.loc[df_data['clusters'] == 6]
cluster5=df_data.loc[df_data['clusters'] == 7]


scatter1 = dict(
    mode = "markers",
    name = "Cluster 1",
    type = "scatter3d",    
    x = cluster1.to_numpy()[:,0], y = cluster1.to_numpy()[:,1], z = cluster1.to_numpy()[:,2],
    marker = dict( size=2, color='green')
)
scatter2 = dict(
    mode = "markers",
    name = "Cluster 2",
    type = "scatter3d",    
    x = cluster2.to_numpy()[:,0], y = cluster2.to_numpy()[:,1], z = cluster2.to_numpy()[:,2],
    marker = dict( size=2, color='blue')
)
scatter3 = dict(
    mode = "markers",
    name = "Cluster 3",
    type = "scatter3d",    
    x = cluster3.to_numpy()[:,0], y = cluster3.to_numpy()[:,1], z = cluster3.to_numpy()[:,2],
    marker = dict( size=2, color='red')
)

scatter4 = dict(
    mode = "markers",
    name = "Cluster 4",
    type = "scatter3d",    
    x = cluster4.to_numpy()[:,0], y = cluster4.to_numpy()[:,1], z = cluster4.to_numpy()[:,2],
    marker = dict( size=2, color='orange')
)

scatter5 = dict(
    mode = "markers",
    name = "Cluster 5",
    type = "scatter3d",    
    x = cluster5.to_numpy()[:,0], y = cluster5.to_numpy()[:,1], z = cluster5.to_numpy()[:,2],
    marker = dict( size=2, color='yellow')
)


################## Clusters  ##############

cluster1 = dict(
    alphahull = 5,
    name = "Cluster 1",
    opacity = .1,
    type = "mesh3d",    
    x = cluster1.to_numpy()[:,0], y = cluster1.to_numpy()[:,1], z = cluster1.to_numpy()[:,2],
    color='green', showscale = True
)
cluster2 = dict(
    alphahull = 5,
    name = "Cluster 2",
    opacity = .1,
    type = "mesh3d",    
    x = cluster2.to_numpy()[:,0], y = cluster2.to_numpy()[:,1], z = cluster2.to_numpy()[:,2],
    color='blue', showscale = True
)
cluster3 = dict(
    alphahull = 5,
    name = "Cluster 3",
    opacity = .1,
    type = "mesh3d",    
    x = cluster3.to_numpy()[:,0], y = cluster3.to_numpy()[:,1], z = cluster3.to_numpy()[:,2],
    color='red', showscale = True
)

cluster4 = dict(
    alphahull = 5,
    name = "Cluster 4",
    opacity = .1,
    type = "mesh3d",    
    x = cluster4.to_numpy()[:,0], y = cluster4.to_numpy()[:,1], z = cluster4.to_numpy()[:,2],
    color='orange', showscale = True
)

cluster5 = dict(
    alphahull = 5,
    name = "Cluster 5",
    opacity = .1,
    type = "mesh3d",    
    x = cluster5.to_numpy()[:,0], y = cluster5.to_numpy()[:,1], z = cluster5.to_numpy()[:,2],
    color='yellow', showscale = True
)

layout = dict(
    title = '3D visualisation of Clusters',
    scene = dict(
        xaxis = dict( zeroline=True ),
        yaxis = dict( zeroline=True ),
        zaxis = dict( zeroline=True ),
    )
)
fig = dict( data=[scatter1, scatter2, scatter3, scatter4, scatter5, cluster1, cluster2, cluster3, cluster4, cluster5], layout=layout )
# Use py.iplot() for IPython notebook
plotly.offline.iplot(fig, filename='mesh3d_sample')


# In[ ]:


#fig  = plt.figure(figsize = (7200,30))
for c in data_model:
    g   = sns.FacetGrid(data_model, col='Cluster')
    g.map(plt.hist, c, color = "red")


# ## Profiling of Clusters
# ### Cluster 0: Who do not purchase but have good credit limit. Also miss payments 
# ### Cluster 1: Who have a good balance, make average purchases and do make payments
# ### Cluster 4: Who buy frequntly and have a high credit limit
# ### Cluster 6: Who buy very small, keeps low balance but frequently pay dues
# ### Cluster 7: Who buy in installments only

# ## **Agglomerative Clustering(Hierarchical Clustering)**
# Let us also validate our results with some other types of clustering too. k-mean is Non-heirarchial but let us also try hierarchial clustering using AgglomerativeClustering

# In[ ]:


cluster = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')  
cl = cluster.fit_predict(data_model_std)


# Now, We will visualize our clusters and compare the results with k-means clustering

# In[ ]:


fig = plt.figure(figsize=(10, 7))

plt.scatter(x,y, c=cluster.labels_)
plt.title("Agglomerative Clustering with % Clusters")


# As we can see the resulting clusters from Agglomerative Clustering doesn't show that good results. The three clusters on the right are separted but that is not the case with the left most clusters. The results match up with k-means but there is no improvement in the clusters.

# ## Comparison of between Non-Hierarchial and Hierarchial Clustering
# ### Here we will compare both the type sof clusters to see which one gives better results. The way it is done is after getting the clusters, these will be profiled using Average and median of the variable and will be seen visually.
# ### k-means (Non-Hierarchial)

# In[ ]:


### Getting the required 
data_k_means  = pd.concat([data_,data_model['Cluster']], axis=1)
data_k_means  = data_k_means.drop(['CUST_ID', 'Balance_decile', 'CC_utilisation','CC_util_decile'], axis =1)

### Checking some basic stats/distributions from our clusters
data_freq     = data_k_means.filter(regex="FREQUENCY")
data_amount   = data_k_means.drop(list(data_freq.columns), axis=1)

#Finding Mean
data_amnt_m   = data_amount.groupby(['Cluster'],as_index=False).mean()

#Finding Median
data_freq     = pd.concat([data_freq,data_model['Cluster']], axis=1)
data_freq_m   = data_freq.groupby(['Cluster'],as_index=False).median()

#Join both of them
data_all_kmeans = pd.merge(data_amnt_m, data_freq_m)


# In[ ]:


data_all_kmeans


# ### Agglomerative Clustering (Non-Hierarchical)

# In[ ]:


## Getting the Data
cl            = pd.DataFrame(cl)
cl.columns    = ['Cluster']
data_agg      = pd.concat([data_,cl], axis=1)

data_agg          = data_agg.drop(['CUST_ID', 'Balance_decile', 'CC_utilisation','CC_util_decile'], axis =1)
data_freq_agg     = data_agg.filter(regex="FREQUENCY")
data_amount_agg   = data_agg.drop(list(data_freq_agg.columns), axis=1)

#Finding Mean
data_amnt_agg_m   = data_amount_agg.groupby(['Cluster'],as_index=False).mean()

#Finding Median
data_freq_agg         = pd.concat([data_freq_agg,cl], axis=1)
data_freq_agg_m       = data_freq_agg.groupby(['Cluster'],as_index=False).median()

#Join both of them
data_all_agg = pd.merge(data_amnt_agg_m, data_freq_agg_m)


# In[ ]:


data_all_agg


# ### In order to plot them the data needs to be converted into a long format from a wider format.

# In[ ]:


## Changing the Data Structure so that we could compare them visually
## k-means

## Amount 
data_amnt_m_sub1     = data_amnt_m.drop(['TENURE','PRC_FULL_PAYMENT','CASH_ADVANCE_TRX','PURCHASES_TRX'], axis =1)
cols  = list(data_amnt_m_sub1.columns)[1:9]
data_amnt_m_sub11   = pd.melt(data_amnt_m_sub1, id_vars = ['Cluster'],value_vars=cols ,var_name='cols')

## Frequency
cols1               = list(data_freq_m.columns)[1:5]
data_freq_m_sub11   = pd.melt(data_freq_m, id_vars = ['Cluster'],value_vars=cols1 ,var_name='cols')


## Agglomerative

## Amount 
data_amnt_m_sub2     = data_amnt_agg_m.drop(['TENURE','PRC_FULL_PAYMENT','CASH_ADVANCE_TRX','PURCHASES_TRX'], axis =1)
cols  = list(data_amnt_m_sub2.columns)[1:9]
data_amnt_m_sub22   = pd.melt(data_amnt_m_sub2, id_vars = ['Cluster'],value_vars=cols ,var_name='cols')

## Frequency
cols2               = list(data_freq_agg_m.columns)[1:5]
data_freq_m_sub22   = pd.melt(data_freq_agg_m, id_vars = ['Cluster'],value_vars=cols2 ,var_name='cols')


# In[ ]:


## Creating catplots for both the graphs
fig = plt.figure(figsize = (22,12))

## k-means Clustering

ax1 = fig.add_subplot(221)
g = sns.pointplot(x="cols", y="value", hue='Cluster', data=data_amnt_m_sub11, kind ="point", ax = ax1)
plt.title("Clusters across avg of Amount Variables", fontsize = 18)
plt.xlabel("Variables related to Amount", fontsize = 14)
plt.ylabel("Average", fontsize = 14)
g.set_xticklabels(g.get_xticklabels(), rotation=60)

ax2 = fig.add_subplot(222)
g = sns.pointplot(x="cols", y="value", hue='Cluster', data=data_freq_m_sub11, kind ="point", ax = ax2)
plt.title("Clusters across median of Frequency Variables", fontsize = 18)
plt.xlabel("Variables related to Frequency", fontsize = 14)
plt.ylabel("Median", fontsize = 14)
g.set_xticklabels(g.get_xticklabels(), rotation=60)


## Agglomerative Clustering

ax3 = fig.add_subplot(223)
g = sns.pointplot(x="cols", y="value", hue='Cluster', data=data_amnt_m_sub22, kind ="point", ax = ax3)
plt.title("Agglomerative Clustering", fontsize = 18)
plt.xlabel("Variables related to Amount", fontsize = 14)
plt.ylabel("Average", fontsize = 14)
g.set_xticklabels(g.get_xticklabels(), rotation=60)

ax4 = fig.add_subplot(224)
g = sns.pointplot(x="cols", y="value", hue='Cluster', data=data_freq_m_sub22, kind ="point", ax = ax4)
plt.title("Agglomerative Clustering", fontsize = 18)
plt.xlabel("Variables related to Frequency", fontsize = 14)
plt.ylabel("Median", fontsize = 14)
g.set_xticklabels(g.get_xticklabels(), rotation=60)

fig.tight_layout() 


# ### Conclusion: We conclude that k-means and Agglomerative clustering almost gives similar results even when we see them visually or see them when we profile them. Both of them have their pros and cons:
# 
# * k-means doesn't tell you the number of clusters. This has to be found by the user and then move ahead with the clusters
# * Agglomerative Clustering can be used with large sized data while k-means doesn't give accuracte results with large sized data
