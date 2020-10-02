#!/usr/bin/env python
# coding: utf-8

# # Purpose
# 
# [disclaimer: newbie in data science, working in finance. Any comment/help is welcome :)]
# 
# Watching you guy's kernel and trying to play myself with the data, I noticed that there is a high number of missing values. 
# 
# What I am trying to do here is to check if the assumption that ids in the dataset correspond to the same type of financial securities holds. The idea behind this is that certain variables does not make sense for different securities.
# 
# For instance, it would make no sense for an equity security to have an interest rate (putting aside preferred shares). Even for securities that "looks" the same, you may have differences. Assuming that "fundamental" variables correspond to data concerning the issuer, calculating the GDP of a corporate bond issuer makes no sense, and governments do not have a US-GAAP/IFRS P&L. 
# 
# That being said, I would like to see if I can create clusters of ids solely based on the appearance of certain variables.
# 
# Let's try!
# 
# # Conclusion so Far
# 
# - Using a very naive approach confirmed that clustering according missing values could potentially yield interesting results
# - A heatmap confirmed the previous result and suggested that about 5 clusters would be sufficient
# - Using the silouhette score as a metric suggest that 8 clusters would be optimal
# - When plotting the distribution of y for each cluster, we recover the Gaussian shape present in the original data
# 
# # Work in Process
# 
# Now that I found a way to create clusters, my goal is to assess these clusters.
# 
# # Getting Data
# 
# First, let's import some libraries and retrieve the data.

# In[ ]:


#Importing libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')

#Getting data

#I owe this to SRK's work here:
#https://www.kaggle.com/sudalairajkumar/two-sigma-financial-modeling/simple-exploration-notebook/notebook
with pd.HDFStore("../input/train.h5", "r") as train:
    df = train.get("train")


# I will create some variables that might be useful later.

# In[ ]:


input_variables = [x for x in df.columns.values if x not in ['id','y','timestamp']]

print("Number of predicting variables: {0}".format(len(input_variables)))


# # A Naive Implementation
# 
# My first try will be rather simple:
# 
#  1. First, I create a boolean matrix indicating whether a given variable is present (non-null and non-zero) for a given id
#  2. Then, I check how many unique line I obtain
# 
# ## Formatting Data
# 
# Let us create the dataframe. In this case, I will consider missing values and zeros to be the same. Thus, I begin by transforming all null values to 0.

# In[ ]:


df_id_vs_variable = df[['id']+input_variables]       #Removes 'y' and 'timestamp'
df_id_vs_variable = df_id_vs_variable.fillna(0)      #Replace na by 0

df_id_vs_variable.head()


# Then I aggregate the dataframe by id and sum the variables.

# In[ ]:


def makeBinary(x):
    if abs(x) > 0.00000:
        return 1
    else:
        return 0

df_id_vs_variable = df_id_vs_variable.groupby('id').agg('sum').applymap(makeBinary)

df_id_vs_variable.head()


# ## Using our New Dataframe
# 
# Let's try to make something out of this dataframe. First, let's get unique rows and try to see how many ids match them.

# In[ ]:


df_unique_set_variables = df_id_vs_variable.drop_duplicates(keep="first")

print("Number of securities: {0}".format(df_id_vs_variable.shape[0]))
print("Number of unique lines: {0}".format(df_unique_set_variables.shape[0]))


# It seems that with this naive implementation, it is possible to categorize our securities within 809 "clusters". This is not really efficient, but before to see if I can go any further, I will try to see if some clusters are larger than others.

# In[ ]:


#These lines do not correspond to any "cluster"
df_no_cluster = df_id_vs_variable.loc[~df_id_vs_variable.duplicated(input_variables,keep=False)]

#These lines are duplicated so they can be "clustered"
df_cluster = df_id_vs_variable.loc[df_id_vs_variable.duplicated(input_variables,keep=False)]

df_cluster = df_cluster.groupby(input_variables).size()

array_cluster = df_cluster.values

print("Number of securities that do not belong to a cluster:{0}".format(df_no_cluster.shape[0]))
print("Number of clusters: {0}".format(len(array_cluster)))
print("Number of securities that belong to a cluster: {0}".format(sum(array_cluster)))
print("##########################")
print("   Clusters Statistics")
print("##########################")
print(df_cluster.describe())
n, bins, patches = plt.hist(array_cluster, 50, normed=1, facecolor='green', alpha=0.75)
plt.xlabel('Cluster size')
plt.ylabel('Distribution')
plt.title(r'Distribution of Cluster Size')
plt.show()


# ## Conclusion
# 
# Using a naive implementation, I found that:
# 
# - Considering only the presence/absence of data, we could consider that 809 types of line were present
# - Among them, 723 are unique and can't be clustered
# - The remaining 701 securities can be grouped in 86 clusters
# - 75% of these cluster contain less than 6 securities. However, the largest ones contain more than 70 ids.
# 
# This suggest that some variables may be relevant to only certain securities, which implies that these securities are inherently different.
# 
# Following this conclusion, I intend to use more sophisticated methods to reduce the number of clusters.
# 
# # Unsupervised Learning: Building on the Naive Implementation
# 
# From now on, I will keep the same idea that the presence/absence of variable might be explained by the type of variables. My first try will be to use the binary matrix that I previously obtained and to see what I can get from this using clustering techniques.
# 
# ## Visualising the Situation
# 
# Let's begin with a heatmap.

# In[ ]:


sns.clustermap(df_id_vs_variable.transpose())


# Visually, I would expect 5 clusters (4 + 1 for uncategorisable ids) to fit. 
# 
# ## Clustering
# 
# In order to assess this visual result, I plot a silouhette score for the k-means algorithm. 

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

#Range for k
kmin = 2
kmax = 15
sil_scores = []

#Compute silouhette scoeres
for k in range(kmin,kmax):
    km = KMeans(n_clusters=k, n_init=20).fit(df_id_vs_variable)
    sil_scores.append(silhouette_score(df_id_vs_variable, km.labels_))

#Plot
plt.plot(range(kmin,kmax), sil_scores)
plt.title('KMeans Results')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.show()


# Putting aside the pike for two clusters, this analysis confirms that 3 to 8 clusters would be optimal. 
# 
# ## Assessing The Cluster Design
# 
# Using the number of cluster for the highest silouhette score excepting two clusters (i.e. 8 clusters) I intend to see if this clustering makes sense.
# 
# Drawing on the work of [Raphael Suter](https://www.kaggle.com/rasuter/two-sigma-financial-modeling/a-basic-approch-to-cluster-securities), I'll first try to see what the y distribution within each cluster looks like.
# 
# ### Splitting Dataframes Between Clusters
# 
# First, let's create the unsupervised learning model from previous insights.

# In[ ]:


n_clust = 8

km = KMeans(n_clusters=n_clust, n_init=20).fit(df_id_vs_variable)
clust = km.predict(df_id_vs_variable)


# Then, we retrieve the indexes in the original dataframe for each cluster.

# In[ ]:


#Init table of indexes
df_clust_index = {}
for i in range(0,n_clust):
    df_clust_index[i]=[]

#Fill the cluster index
for i in range(0,len(clust)):
    df_clust_index[clust[i]].append(i)

for i in range(0,n_clust):
    df_clust_index[i] = df_id_vs_variable.iloc[df_clust_index[i]].index.values


# Then we store each dataframe in a list.

# In[ ]:


df_clust = []

for i in range(0,n_clust):
    df_clust.append(df.loc[df.id.isin(df_clust_index[i])])


# ### Results
# 
# Let's have a look to the repartition of ids between each cluster.

# In[ ]:


for i in range(0,n_clust):
    print(df_clust[i].shape[0])


# Let's have a look to the y value distribution within clusters.

# In[ ]:


for i in range(0,n_clust):
    n, bins, patches = plt.hist(df_clust[i].y.values, 50, normed=1, facecolor='green', alpha=0.75)
    plt.xlabel('y Value')
    plt.ylabel('Occurence')
    plt.title(r'Distribution of y Value for Cluster '+str(i))
    plt.show()

