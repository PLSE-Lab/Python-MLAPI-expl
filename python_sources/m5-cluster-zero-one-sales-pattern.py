#!/usr/bin/env python
# coding: utf-8

# # Cluster zero / one sales pattern
# 
# Hi all :)
# 
# I am novice at Data Science and analysis. If you have any suggestion or advice, Please comment it. It will relly help me!!
# 
# I have tried to make some clusters with zero / one sales pattern based on the concept of **Jaccard similarity**, inspired by [this amazing kernel](https://www.kaggle.com/jpmiller/grouping-items-by-stockout-pattern)
# 
# The output of clusters for all ids might (i think) lead to feature engineering or new cross validation strategy or another new insights!
# 
# 
# The logic is
# 
# 1. Level 1 clustering using kernel density estimation based on missing values
#     - A wide range of missing values count for all ids => To compare zero / one sales pattern, grouping ids which have similar missing distribution into one cluster.
# 
# 2. Substitute original sales value
#     - Substitute nan for 0, 0 for -1, values > 0 for 1.
#     
# 3. Level 2 clustering
#     - Hierarchy Clustering level 1 clusters into more groups.
#     
#     
# Then, enjoy kaggle ~!

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os

import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from scipy.cluster.hierarchy import linkage, fcluster

import warnings
warnings.filterwarnings("ignore")


# I used `grid_part_1.pkl` file from [this great kernel](https://www.kaggle.com/kyakovlev/m5-simple-fe). 

# In[ ]:


grid_df = pd.read_pickle('/kaggle/input/m5-simple-fe/grid_part_1.pkl')


# In[ ]:


grid_df.head()


# Transform to original data set form

# In[ ]:


grid_df = grid_df[['id','d','sales']].pivot(index='id',columns='d').reset_index()
ids = pd.DataFrame(grid_df['id'])


# In[ ]:


grid_df = grid_df['sales'].iloc[:,:1913]

# values > 0 for 1, missing for nan, 0 for -1
grid_df = pd.DataFrame(np.where(grid_df.isnull(),np.nan,
                                np.where(grid_df > 0, 1, -1)))

grid_df.columns = [f'd_{i}' for i in range(1,1914)]


# In[ ]:


grid_df.head()


# In[ ]:


d1_peak = grid_df.notnull().sum(axis=1)
cluster = d1_peak.copy()


# Plot kde plot of the number of missing values over all ids
# 
# I set threshold below based on Heuristic. You can change it.

# In[ ]:


plt.figure(figsize=(12,6))
sns.kdeplot(d1_peak)
plt.plot([925,925],[0,0.0030]); plt.plot([1300,1300],[0,0.0030]); plt.plot([1700,1700],[0,0.0030])
plt.text(x=700,y=0.002,s='Cluster 1'); plt.text(x=1020,y=0.002,s='Cluster 2')
plt.text(x=1420,y=0.002,s='Cluster 3'); plt.text(x=2000,y=0.002,s='Cluster 4')
plt.title('# of Nan distribution')
plt.show()


# In[ ]:


c1_mask = (d1_peak <= 925)
c2_mask = (d1_peak > 925) & (d1_peak <= 1300)
c3_mask = (d1_peak > 1300) & (d1_peak <= 1700)
c4_mask = (d1_peak > 1700)

cluster[c1_mask] = 1
cluster[c2_mask] = 2
cluster[c3_mask] = 3
cluster[c4_mask] = 4


# In[ ]:


grid_df['cluster'] = cluster

# missing values for 0
grid_df = grid_df.fillna(0)


# In[ ]:


grid_df['cluster'].value_counts()


# As you can see on above graph, cluster 4 consists an half of all ids.
# 
# So, I tried to level 2 group only for cluster 4.
# 
# Before level 2 cluster, i will show you very simple example for how Jaccard similarity is calculated.

# ### Jaccard similarity calculation
# 
# If there are two binary feature A,B
# 
# `A = [1,0,0,0,1,0]`
# 
# `B = [1,1,0,0,0,0]`
# 
# Let's calculate Jaccard similarity

# In[ ]:


A = np.array([1,0,0,0,1,0])
B = np.array([1,1,0,0,0,0])


# In[ ]:


np.sum(A == B) / len(A)


# There are 4 same values over 6 on each index.
# 
# So, Jaccard similarity between A and B becomes 0.67
# 
# So simple and intuitive definition. Then Lets move onto level 2 cluster

# I tried two similarity, Jaccard and the score similar to Jaccard.
# 
# Lets take a look at these examples.

# ### 1. Jaccard similarity clustering

# In[ ]:


cluster_df = grid_df[grid_df['cluster'] == 1]
cluster_array = cluster_df.values
cluster_array = np.where(cluster_array == 0, np.nan, cluster_array)


# In[ ]:


length = cluster_array.shape[0] 
for i in tqdm(range(0, int(length/10))):
    for j in range(i, length):
        np.sum(cluster_array[i,:-1] == cluster_array[j,:-1])


# It takes 90 sec for 500 instances to calculate. 
# 
# cluster 1,2,3 have about 5000 instances respectively, cluster 4 has 15000 instances.
# 
# It will takes about an hour only to calculate distance matrix. 
# 
# So, i found out another way.

# ### 2. the index similar to Jaccard 
# 
# $
# Index = \frac{sum(A==B)~ -~ sum(A!=B)}{len(A)}
# $
# 
# I used this index by matrix inner product.
# 
# As i noted above, cluster 4 has 15000 instances, so i made level 2 cluster only for cluster 4

# In[ ]:


def Clustering(cluster_lv1_name, cluster_lv2_num):
    
    cluster_df = grid_df[grid_df['cluster'] == cluster_lv1_name]
    
    if cluster_lv2_num == 1:
        print('Pass : Cluster', cluster_lv1_name)
        
    else:
        print('Making dist_matrix : Cluster', cluster_lv1_name)
        cluster_array = cluster_df.values
        dist_matrix = np.dot(cluster_array, cluster_array.T)

        ## this part, linkage, takes about 30 minutes.
        ## If you have another idea for reducing running time,
        ## Please advise me !
        Z = linkage(dist_matrix, method='ward')
        cluster_num = fcluster(Z, t=cluster_lv2_num, criterion='maxclust')
        cluster_df['cluster'] = cluster_df['cluster'].astype(str) + '_' + cluster_num.astype(str)

    return cluster_df


# In[ ]:


plan_clustering = {
    #cluster_lv1_name : how many cluster_lv2 to make
    1:1,
    2:1,
    3:1,
    4:4
}


# In[ ]:


get_ipython().run_line_magic('time', '')
df_list = list()
for lv1, lv2 in plan_clustering.items():
    df_name = f'cluster_{lv1}_df'

    cluster_df = Clustering(cluster_lv1_name = lv1, cluster_lv2_num = lv2)
    globals()[df_name] = cluster_df
    
    df_list += [cluster_df['cluster']]


# In[ ]:


cls_total = pd.concat(df_list)


# In[ ]:


cluster_df = pd.concat([ids, cls_total], axis=1)


# In[ ]:


cluster_df


# In[ ]:


cluster_df['cluster'].value_counts()


# In[ ]:


cluster_df.to_pickle('zero_one_cluster.pkl')

