#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Clustering data based on filling of column

import pandas as pd

# Load train data
df_train = pd.read_csv('../input/train.csv')
# Drop ID and target, they are not needed for the analys
df_train = df_train.drop(['ID', 'target'], axis=1)

# Similar for test
df_test = pd.read_csv('../input/test.csv')
df_test = df_test.drop(['ID'], axis=1)

# Concat both datasets
all_data = df_train.append(df_test, ignore_index=True)

all_data.info()


# Now we will find all columns with about 100% filling In the future you will see that we can split our set with 80% border

# In[ ]:


all_l = len(all_data)  # Full length

filled_lst = []   # 100% filled
empty_lst = []    # is not empty, but not filled

# Separate draw categorical (str) and float+int
x_int = []
y_int = []

x_str = []
y_str = []

for i,(name,series) in enumerate(all_data.iteritems()):
    c = series.count()  # series.count() return count of filled rows
    fill = c/all_l*100  
    #print('%s: %.2f  (type: %s)' % (name, fill, series.dtype))
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)
    #
    if fill>80:
        filled_lst.append(name)
    else:
        empty_lst.append(name)
    #        


# 100% filled list:

# In[ ]:


filled_lst


# not filled list (empty_lst) include all other columns

# In[ ]:


len(empty_lst)


# Now plot:

# In[ ]:


from matplotlib import pyplot as plt
plt.subplots(figsize=(10, 10))
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)
plt.axhline(80,color='r') # our 80% treshold

plt.show()


# Now we want to build affinity matrix. To do this we need to calculate the distance between each of the two dataseries
# 
# Calculate "distance" example:

# In[ ]:


tmp_row1 = list(range(10))
tmp_row2 = list(range(10,20))
tmp_row2[2:4] = [None,None]
tmp_row3 = list(range(20,30))
tmp_row3[2:6] = [None,None,None,None]
example_df = pd.DataFrame(data={'v1':tmp_row1, 'v2':tmp_row2, 'v3':tmp_row3})
example_df


# In[ ]:


import numpy as np
def dist(series1, series2, length):
    #Calculate correlation between data series
    c = series1.isnull().values == series2.isnull().values
    return np.sum(c.astype(int))/length


# In[ ]:


dist(example_df['v1'],example_df['v2'],10)


# Length of series = 10, and v1 have 8 elemnts filled simultaneously with v2

# In[ ]:


dist(example_df['v1'],example_df['v3'],10)


# v1 and v3: inly 6 elements filled simultaneously

# In[ ]:


dist(example_df['v1'],example_df['v2'],10)


# Again 8 elements (NaN - NaN also considered)

# Go ahead!
# 
# Let's build affinity matrix: it will be 102x102 matrix

# In[ ]:


# First of all: we need to drop filled data
# all data low filled
all_data_lf = all_data.drop(filled_lst, axis=1)


# In[ ]:


# Now build matrix
a = np.eye(len(empty_lst))  # This matrix already have 1 in main diagonal
length = len(all_data_lf)
#
for i, (name1, series1) in enumerate(all_data_lf.iteritems()):
    for j, (name2, series2) in enumerate(all_data_lf.iteritems()):
        if j == i:  # Only under main diag
            break
        else:
            tmp_d = dist(series1, series2, length)
            a[i,j] = tmp_d
            a[j,i] = tmp_d
#           
a.shape


# In[ ]:


size = len(empty_lst)
fig, ax = plt.subplots(figsize=(15, 15))
ax.matshow(a)
locs, labels = plt.xticks(range(size), empty_lst)
plt.setp(labels, rotation=90)
plt.yticks(range(size), empty_lst)
plt.show()


# How you can see, there is highly correlated filling. Only v30 and v113 have some differences

# Now, try to cluster our data
# 
# To do this we will use sklearn.cluster.SpectralClustering with setting: affinity='precomputed' 
# 
# And to choose the best partition (n_clusters) we will use sklearn.metrics.silhouette_score (and again metric='precomputed')

# In[ ]:


from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score

x_score = []
y_score = []

# Get scores for n_clusters from 2 to 10:
for i in range(2,10):
    tmp_clf = SpectralClustering(n_clusters=i, affinity='precomputed')
    tmp_clf.fit(a)
    score = silhouette_score(a, tmp_clf.labels_, metric='precomputed')
    x_score.append(i)
    y_score.append(score)

# Draw
plt.subplots(figsize=(10, 10))
plt.plot(x_score,y_score)
plt.grid()
plt.show()


# We use the Elbrow method to determining the number of clusters: 3 or 4
# 
# Let's print this clusters

# In[ ]:


# for 3 clusters:
clusters_count = 3
clusters = [[] for i in range(clusters_count)]
clf = SpectralClustering(n_clusters=clusters_count, affinity='precomputed', random_state=42)
clf.fit(a)

for name,cluster_n in zip(empty_lst, clf.labels_):
    clusters[cluster_n].append(name)
    
for tmp_cluster in clusters:
    print('---')
    print(tmp_cluster)


# If you change clusters_count to 4 you will see that v30 and v113 will be 3 and 4 clusters respectively. So I think that n_clusters=3 - best partition for this dataset
# 
# What does this partition? If the object is filled with at least one column of the cluster, the whole group will have a high level of filling.
# 
# Check this in the first cluster:

# In[ ]:


# This function return mask for DataFrame, elements wherein at least one filled column from list
def get_mask_notnull(df,columns_list):
    i = iter(columns_list)
    #Take first column in list
    first_v = next(i)
    #Get notnull mask
    current_mask = df[first_v].notnull()
    for tmp_v in i:
        current_mask = current_mask | df[tmp_v].notnull() #logical "or"
    #
    return current_mask

# Get elements from first cluster
df_first_cluster = all_data_lf[get_mask_notnull(all_data_lf, clusters[0])]
print('objects count from cluster 1: %d' % len(df_first_cluster))

# And draw filling percentage as in the beginning of script

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_first_cluster)
for i,(name,series) in enumerate(df_first_cluster.iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)
    
plt.subplots(figsize=(10, 10))  
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()


# You can see, that columns from cluster 1 almost completely filled
# 
# 
# Actually, even if we combine cluster1+cluster2 we obtain good filling:

# In[ ]:


# Get elements from combine (1+2) cluster
df_combine = all_data_lf[get_mask_notnull(all_data_lf, clusters[0]+clusters[1])]
print('objects count from clusters 1 and 2: %d' % len(df_combine))

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_combine)
for i,(name,series) in enumerate(df_combine.iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)

plt.subplots(figsize=(10, 10)) 
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()


# but if you need a more accurate separation is better to use these clusters separately
# 
# 

# About 3 cluster ['v30', 'v113']: this cluster have very low filling

# In[ ]:


# Get elements from 2 cluster
df_third = all_data_lf[get_mask_notnull(all_data_lf, clusters[2])]
print('objects count from cluster 3: %d' % len(df_third ))

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_third )
for i,(name,series) in enumerate(df_third .iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)

plt.subplots(figsize=(10, 10))         
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()


# As you can see it have more high filling for 2 cluster, but other columns have low filling

# CONCLUSION 
# 
# It may be useful to consider the problem, dividing it into three parts: 
# 
# 1) filled columns only from 'filled_lst' 
# 
# 2) filled cluster 1 (clusters[0]) 
# 
# 3) filled cluster 2 (clusters[1])
# 
# 
# And in the end I want to draw from (1 variance):

# In[ ]:


df_high_filled = all_data[~get_mask_notnull(all_data, clusters[0]+clusters[1])]
print('only high filled objects: %d' % len(df_high_filled))

x_int = []
y_int = []

x_str = []
y_str = []

all_l=len(df_high_filled)
for i,(name,series) in enumerate(df_high_filled.iteritems()):
    c = series.count()
    fill = c/all_l*100  
    if series.dtype == 'O':
        x_str.append(i)
        y_str.append(fill)
    else:
        x_int.append(i)
        y_int.append(fill)

plt.subplots(figsize=(10, 10))         
plt.plot(x_int,y_int,'o',color = 'red', markersize = 10, alpha = 0.3)
plt.plot(x_str,y_str,'o',color = 'green', markersize = 10, alpha = 0.3)

plt.show()

