#!/usr/bin/env python
# coding: utf-8

# Hey, let me start by saying i am new to machine learning. So a constructive feedback would really be helpful.
# What i am trying to achieve here is create a playlist of similar songs together.
# 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.cluster import KMeans

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('/kaggle/input/ultimate-spotify-tracks-db/SpotifyFeatures.csv')
data.head()


# In[ ]:


data.describe()


# Looking for nulls. Fortunately there aren't any.

# In[ ]:


# columns = data.columns.values
# for col in columns:
#     df_null = data.isnull().groupby(['genre']).size()
#     print(df_null.head())

data.isnull().sum()


# In[ ]:


data.info()


# In[ ]:


for col in data.columns.values:
    print(col,'\t',data[col].nunique())


# Checking for correlation between the columns

# In[ ]:


plt.figure(figsize=(10,8))
sns.heatmap(data.corr(),annot=True)


# Checking different genre present.
# and also looking for if all of genres are equal or not

# In[ ]:


bar_cols = data[['genre','key','mode','time_signature']].columns.values
for col in bar_cols:
    df_temp = data.groupby([col]).size().reset_index(name='count')
    plt.figure(figsize=(18,8))
    plt.xticks(rotation=45)
    sns.set_style("ticks")
    sns.barplot(data = df_temp, x= col, y= 'count')


# Picking attributes that can help find similar music

# In[ ]:


# 'popularity','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence'
# dropping popularity adding key and mode
# data = data[data['popularity'] > 50]
X = data[['genre','acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']]
X.head()
# len(X)


# One hot encoding genre attribute

# In[ ]:


dummy_genre = pd.get_dummies(X['genre'],drop_first=True)
X = pd.concat([dummy_genre, X.iloc[:,1:]], axis=1)
X.head()


# Scaling all attributes together. 
# Loudness and tempo had high values so they would impact the dataset. Instead reduced them to be equal to other attributes. So all cols have similar equal values
# 
# Reason for multiplying all be 10 -> Most of the values were less than 1. Runnning clustering algorithms were giving memory issues when they were less so instead upscaled it. I assume its because in clustering algorithm all values will be closer at each neighoour and which would require more memory to compute

# In[ ]:


X['loudness'] = X['loudness']/10.0
X['tempo'] = X['tempo']/100.0
print(X.columns.values)
l_ofcols = ['acousticness','danceability','energy','instrumentalness','liveness','loudness','speechiness','tempo','valence']
for cols in l_ofcols:
    X[cols] = X[cols] * 10.0
    print(cols,'\tMAX\t',X[cols].max(),'\tMIN\t',X[cols].min())


# In[ ]:


df_artist = data.groupby(['artist_name']).size().reset_index(name='count').sort_values(['count'],ascending = False)
print(df_artist.head(25))


# In[ ]:


# 
# import time
# wcss = []
# print('Started\n')
# for i in range(1,50):
#     kmeans = KMeans(n_clusters = i,random_state = 0,n_jobs = -1)
#     y = kmeans.fit(X)
#     wcss_temp = kmeans.inertia_
#     if(len(wcss)>0):
#         print(time.ctime()," done for \t",i," with wcss as \t",wcss_temp," diff \t",(wcss[-1] - wcss_temp))
#     else:
#         print(time.ctime()," done for \t",i," with wcss as \t",wcss_temp," diff \t",(wcss_temp))
#     wcss.append(wcss_temp)

# plt.figure(figsize=(18,8))
# plt.plot(range(1,50),wcss)

# kmeans = KMeans(n_clusters = 10,random_state = 0,n_jobs = -1)
# clusters = kmeans.fit_predict(X)
# print(clusters)


# In[ ]:


# from sklearn.cluster import AgglomerativeClustering
# x_limit = X.iloc[:50000,:]
# # x_limit.head()
# agg_clustering = AgglomerativeClustering(n_clusters = 2)
# agg_clusters = agg_clustering.fit_predict(x_limit)
# print(agg_clusters)


# In[ ]:


# from sklearn.cluster import DBSCAN
# dbs_clustering = DBSCAN(eps= 1.10)
# dbs_clusters = dbs_clustering.fit_predict(X)
# print(dbs_clusters)


# In[ ]:


# from sklearn.cluster import OPTICS
# import time

# print('Start at ',time.ctime())
# opt_clustering = OPTICS(min_samples = 2,max_eps= 12.0,xi = 0.05,min_cluster_size=5,n_jobs = -1)
# # opt_clustering = OPTICS(n_jobs = -1)
# opt_clusters = opt_clustering.fit_predict(X_temp)
# print(np.unique(opt_clusters))
# print('End at ',time.ctime())


# For getting the desired output. I tried a few clustering algorithms.
# * k-means -> I figured out it wouldnt work as the number of clusters will be low. If we go with it we could max generate 100 playlist
# * AgglomerativeClustering -> Memory issue had to reduce the dataset to half to make it work.
# * OPTICS -> It took way to long(6 hrs+) to process but output was not great.
# 
# So after trial and testing above algorithms I tried Birch
# 

# In[ ]:


from sklearn.cluster import Birch
import time

print('Start at ',time.ctime())
birch_clustering = Birch(threshold = 3.0,branching_factor = 50,n_clusters = None ,compute_labels = True)
# opt_clustering = OPTICS(n_jobs = -1)
birch_clusters = birch_clustering.fit_predict(X)
print(np.unique(birch_clusters))
print('End at ',time.ctime())


# In[ ]:


# unique,counts = np.unique(birch_clusters,return_counts = True)
# print(type(unique),type(counts))
# plt.figure(figsize=(22,8))
# plt.xticks(rotation=90)
# sns.set_style("ticks")
# sns.barplot(x=unique,y=counts)


# In[ ]:


# t_df = birch_clusters
# plt.figure(figsize=(18,8))
# plt.xticks(rotation=45)
# sns.set_style("ticks")
# plot_cluster_df = pd.DataFrame({'cluster':t_df[:]})
# plot_cluster_df = plot_cluster_df.groupby(['cluster']).size().reset_index(name = 'count')
# sns.barplot(data = plot_cluster_df,x='cluster',y='count')


# Joining output of clusters to original dataset.

# In[ ]:


t_clusters = birch_clusters
output_df = data[['artist_name','track_name','track_id']]
output_df['cluster'] = t_clusters.tolist()
temp = output_df.groupby(['cluster']).size().reset_index(name = 'count').sort_values(['count'],ascending = False)
temp.head(10)


# In[ ]:


output_df[output_df['artist_name'] == 'Drake'].head(20)


# Let's look for similar songs to God's plan
# 
# There are 37 sons similar to it

# In[ ]:


cluster_look_up = 3084
output_df[output_df['cluster'] == cluster_look_up]['track_id'].nunique()


# Output of similar songs
# If we observe God's plan fall under 3 genres hip-hop,rap and pop.

# In[ ]:


s_ofsongs = output_df[output_df['cluster'] == cluster_look_up].iloc[:,2]
# print(s_ofsongs)
data[data['track_id'].isin(s_ofsongs)].sort_values(by=['popularity'],ascending = False).head(10)


# Below are list of genres for this cluster/playlist.

# In[ ]:


plot_data = data[data['track_id'].isin(s_ofsongs)][['genre','artist_name','track_name']]
# sns.barplot(data = plot_data,X = 'genre', y = plot_data.value_counts())


# In[ ]:


# unique,counts = np.unique(birch_clusters,return_counts = True)
# print(type(unique),type(counts))
plt.figure(figsize=(22,8))
plt.xticks(rotation=90)
sns.set_style("ticks")
# sns.barplot(x=unique,y=counts)
sns.barplot(data = plot_data, x= plot_data['genre'].unique(), y=plot_data['genre'].value_counts() )


# Below are artist with similar songs

# In[ ]:


# correct lable names
plt.figure(figsize=(22,8))
plt.xticks(rotation=90)
sns.set_style("ticks")
sns.barplot(data = plot_data, x= plot_data['artist_name'].unique(), y=plot_data['artist_name'].value_counts() )


# Below is unique list of all the sonns in this cluster

# In[ ]:


plot_data['List'] = plot_data['artist_name']+' -> '+plot_data['track_name']
plot_data['List'].unique()

