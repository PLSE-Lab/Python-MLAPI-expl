#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this notebook, we will take a look at the the base statistics of Pokemons from Generation I to VII.
# 
# Simple visualisations of each stats will be shown first. Afterwards, we will perform clustering to find out if there are any interesting clusters that we can find.

# <a id='content'></a>
# # Contents
# 
# 1. [Stats summary](#stats_sum)
# 2. [Stats summary by type](#stats_sum_type)
# 3. [Stats correlation](#stats_cor)
# 4. [Clustering with KMeans](#clust_kmeans)
# 5. [Clustering with binned stats](#clust_bin)
# 6. [Clustering with binned stats using KMeans](#clust_kmbin)
# 7. [Trying with different number of clusters](#n_clust)

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('../input/pokemon.csv')
print(df.shape)
df.info()


# In[ ]:


df.head()


# In[ ]:


df.Total.describe()


# <a id='stats_sum'></a> 
# ## Stats summary
# In this section we look at the distribution plots of each stats:
# 
# 1. Total
# 2. HP
# 3. Attack
# 4. Defense
# 5. Sp. Atk
# 6. Sp. Def
# 7. Speed

# In[ ]:


sns.distplot(df.Total)
plt.show()


# In this plot for total stats, we can see that there are three distinct peaks in the histogram. The peak near 300 is probably due to common Pokemons still in the first stage of their evolution line, while the peak near 500 is probably due to Pokemons in their later stages of evolution. The peak near 700 is probably due to the Mythical and Legendary Pokemons, which usually have very high total stats.

# In[ ]:


sns.distplot(df.HP)
plt.show()


# In[ ]:


sns.distplot(df.Attack)
plt.show()


# In[ ]:


sns.distplot(df.Defense)
plt.show()


# In[ ]:


sns.distplot(df['Sp. Atk'])
plt.show()


# In[ ]:


sns.distplot(df['Sp. Def'])
plt.show()


# In[ ]:


sns.distplot(df['Speed'])
plt.show()


# In[ ]:


df[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']].describe()    


# From the above summary and plots, we can see that all the basic stats have very similar distribution, which is a right-skewed distribution (has a longer right tail).
# 
# [Back to content](#content)

# <a id='stats_sum_type'></a> 
# ## Stats summary by type
# 
# In this section, we further look at stats breakdown to each Pokemon types.

# In[ ]:


for pkm_type in df['Type 1'].unique():
    df[pkm_type] = df[['Type 1', 'Type 2']].apply(lambda x: 1 if pkm_type in x.values else 0, axis = 1)


# A summary function was written to summarise each stats according to each Pokemon type.

# In[ ]:


def summarise_stats(stats):
    
    summary_stats = []
    for i, pkm_type in enumerate(df['Type 1'].unique()):
        temp = df.loc[df[pkm_type]==1,stats].describe()
        temp.name = pkm_type
        summary_stats.append(temp)

    total_stats_summary = pd.concat(summary_stats, axis=1)
    
    print('Top types by average:')
    print(total_stats_summary.loc['mean'].sort_values(ascending=False).head(3))
    print(total_stats_summary.loc['50%'].sort_values(ascending=False).head(3))
        
    print('\nBottom types by average:')
    print(total_stats_summary.loc['mean'].sort_values(ascending=False).tail(3))
    print(total_stats_summary.loc['50%'].sort_values(ascending=False).tail(3))
    
    print('\nBiggest variance:')
    print(total_stats_summary.loc['std'].sort_values(ascending=False).head(3))
    
    plt_type=total_stats_summary.loc['mean'].sort_values(ascending=False).head(3).index.tolist()+total_stats_summary.loc['mean'].sort_values(ascending=False).tail(3).index.tolist()
    
    for i, pkm_type in enumerate(plt_type):
        sns.distplot(df.loc[df[pkm_type]==1,stats], label=pkm_type, hist=False)
    


# In[ ]:


summarise_stats('Total')


# From this summary of total stats, we can see that the Dragon type has the highest average stats. It can be clearly seen from the distribution plot that the Dragon type has a much higher average than the rest of the type. The difference from Dragon type and Steel type (which has the second highest average total stats) is around 50 stats while the difference from Steel type and Psychic type is only around 2 stats. This shows that the average Dragon type Pokemon is much more powerful than other Pokemons.
# 
# On the other end, Normal, Poison and Bug types are the three lowest average total stats. An average Bug type Pokemon clearly compete with an average Dragon type Pokemon.

# In[ ]:


summarise_stats('HP')


# In[ ]:


summarise_stats('Attack')


# In[ ]:


summarise_stats('Defense')


# In[ ]:


summarise_stats('Sp. Atk')


# In[ ]:


summarise_stats('Sp. Def')


# In[ ]:


summarise_stats('Speed')


# [Back to content](#content)

# <a id='stats_cor'></a>
# ## Stats correlation
# 
# In this section, we try to find if there is any correlation between stats through a pairplot.

# In[ ]:


sns.pairplot(df[['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']])
plt.show()


# From this pairplot, we cannot really see any strong correlation between two stats.
# 
# [Back to content](#content)

# <a id='clust_kmeans'></a>
# ## Clustering with KMeans
# 
# In this section, we look at clustering of Pokemons according to their stats.

# In[ ]:


from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer

# Instantiate the clustering model and visualizer
model = KMeans(max_iter=1000, random_state=42)
visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')

features = ['HP','Attack','Defense','Sp. Atk','Sp. Def','Speed']


# In[ ]:


X = df[features].values


# In[ ]:


visualizer.fit(X)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# From this plot, we can see that our data may not cluster very well.
# 
# Perhaps some scaling might help the data to cluster better.

# In[ ]:


from sklearn.preprocessing import MinMaxScaler


# In[ ]:


scaler = MinMaxScaler()
X1 = scaler.fit_transform(X)


# In[ ]:


visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
visualizer.fit(X1)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# Even with scaling done on the data, our data seems to be not well clustered. Anyhow, we will perform a simple clustering with 3 clusters.

# In[ ]:


model_kmeans = KMeans(n_clusters=3, max_iter=1000, random_state=42)


# In[ ]:


df['kmeans_group'] = model_kmeans.fit_predict(X)


# We look at the stats for the centroid of each cluster.

# In[ ]:


cluster_center = pd.DataFrame(model_kmeans.cluster_centers_)
cluster_center.columns = features
cluster_center['total'] = cluster_center.sum(axis=1)

cluster_center['ordered_label'] = cluster_center.total.rank().astype(int)
cluster_center.sort_values(by='ordered_label').set_index('ordered_label')


# In[ ]:


relabel = cluster_center.ordered_label.to_dict()
df.kmeans_group = df.kmeans_group.map(lambda x: relabel[x])


# In[ ]:


df.kmeans_group.value_counts()


# From the table above, we can see that Pokemons in cluster 1 are probably weaker Pokemons with low stats in generally. The total stats for the centroid is the lowest among the three clusters.
# 
# In contrast, Pokemons in cluster 3 are probably the stronger Pokemons, as the centroid has the highest total stats. 
# 
# We can also observe that the centroid for cluster 2 actually has highest Defense stats. It is much higher than the Defense stats of cluster 3 centroid. Maybe, the Pokemons in cluster 2 are bulkier Pokemons with much higher Defense. Perhaps, Pokemons in cluster 3 are attackers while Pokemons in cluster 2 are defenders.

# In[ ]:


df.loc[df.kmeans_group==1].sample(10, random_state=42)


# In[ ]:


df.loc[df.kmeans_group==2].sample(10, random_state=42)


# In[ ]:


df.loc[df.kmeans_group==3].sample(10, random_state=42)


# [Back to contents](#content)

# <a id='clust_bin'></a>
# ## Clustering with binned stats
# 
# In this section, we explore clustering by binning the stats of the Pokemons. We will be binning the stats using a quantile strategy i.e. each bin will have the same number of Pokemons.
# 
# We will be creating 3 bins for each stats.

# In[ ]:


def bin_value(some_series, bins=3):
    cumsum_series = some_series.value_counts().sort_index().cumsum()
    limits = [len(some_series)/bins*i for i in range(1, bins)]
    right_edge = [abs(cumsum_series-i).idxmin() for i in limits]
    return [sum([x>i for i in right_edge]) for x in some_series]


# In[ ]:


X2 = df[features].apply(bin_value)


# In[ ]:


visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
visualizer.fit(X2)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# In[ ]:


model_kmeans_bins = KMeans(n_clusters=3, max_iter=1000, random_state=42)


# In[ ]:


df['kmeans_bin_group'] = model_kmeans_bins.fit_predict(X2)


# In[ ]:


cluster_center_bins = pd.DataFrame(model_kmeans_bins.cluster_centers_)
cluster_center_bins.columns = features
cluster_center_bins['total'] = cluster_center_bins.sum(axis=1)
cluster_center_bins['ordered_label'] = cluster_center_bins.total.rank().astype(int)

cluster_center_bins.sort_values(by='ordered_label').set_index('ordered_label')


# In[ ]:


relabel_bins = cluster_center_bins.ordered_label.to_dict()
df.kmeans_bin_group = df.kmeans_bin_group.map(lambda x: relabel_bins[x])


# In[ ]:


df.kmeans_bin_group.value_counts().sort_index()


# In[ ]:


df.kmeans_group.value_counts().sort_index()


# In[ ]:


df['group_combi'] = df.iloc[:,-2:].astype(str).apply(lambda x: ''.join(x), axis=1)


# In[ ]:


group_count = df.group_combi.value_counts().sort_index()
group_count


# In[ ]:


from sklearn.metrics import adjusted_mutual_info_score, normalized_mutual_info_score, adjusted_rand_score, confusion_matrix


# In[ ]:


print(adjusted_mutual_info_score(df['kmeans_bin_group'], df['kmeans_group']))
print(adjusted_rand_score(df['kmeans_bin_group'], df['kmeans_group']))


# In[ ]:


confusion_matrix(df['kmeans_bin_group'], df['kmeans_group'])


# We can see that there are some differences arising from clustering using binned values. We can see some Pokemons that has a big change in their group i.e. Pokemons which was originally in cluster 1, but now in cluster 3, an Pokemons which was originally in cluster 3, but now in cluster 1.?

# In[ ]:


df.loc[df.group_combi=='13']


# In[ ]:


df.loc[df.group_combi=='31']


# We can also change how many bins we want for each stats. We will try with 4, 5 and 10 bins and compare the clustering results.

# In[ ]:


X2_4 = df[features].apply(bin_value, bins=4)


# In[ ]:


visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
visualizer.fit(X2_4)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# In[ ]:


model_kmeans_4bins = KMeans(n_clusters=3, max_iter=1000, random_state=42)
df['kmeans_4bins_group'] = model_kmeans_4bins.fit_predict(X2_4)

cluster_center_4bins = pd.DataFrame(model_kmeans_4bins.cluster_centers_)
cluster_center_4bins.columns = features
cluster_center_4bins['total'] = cluster_center_4bins.sum(axis=1)
cluster_center_4bins['ordered_label'] = cluster_center_4bins.total.rank().astype(int)

cluster_center_4bins.sort_values(by='ordered_label').set_index('ordered_label')


# In[ ]:


relabel_4bins = cluster_center_4bins.ordered_label.to_dict()
df.kmeans_4bins_group = df.kmeans_4bins_group.map(lambda x: relabel_4bins[x])


# In[ ]:


df.kmeans_4bins_group.value_counts().sort_index()


# In[ ]:


print(adjusted_mutual_info_score(df['kmeans_4bins_group'], df['kmeans_group']))
print(adjusted_rand_score(df['kmeans_4bins_group'], df['kmeans_group']))


# In[ ]:


confusion_matrix(df['kmeans_4bins_group'], df['kmeans_group'])


# As before, we look at Pokemons that has a large switch in their cluster.

# In[ ]:


df.loc[(df.kmeans_group==1)&(df.kmeans_4bins_group==3)]


# In[ ]:


df.loc[(df.kmeans_group==3)&(df.kmeans_4bins_group==1)]


# In[ ]:


X2_5 = df[features].apply(bin_value, bins=5)


# In[ ]:


def build_cluster_bins(X, bins=3):
    
    model = KMeans(n_clusters=3, max_iter=1000, random_state=42)
    df['kmeans_{}bins_group'.format(bins)] = model.fit_predict(X)

    cluster_center_df = pd.DataFrame(model.cluster_centers_)
    cluster_center_df.columns = features
    cluster_center_df['total'] = cluster_center_df.sum(axis=1)
    cluster_center_df['ordered_label'] = cluster_center_df.total.rank().astype(int)
    relabel = cluster_center_df.ordered_label.to_dict()
    df['kmeans_{}bins_group'.format(bins)] = df['kmeans_{}bins_group'.format(bins)].map(lambda x: relabel[x])
    return cluster_center_df.sort_values(by='ordered_label').set_index('ordered_label')


# In[ ]:


visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
visualizer.fit(X2_5)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# In[ ]:


build_cluster_bins(X2_5, bins=5)


# In[ ]:


print(adjusted_mutual_info_score(df['kmeans_5bins_group'], df['kmeans_group']))
print(adjusted_rand_score(df['kmeans_5bins_group'], df['kmeans_group']))
confusion_matrix(df['kmeans_5bins_group'], df['kmeans_group'])


# In[ ]:


df.loc[(df.kmeans_group==1)&(df.kmeans_5bins_group==3)]


# In[ ]:


df.loc[(df.kmeans_group==3)&(df.kmeans_5bins_group==1)]


# In[ ]:


X2_10 = df[features].apply(bin_value, bins=10)


# In[ ]:


visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
visualizer.fit(X2_10)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# In[ ]:


build_cluster_bins(X2_10, bins=10)


# In[ ]:


print(adjusted_mutual_info_score(df['kmeans_10bins_group'], df['kmeans_group']))
print(adjusted_rand_score(df['kmeans_10bins_group'], df['kmeans_group']))
confusion_matrix(df['kmeans_10bins_group'], df['kmeans_group'])


# In[ ]:


df.loc[(df.kmeans_group==1)&(df.kmeans_10bins_group==3)]


# In[ ]:


df.loc[(df.kmeans_group==3)&(df.kmeans_10bins_group==1)]


# In[ ]:


df['nunique_group'] = df[[x for x in df.columns if 'bin' in x]].apply(pd.Series.nunique, axis=1)

df.nunique_group.value_counts()


# In[ ]:


df.loc[df.nunique_group==3]


# With different number of bins, there are some differences in the clustering results. 754 of them ended up with the same clusters, 157 Pokemons had two different clusters assigned. Lastly 6 Pokemons had 3 different clusters assigned to them.
# 
# [Back to content](#content)

# <a id='clust_kmbin'></a>
# ## Clustering with binned stats using KMeans
# 
# In this section, instead of binning the stats using a quantile strategy, we will bin them by performing KMeans clustering on each of the stats.

# In[ ]:


for x in features:
    visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
    visualizer.fit(df[x].values.reshape(-1,1))    # Fit the data to the visualizer
    print(x)
    visualizer.poof()    # Draw/show/poof the data


# Based on the graphs above, we will choose the optimal number of clusters for each stats.
# 
# 1. HP: 5
# 2. Attack: 7
# 3. Defense: 5
# 4. Sp. Atk: 3
# 5. Sp. Def: 7
# 6. Speed: 7

# In[ ]:


num_bins = [5,7,5,3,7,7]
bins_required = {x:num_bins[i] for i, x in enumerate(features)}


# In[ ]:


bins_stats = {}
stats_cluster_centers = {}
for x in features:
    model_stats = KMeans(n_clusters=bins_required[x], random_state=42)
    bins_stats[x] = model_stats.fit_predict(df[x].values.reshape(-1,1))
    stats_cluster_centers[x] = model_stats.cluster_centers_


# In[ ]:


stats_relabel = {}
for x in features:
    df1 = pd.DataFrame({x:stats_cluster_centers[x].flatten()})
    df1['ordered_label'] = df1.rank().astype(int)
    print('\n',df1.set_index('ordered_label').sort_index())
    stats_relabel[x] = df1.ordered_label.to_dict()


# In[ ]:


stats_relabel


# In[ ]:


temp = pd.DataFrame(bins_stats)
for x in features:
    temp[x] = temp[x].map(lambda y: stats_relabel[x][y])
    
temp.head()


# In[ ]:


for x in features:
    print('\n',temp[x].value_counts().sort_index())


# In[ ]:


X3 = temp


# In[ ]:


visualizer = KElbowVisualizer(model, k=(2,10), metric='silhouette')
visualizer.fit(X3)    # Fit the data to the visualizer
visualizer.poof()    # Draw/show/poof the data


# In[ ]:


model_kmeans_bins_km = KMeans(n_clusters=3, max_iter=1000, random_state=42)

df['kmeans_kmbin_group'] = model_kmeans_bins_km.fit_predict(X3)

cluster_center_bins_km = pd.DataFrame(model_kmeans_bins_km.cluster_centers_)
cluster_center_bins_km.columns = features
cluster_center_bins_km['total'] = cluster_center_bins_km.sum(axis=1)
cluster_center_bins_km['ordered_label'] = cluster_center_bins_km.total.rank().astype(int)

cluster_center_bins_km.sort_values(by='ordered_label').set_index('ordered_label')


# In[ ]:


relabel_bins_km = cluster_center_bins_km.ordered_label.to_dict()
df.kmeans_kmbin_group = df.kmeans_kmbin_group.map(lambda x: relabel_bins_km[x])


# In[ ]:


df.kmeans_kmbin_group.value_counts().sort_index()


# In[ ]:


print(adjusted_mutual_info_score(df['kmeans_kmbin_group'], df['kmeans_group']))
print(adjusted_rand_score(df['kmeans_kmbin_group'], df['kmeans_group']))


# In[ ]:


confusion_matrix(df['kmeans_kmbin_group'], df['kmeans_group'])


# [Back to content](#content)

# <a id='n_clust'></a>
# ## Trying different number of cluster 
# 
# In this section, we try with different number of clusters. We will cluster with 5 and 7 groups, and compare their results to our initial 3-groups clusters.

# In[ ]:


def build_cluster_n(X, n_clust=3):
    
    model = KMeans(n_clusters=n_clust, max_iter=1000, random_state=42)
    df['kmeans_{}_group'.format(n_clust)] = model.fit_predict(X)

    cluster_center_df = pd.DataFrame(model.cluster_centers_)
    cluster_center_df.columns = features
    cluster_center_df['total'] = cluster_center_df.sum(axis=1)
    cluster_center_df['ordered_label'] = cluster_center_df.total.rank().astype(int)
    relabel = cluster_center_df.ordered_label.to_dict()
    df['kmeans_{}_group'.format(n_clust)] = df['kmeans_{}_group'.format(n_clust)].map(lambda x: relabel[x])
    return cluster_center_df.sort_values(by='ordered_label').set_index('ordered_label')


# In[ ]:


build_cluster_n(X)


# In[ ]:


result = {}
for n_clust in [9, 11, 14, 17, 19]:
    result[n_clust] = build_cluster_n(X, n_clust=n_clust)


# In[ ]:


from sklearn.metrics import silhouette_samples


# In[ ]:


for n_clust in [9, 11, 14, 17, 19]:
    df['silhouette_{}'.format(n_clust)] = silhouette_samples(X, df['kmeans_{}_group'.format(n_clust)])


# In[ ]:


df.groupby('kmeans_9_group').silhouette_9.mean()


# In[ ]:


df.loc[df.kmeans_9_group==9,['Name','silhouette_9']].sort_values(by='silhouette_9')


# In[ ]:




