#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

seed = 0
np.random.seed(seed=seed)

import warnings
warnings.filterwarnings('ignore')


# In this notebook we will try to see if the players can be clustered, only using the FIFA skill grades (eg. "finishing", "dribbling" ...). The hypothesis is that the players with the same role will end up in the same cluster. 
# 
# We filter the goalkeepers because they have their own skillset.

# In[ ]:


players_df = pd.read_csv('../input/data.csv', index_col=0)

# The skills are rated with a grade up to 100
skills = players_df.columns[53:81]
print('Skills: ' + ', '.join(skills))

# Drop the lines containing NaN values
players_df.dropna(axis=0, inplace=True, subset=list(skills)+['Position'])


# In[ ]:


# Filter the goalkeepers, then count the remaining players
players_df = players_df[players_df.Position!='GK']
print('Number of players: {}'.format(players_df.shape[0]))


# We build `X`, our feature matrix. We normalize each line such that the sum of players' ratings is equal to 1. That way we can identify more easily their strong assets. 

# In[ ]:


X = players_df[skills].as_matrix()
# Normalization
X = X/np.sum(X, axis=1).reshape(-1,1)


# 
# We compare normalized ratings of two players with different roles: Griezmann (forward player) and Varane (defender). The features of the two players are quite different and consistent with their role on the pitch. Griezmann has batter rating than Varane on "finishing" (ability to score a goal with a few opportunities) but a worse on "interceptions", which is a defender skill. 

# In[ ]:


forward_features = X[players_df.Name=='A. Griezmann'].reshape(-1)
defender_features = X[players_df.Name=='R. Varane'].reshape(-1)

sort_idx = np.argsort(forward_features)[::-1]
forward_features = forward_features[sort_idx]
defender_features = defender_features[sort_idx]
skills_sorted = skills[sort_idx]

visualization_df = pd.DataFrame({
    'Norm. Ratings': np.concatenate((forward_features, defender_features)),
    'Skills': np.concatenate((skills_sorted, skills_sorted)),
    'Player': ['Griezmann (Forward)']*len(forward_features)+['Varane (Defender)']*len(defender_features)
})
f, ax = plt.subplots(figsize=(16, 8))
sns.barplot(x='Norm. Ratings', y='Skills', hue='Player', data=visualization_df)
sns.despine(left=True, bottom=True)


# We reduce the dimensions to 2, in order to see if we can already distinguish some clusters with a density plot. We can observe at least two modes. There is no "good answer" to find here, but visualizing the data gives clues about what we can find in the next steps. 

# In[ ]:


# Visualisation in 2 Dimensions
X_pca = PCA(n_components=2).fit_transform(X)
sns.kdeplot(X_pca[:,0],X_pca[:,1],cmap="Reds", shade=True, shade_lowest=False)
plt.show()


# Now we want to evaluate the number of clusters (ie. groups of players with a similar profile). We'll use a technique inspired by the paper from [Tibshirani et al, 2000](https://statweb.stanford.edu/~gwalther/gap), we could also use the well-known [elbow criterion](https://en.wikipedia.org/wiki/Elbow_method_(clustering) or [silhouette score](http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html). **The advantage of the former is that we can also evaluate the hypothesis "the data is not clustered" (ie. only 1 big cluster).**
# 
# The algorithm is pretty simple. First, we choose a plausible range of value containing the cluster number (generally between 1 and `kmax`). For each value `k` in this range, we build a score called "gap score". In the end, we select the number of cluster `k` corresponding to the highest gap score (there are a lot of other ways to select k depending on the situation, see [the part "method" in the R documentation](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/clusGap.html).
# 
# ![](https://tinyurl.com/y8yqnvao)
# 
# For calculating the gap score, we build a random dataset (ie. without cluster structure). The dataset contains the same number of points as the original, but for each dimension, the values are uniformly distributed between the same boundaries. An example below:

# In[ ]:


# Here we build a dataset that clearly contains 5 clusters
# We'll test our algorithm on this toy example
real_k = 5

x_coord = np.array([])
y_coord = np.array([])
for i in range(real_k):
    cluster_x_coord = np.random.normal(i*10, 1, 20)
    x_coord = np.concatenate([x_coord, cluster_x_coord])
    cluster_y_coord = np.random.normal(0, 1, 20)
    y_coord = np.concatenate([y_coord, cluster_y_coord])
X_toy = np.column_stack((x_coord, y_coord))

# plot
plt.scatter(x = X_toy[:,0], y = X_toy[:,1])
plt.axis('scaled')
plt.show()


# In[ ]:


def gen_rand_dataset(dataset):
    data_size, nb_dim = dataset.shape
    rand_dataset = []
    for dim in range(nb_dim):
        # For each dimension, we generate a uniform distribution of values between the original min and max.
        rand_dataset.append(np.random.uniform(min(dataset[:,dim]), max(dataset[:,dim]), data_size))
        
    return np.matrix(rand_dataset).T

# We give our toy dataset as input
X_rand = gen_rand_dataset(X_toy)
# plot
plt.scatter(x = X_rand[:,0].A1, y = X_rand[:,1].A1)
plt.axis('scaled')
plt.show()


# Now we have our function that is able to generate a "not-clustered" alter ego of our dataset. Given a `k`, we just have to apply k-means on the two datasets. The score is the difference of inertia between the two clustering. The inertia is given by sklearn, and corresponds to the sum of distances of each point to the centroid of the cluster it belongs to. The best the dataset is clustered, the lower the inertia. As k-means is designed to minimize inertia, we could expect that this value is higher for the random dataset (not clustered) than for the real dataset (maybe clustered). *Note that the original paper does not use inertia but within sum of square.* 
# 
# Because k-means depends strongly on the initial values, and that we use a randomly generated dataset, it is better to repeat the process `B` times and give back the mean value.

# In[ ]:


def get_gap(X_, k):
    X_rand = gen_rand_dataset(X_)
    inertia = KMeans(n_clusters=k).fit(X_).inertia_
    inertia_rand = KMeans(n_clusters=k).fit(X_rand).inertia_
    return inertia_rand - inertia

def get_gap_scores(X_, kmax, B = 5):
    gap_scores = []
    gap_k = []
    
    # Special case where k=1
    def get_inertia_unique_cluster(X__):
        centroid = np.mean(X__, axis=0)
        # Calculus of inertia
        return np.sum(np.apply_along_axis(lambda v: np.sum(np.square(v-centroid)), 1, X__))
    
    gap = get_inertia_unique_cluster(gen_rand_dataset(X_)) - get_inertia_unique_cluster(X_)
    gap_scores.append(gap)
    gap_k.append(1)
    
    # For k in [2,kmax]
    for k in range(2, kmax+1):
        gap = np.mean([get_gap(X_, k) for b in range(B)])
        gap_scores.append(gap)
        gap_k.append(k)
    
    return gap_k, gap_scores


# Let's try it with our toy example. After plotting, we can observe that our algorithm is able to find the right number of clusters (5). 

# In[ ]:


# Compute the gap scores
gap_k, gap_scores = get_gap_scores(X_toy, 10)
# Plot the gap score for each k
def plot_gap_score(gap_k, gap_scores):
    plt.plot(gap_k, gap_scores)
    plt.xlabel('k')
    plt.ylabel('Gap Score')
    plt.xticks(gap_k, gap_k)
    k_best = gap_k[np.argmax(gap_scores)]
    plt.axvline(x=k_best, color='r')
    plt.show()
    
plot_gap_score(gap_k, gap_scores)


# Let's apply this algorithm to real data now. Here we set `kmax=10` because there are 10 players on the pitch (we don't consider the goalkeepers here). As expected, a 2-clusters partition seems to hold the best score.

# In[ ]:


gap_k, gap_scores = get_gap_scores(X, 10)
plot_gap_score(gap_k, gap_scores)


# We divide the dataset into two clusters with the famous `KMeans`.

# In[ ]:


labels = KMeans(n_clusters=2, random_state=seed).fit(X).labels_
players_df['cluster'] = labels


# As the player roles depend on the strategic systems (eg. 4-4-2, 3-5-2 ...), it exists much more than 10 positions. In order to facilitate their understanding (and vizualisation), we'll assign to them XY coordinates. 
# 
# - **X** : From `1` (Defense) to `5` (Attack) 
# - **Y** : From `-1` (Left side) to `1` (Right side)

# In[ ]:


print('Positions: ' + ', '.join(players_df.Position.unique()))


# In[ ]:


players_df['Pos_coord_X'] = np.zeros(players_df.shape[0])
players_df.loc[players_df.Position.isin(['CB','LCB','RCB','RM','CM','CAM','CF','ST']),'Pos_coord_X'] = 0
players_df.loc[players_df.Position.isin(['LB','LWB','LDM','RCM','LM','LAM','LW','LF','LS']),'Pos_coord_X'] = -1
players_df.loc[players_df.Position.isin(['RB','RWB','RDM','LCM','RM','RAM','RW','RF','RS']),'Pos_coord_X'] = 1

players_df['Pos_coord_Y'] = np.zeros(players_df.shape[0])
players_df.loc[players_df.Position.isin(['RCB','LCB','CB','LB','RB']),'Pos_coord_Y'] = 1
players_df.loc[players_df.Position.isin(['CDM','LWB','RWB']),'Pos_coord_Y'] = 2
players_df.loc[players_df.Position.isin(['LDM','RDM','RCM','LCM','RM','CM','LM','RM','RAM','LAM']),'Pos_coord_Y'] = 3
players_df.loc[players_df.Position.isin(['RW','LW','CAM']),'Pos_coord_Y'] = 4
players_df.loc[players_df.Position.isin(['RF','LF','CF', 'ST','LS','RS']),'Pos_coord_Y'] = 5


# As we can see below, it seems that the partition separates defenders from forwards.

# In[ ]:


players_df.groupby(['Pos_coord_Y', 'cluster'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_Y', columns='cluster', values='ID').plot.bar(stacked=True)
plt.show()


# In[ ]:


players_df.groupby(['Pos_coord_X', 'cluster'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='cluster', values='ID').plot.bar(stacked=True)
plt.show()


# Now we will visualize how the players from both clusters are distributed with respect to the 2 dimensions at the same time. The heatmap can be seen as the football pitch (divided into 15 zones). For a given cluster, the percentage of players in this zone that belongs to this cluster is indicated.

# In[ ]:


total = players_df.groupby(['Pos_coord_X', 'Pos_coord_Y'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='Pos_coord_Y', values='ID').as_matrix()

def plot_distribution(distribution):
    fig, ax = plt.subplots()
    im = ax.imshow(distribution)
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(3))
    ax.set_xticklabels(['Defense','','Midfield','','Attack'])
    ax.set_yticklabels(['Left', 'Center', 'Right'])
    for i in range(3):
        for j in range(5):
            col = 'b' if distribution[i, j]>50 else 'w'
            text = ax.text(j, i, '{} %'.format(distribution[i, j]),
                           ha='center', va='center', color=col)


# In[ ]:


# Defenders cluster
distribution = players_df[players_df.cluster==0].groupby(['Pos_coord_X', 'Pos_coord_Y'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='Pos_coord_Y', values='ID').fillna(0).as_matrix()
distribution = np.round(distribution / total * 100)
plot_distribution(distribution)


# In[ ]:


# Forward players cluster
distribution = players_df[players_df.cluster==1].groupby(['Pos_coord_X', 'Pos_coord_Y'], as_index=False).agg({'ID': 'count'}).pivot(index='Pos_coord_X', columns='Pos_coord_Y', values='ID').fillna(0).as_matrix()
distribution = np.round(distribution / total * 100)
plot_distribution(distribution)


# The goal of this notebook was to show the use of "Gap Score" method from scratch. Even if a 2-clustered partition seems the optimal solution here, there are probably many other ways to group the players. I encourage you to:
# - Change K in order to get different numbers of partitions (and refine the analysis of players' similarities).
# - Visualize the [agglomerative clustering dendrogram](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.dendrogram.html) (hierarchical clustering). 
# - Use a density-based algorithm, like [DBScan](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html).
