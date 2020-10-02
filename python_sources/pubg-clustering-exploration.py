#!/usr/bin/env python
# coding: utf-8

# # A Quick Exploration of Clustering in the PUBG Dataset
# 
# Clustering can be a powerful tool, both for EDA and for feature engineering. I wanted to take a look at the PUBG data and see what kind of clusters were hiding in the data.

# In[ ]:


import gc
import time
# Data
import numpy as np
import pandas as pd

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import random
random.seed(1337)
np.random.seed(1337)

# Credit for this method here: https://www.kaggle.com/rejasupotaro/effective-feature-engineering
def reload():
    gc.collect()
    df = pd.read_csv('../input/train_V2.csv')
    invalid_match_ids = df[df['winPlacePerc'].isna()]['matchId'].values
    df = df[-df['matchId'].isin(invalid_match_ids)]
    return df


# In[ ]:


df = reload()
df.head()


# In[ ]:


# Memory saving function credit to https://www.kaggle.com/gemartin/load-data-reduce-memory-usage
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    #start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    #end_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    #print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
df = reduce_mem_usage(df)


# ## 1. Dimensionality Reduction
# 
# Clustering is **highly** sensitive to dimensionality, and also hard to visualize outside of 3-Dimensions. One thing I like to do when clustering is to use PCA to reduce the dimensionality of the data. You can see the example I used to build off of [here](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html#sphx-glr-auto-examples-decomposition-plot-pca-iris-py) but I'll try to explain what I'm doing as I go.

# ### 1.1 Normalize the Data
# 
# In my first version of this Kernel I didn't do any normalization. I'm changing that this iteration. I'm going to normalize the `kills`, `damageDealt`, `maxPlace` and `matchDuration` with the number of players in the match. Inspired by [this kernel](https://www.kaggle.com/carlolepelaars/pubg-data-exploration-rf-funny-gifs) by Carlo.

# In[ ]:


df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')


# In[ ]:


df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
df = reduce_mem_usage(df)
df.head()


# ### 1.2 Preprocess the Non-Numerical Columns
# 
# This dataset has some non-numerical columns and PCA can only take numerical values. Consequently, we need to one-hot encode or drop things we're not using for this. We're also going to drop the `winPlacePerc` since it's our target variable, we obviously don't want it in included in the PCA.
# 
# Additionally, while the -1 values for `rankPoints` should probably be considered NaN values according to the data description, PCA can't take NaN values, so I'm going to leave them as -1 for now.
# 

# In[ ]:


# Dropping columns with too many categories or unique values as well as the target column.
target = 'winPlacePerc'
drop_cols = ['Id', 'groupId', 'matchId', target]
select = [x for x in df.columns if x not in drop_cols]
X = df.loc[:, select]
X.head()


# In[ ]:


# Now one-hot encode the remaining category column (matchType)
X = pd.get_dummies(X)
X.head()


# ### 1.3 PCA With Two Components
# 
# Now we're going to try PCA with two components and see if we can see any clear clusters on the graph.**

# In[ ]:


from sklearn.decomposition import PCA


# In[ ]:


pca2 = PCA(n_components=2)
pca2.fit(X)


# In[ ]:


print(sum(pca2.explained_variance_ratio_))
P2 = pca2.transform(X)


# **Wowza!** That's a lot of variance explained using only two components, which is nice because it'll graph extrermely well in two dimensions!
# 
# Because we have several million datapoints, it would take forever for matplotlib to graph them all. So we'll just graph the first 100k, the distribution is the same (I checked!) and all of our math will still be done on the whole data.

# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1])
plt.show()


# Nice! You can see two pretty clear clusters in the data with two components! It's too early to tell whether this will be useful for building a model, but it's helpful to see nonetheless. Now lets try it with three dimensions to see if clusters show up there as well!

# ### 1.4 PCA With Three Components
# 
# Although our first PCA with two components explained quite a bit of the variance, it still fell short of the 80% variance rule of thumb, so I'm going to do some PCA again, but with three components this time! Similar steps to what we did last time, but we expect it to explain more of the variance due to the addition of another dimension.

# In[ ]:


pca3 = PCA(n_components=3)
pca3.fit(X)
print(sum(pca3.explained_variance_ratio_))
P3 = pca3.transform(X)


# Wow! That's almost all of the variance explained in only three components! That's wild! Lets see if there's any obvious clusters that pop out when we plot it... Once again, we're only plotting the first 100k in order to save time and memory.

# In[ ]:


from mpl_toolkits.mplot3d import Axes3D


# In[ ]:


fig_p3 = plt.figure()
ax = Axes3D(fig_p3, elev=48, azim=134)
ax.scatter(P3[:100000, 0], P3[:100000, 1], P3[:100000, 2])
fig_p3.show()


# Interesting. Sadly it doesn't look as if we got additional clusters from this data with three components, but it does reinforce what we saw on the earlier clusters: that there are two rather distinct clusters. We'll have to see if they're useful for creating a model later on!

# ## 2. Clustering Time
# 
# So now that we have reduced the dimensionality of the data, we can actually perform some clustering in order to get some engineered features out of the data. While we can see two pretty clear clusters at first glance, computers aren't that smart without some coaching.
# 
# ### 2.1 K-Means
# 
# The first clustering method we're going to use is K-means. I'm hoping it will converge on two clusters, the top and bottom clusters we can see in the scatter plots.

# In[ ]:


from sklearn.cluster import KMeans


# In[ ]:


kms = KMeans(n_clusters=2).fit(P2)


# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms.labels_[:100000])
plt.show()


# Well that's unfortunate. It seems K-means doesn't find those two clusters. This is a setback because it means I'll need to experiment with more clustering methods but lets see if we can't get K-means to converge on at least part of the upper and lower clusters.

# In[ ]:


kms3 = KMeans(n_clusters=3).fit(P2)
kms4 = KMeans(n_clusters=4).fit(P2)
kms5 = KMeans(n_clusters=5).fit(P2)


# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms3.labels_[:100000])
plt.show()


# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms4.labels_[:100000])
plt.show()


# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms5.labels_[:100000])
plt.show()


# Interesting! So it's stubornly not converging on the top and bottom "tails", and keeps creating clusters closer to the heads. Lets see if 6, 7 or 8 clusters will separate out those tails. If not, we may need to explore another clustering method.

# In[ ]:


kms6 = KMeans(n_clusters=6).fit(P2)
kms7 = KMeans(n_clusters=7).fit(P2)
kms8 = KMeans(n_clusters=8).fit(P2)


# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms6.labels_[:100000])
plt.show()


# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms7.labels_[:100000])
plt.show()


# In[ ]:


plt.scatter(P2[:100000, 0], P2[:100000, 1], c=kms8.labels_[:100000])
plt.show()


# So it doesn't look like K-means is going to give us exactly what we want. None-the-matter, the 5-cluster K-means produced some interesting clusters, so we can use those to generate some features. We'll explore a few more clustering algorithms later on. For now, lets look at some feature engineering using K-means clusters!

# #### 2.1.1 Feature Generation from K-Means
# 
# We're going to generate features in a few different ways: we're going to calculate distances to cluster centroids and predicted clusters. We're going to create a function that takes in a data frame and returns the processed one.

# In[ ]:


def cluster_features(df, model, pca):
    P = pca.transform(df)
    new_df = pd.DataFrame()
    new_df['cluster'] = model.predict(P)
    one_hot = pd.get_dummies(new_df['cluster'], prefix='cluster')
    new_df = new_df.join(one_hot)
    new_df = new_df.drop('cluster', axis=1)
    new_df = new_df.fillna(0)
    return new_df
    
def centroid_features(df, model, pca):
    P = pd.DataFrame(pca.transform(df))
    new_df = pd.DataFrame()
    cluster = 0
    for centers in model.cluster_centers_:
        new_df['distance_{}'.format(cluster)] = np.linalg.norm(P[[0, 1]].sub(np.array(centers)), axis=1)
        cluster += 1
    return new_df


# Since our features used for the clustering were normalized, we need to normalize them prior to feeding them into our experimenter. This method normalizes them and the following two functions are experiments our processor can take.

# In[ ]:


def norm_features(df):
    df['playersJoined'] = df.groupby('matchId')['matchId'].transform('count')
    df['killsNorm'] = df['kills']*((100-df['playersJoined'])/100 + 1)
    df['damageDealtNorm'] = df['damageDealt']*((100-df['playersJoined'])/100 + 1)
    df['maxPlaceNorm'] = df['maxPlace']*((100-df['playersJoined'])/100 + 1)
    df['matchDurationNorm'] = df['matchDuration']*((100-df['playersJoined'])/100 + 1)
    df = reduce_mem_usage(df)
    return df

def one_hot_encode(df):
    return pd.get_dummies(df, columns=['matchType'])

def remove_categories(df):
    target = 'winPlacePerc'
    drop_cols = ['Id', 'groupId', 'matchId', 'matchType', target]
    select = [x for x in df.columns if x not in drop_cols]
    return df.loc[:, select]


# In[ ]:


def kmeans_5_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms5, pca2))
    
def kmeans_5_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms5, pca2))

def kmeans_3_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms3, pca2))
    
def kmeans_3_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms3, pca2))

def kmeans_4_clusters(df):
    return df.join(cluster_features(remove_categories(one_hot_encode(norm_features(df))), kms4, pca2))
    
def kmeans_4_centroids(df):
    return df.join(centroid_features(remove_categories(one_hot_encode(norm_features(df))), kms4, pca2))


# ## 3. Baseline Model
# 
# In order to determine how useful the clusters are for building a model, we need to first try to fit a model to the data as is, so we can get a baseline to compare with later.
# 
# I'm going to base some of my code off of suggestions from [rejasupotaro's kernel](https://www.kaggle.com/rejasupotaro/effective-feature-engineering) including to split by match id and to use a simple linear regressor to test some experiments.

# In[ ]:


def train_test_split(df, test_size=0.1):
    match_ids = df['matchId'].unique().tolist()
    train_size = int(len(match_ids) * (1 - test_size))
    train_match_ids = random.sample(match_ids, train_size)

    train = df[df['matchId'].isin(train_match_ids)]
    test = df[-df['matchId'].isin(train_match_ids)]
    
    return train, test


# ### 3.1 Linear Experiment
# 
# I'm going to lift some code from the Kernel I mentioned earlier to run and prepare experiments from preprocess methods.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

def run_experiment(preprocess):
    df = reload()    

    df = preprocess(df)
    df.fillna(0, inplace=True)
    
    target = 'winPlacePerc'
    cols_to_drop = ['Id', 'groupId', 'matchId', 'matchType', target]
    cols_to_fit = [col for col in df.columns if col not in cols_to_drop]
    train, val = train_test_split(df, 0.1)
    
    model = LinearRegression()
    model.fit(train[cols_to_fit], train[target])
    
    y_true = val[target]
    y_pred = model.predict(val[cols_to_fit])
    return mean_absolute_error(y_true, y_pred)

def run_experiments(preprocesses):
    results = []
    for preprocess in preprocesses:
        start = time.time()
        score = run_experiment(preprocess)
        execution_time = time.time() - start
        results.append({
            'name': preprocess.__name__,
            'score': score,
            'execution time': f'{round(execution_time, 2)}s'
        })
        gc.collect()
        
    return pd.DataFrame(results, columns=['name', 'score', 'execution time']).sort_values(by='score')


# ### 3.2 Features to Compare With
# 
# The feature generation we did with K-means won't mean much until we compare it with other possible features. I'm going to use some features from the earlier Kernel I referenced: "[Effective Feature Engineering](https://www.kaggle.com/rejasupotaro/effective-feature-engineering)" by rejasupotaro.

# In[ ]:


def original(df):
    return df

def items(df):
    df['items'] = df['heals'] + df['boosts']
    return df

def players_in_team(df):
    agg = df.groupby(['groupId']).size().to_frame('players_in_team')
    return df.merge(agg, how='left', on=['groupId'])

def total_distance(df):
    df['total_distance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']
    return df

def headshotKills_over_kills(df):
    df['headshotKills_over_kills'] = df['headshotKills'] / df['kills']
    df['headshotKills_over_kills'].fillna(0, inplace=True)
    return df

def killPlace_over_maxPlace(df):
    df['killPlace_over_maxPlace'] = df['killPlace'] / df['maxPlace']
    df['killPlace_over_maxPlace'].fillna(0, inplace=True)
    df['killPlace_over_maxPlace'].replace(np.inf, 0, inplace=True)
    return df

def walkDistance_over_heals(df):
    df['walkDistance_over_heals'] = df['walkDistance'] / df['heals']
    df['walkDistance_over_heals'].fillna(0, inplace=True)
    df['walkDistance_over_heals'].replace(np.inf, 0, inplace=True)
    return df

def walkDistance_over_kills(df):
    df['walkDistance_over_kills'] = df['walkDistance'] / df['kills']
    df['walkDistance_over_kills'].fillna(0, inplace=True)
    df['walkDistance_over_kills'].replace(np.inf, 0, inplace=True)
    return df

def teamwork(df):
    df['teamwork'] = df['assists'] + df['revives']
    return df

def min_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId','groupId'])[features].min()
    return df.merge(agg, suffixes=['', '_min'], how='left', on=['matchId', 'groupId'])

def max_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].max()
    return df.merge(agg, suffixes=['', '_max'], how='left', on=['matchId', 'groupId'])

def sum_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].sum()
    return df.merge(agg, suffixes=['', '_sum'], how='left', on=['matchId', 'groupId'])

def median_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].median()
    return df.merge(agg, suffixes=['', '_median'], how='left', on=['matchId', 'groupId'])

def mean_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    return df.merge(agg, suffixes=['', '_mean'], how='left', on=['matchId', 'groupId'])

def rank_by_team(df):
    cols_to_drop = ['Id', 'groupId', 'matchId', 'winPlacePerc']
    features = [col for col in df.columns if col not in cols_to_drop]
    agg = df.groupby(['matchId', 'groupId'])[features].mean()
    agg = agg.groupby('matchId')[features].rank(pct=True)
    return df.merge(agg, suffixes=['', '_mean_rank'], how='left', on=['matchId', 'groupId'])


# ### 3.3 Run the Experiments
# 
# Time to run the experiments alongside the other features. Remember with mean absolute error, a lower error is better! We're using a validation split on the match id, so anything that's lower than the original on the validation set is promising. Only trying it on a tree based model will really give us a good idea though, because linear regression (what we're using for the simple model) is highly sensitive to how we treat categorical data like the `clusters` features. 

# In[ ]:


run_experiments([
    original,
    items,
    players_in_team,
    total_distance,
    headshotKills_over_kills,
    killPlace_over_maxPlace,
    walkDistance_over_heals,
    walkDistance_over_kills,
    teamwork
])


# In[ ]:


run_experiments([
    original,
    kmeans_3_clusters, 
    kmeans_3_centroids,
    kmeans_4_clusters, 
    kmeans_4_centroids,
    kmeans_5_clusters, 
    kmeans_5_centroids
])


# In[ ]:


run_experiments([
    original,
    min_by_team,
    max_by_team,
    sum_by_team,
    median_by_team,
    mean_by_team,
    rank_by_team,
])

