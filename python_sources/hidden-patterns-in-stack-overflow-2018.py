#!/usr/bin/env python
# coding: utf-8

# # Pattern Analysis of Stack Overflow 2018
# In this notebook, I'll try to use different unsupervised learning approaches to find any hidden patterns in the Stack Overflow 2018 survey data. Hopefully they will be able to uncover something interesting which would otherwise go unnoticed. Let's see how far we can go :)
# 
# <img src='http://i66.tinypic.com/261yxjn.png' width='100%' />
# 
# ## Table of Contents:
# * [1. Preprocessing the data](#preprocess)
# * [2. Principal Component Analysis](#pca)
#     * [2.1. Scree Plot](#pca_scree)
#     * [2.1. PCA Biplots](#pca_biplot)
# * [3. Clustering](#clustering)
#     * [3.1. HDBSCAN and TSNE](#hdbscan_tsne)
#     * [3.2. KMeans and TSNE](#kmeans_tsne)
#     * [3.3. Playing with TSNE perplexity](#perplexity)
#     * [3.3. Cluster Correlations](#cluster_correlation)  

# In[2]:


from copy import deepcopy

import numpy as np
import pandas as pd

from tqdm import tqdm

import hdbscan
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import scikitplot as skplt

matplotlib.style.use('ggplot')
matplotlib.rcParams['figure.figsize'] = (20.0, 5.0)


# # 1. Preprocessing the data <a class="anchor" id="preprocess"></a>
# In this dataset we have some columns where people have been able to select multiple values; this has to be extracted and put on a format which can be interpreted by the unsupervised models. My data processing is essentially the same as I've used in [this notebook](https://www.kaggle.com/nanomathias/predicting-r-vs-python), the basic point being to create one-hot-encoding columns for all categorical data in the dataset.

# In[3]:


# Read in the survey results, shuffle results
print(f">> Loading data")
df = pd.read_csv('../input/survey_results_public.csv', low_memory=False).sample(frac=1)

# Columns with multiple choice options
MULTIPLE_CHOICE = [
    'CommunicationTools','EducationTypes','SelfTaughtTypes','HackathonReasons', 
    'DatabaseWorkedWith','DatabaseDesireNextYear','PlatformWorkedWith',
    'PlatformDesireNextYear','Methodology','VersionControl',
    'AdBlockerReasons','AdsActions','ErgonomicDevices','Gender',
    'SexualOrientation','RaceEthnicity', 'LanguageWorkedWith',
    'IDE', 'FrameworkWorkedWith', 'FrameworkDesireNextYear',
    'LanguageDesireNextYear', 'DevType',
]

# Columns which we are not interested in
DROP_COLUMNS = [
    'Salary', 'SalaryType', 'Respondent', 'CurrencySymbol'
]

# Drop too easy columns
print(f">> Deleting uninteresting or redundant columns: {DROP_COLUMNS}")
df.drop(DROP_COLUMNS, axis=1, inplace=True)

# Go through all object columns
for c in MULTIPLE_CHOICE:
    
    # Check if there are multiple entries in this column
    temp = df[c].str.split(';', expand=True)

    # Get all the possible values in this column
    new_columns = pd.unique(temp.values.ravel())
    for new_c in new_columns:
        if new_c and new_c is not np.nan:

            # Create new column for each unique column
            idx = df[c].str.contains(new_c, regex=False).fillna(False)
            df.loc[idx, f"{c}_{new_c}"] = 1

    # Info to the user
    print(f">> Multiple entries in {c}. Added {len(new_columns)} one-hot-encoding columns")

    # Drop the original column
    df.drop(c, axis=1, inplace=True)
        
# For all the remaining categorical columns, create dummy columns
df = pd.get_dummies(df)

# Fill in missing values
df.dropna(axis=1, how='all', inplace=True)
dummy_columns = [c for c in df.columns if len(df[c].unique()) == 2]
non_dummy = [c for c in df.columns if c not in dummy_columns]
df[dummy_columns] = df[dummy_columns].fillna(0)
df[non_dummy] = df[non_dummy].fillna(df[non_dummy].median())
print(f">> Filled NaNs in {len(dummy_columns)} OHE columns with 0")
print(f">> Filled NaNs in {len(non_dummy)} non-OHE columns with median values")

# Create correlation matrix
corr_matrix = df.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.75)]
print(f">> Dropping the following columns due to high correlations: {to_drop}")
df = df.drop(to_drop, axis=1)

# Perform scaling on all non-dummy columns. Create X and y
nondummy_columns = [c for c in df.columns if df[c].max() > 1]
X = deepcopy(df)
X.loc[:, nondummy_columns] = scale(df[nondummy_columns])
X.drop('ConvertedSalary', axis=1, inplace=True)
print(f">> Shape of final dataframe X: {X.shape}")


# # 2. Principal Component Analysis<a class="anchor" id="pca"></a>
# Let us start with one of the classics, namely, principal component analysis (PCA). Right now we have more than 800 columns (dimensions) in our dataset, and with PCA we essentially try to find a new orthogonal vector space which describes most of the variation in the data. See the following image for an illustration:
# 
# <img src='https://static1.squarespace.com/static/5a316dfecf81e0076f50dae2/t/5ac35d702b6a284b3fde6131/1522753187751/PCA.png?format=500w'  />
# <center>Image taken from [osgdigitallabs](https://www.osgdigitallabs.com/blogs/2018/4/3/dimensionality-reduction)</center>
# 
# 

# In[4]:


# Create a PCA object, specifying how many components we wish to keep
pca = PCA(n_components=50)

# Run PCA on scaled numeric dataframe, and retrieve the projected data
pca_trafo = pca.fit_transform(X)

# The transformed data is in a numpy matrix. This may be inconvenient if we want to further
# process the data, and have a more visual impression of what each column is etc. We therefore
# put transformed/projected data into new dataframe, where we specify column names and index
pca_df = pd.DataFrame(
    pca_trafo,
    index=X.index,
    columns=["PC" + str(i + 1) for i in range(pca_trafo.shape[1])]
)


# ## 2.1. Scree plot<a class="anchor" id="pca_scree"></a>
# Now we've reduced our 800-dimensional space into 50 "principal components", and the first thing we want to check is how much of the dataset is retained within these 50 components - sometimes we can get lucky that more than 30% of our data is even contained in the first 2 components, which means that we would be able to plot those two components in a biplot, and see 30% of the variation in the dataset just from a standard 2D plot. 

# In[5]:


# Plot the explained variance# Plot t 
plt.plot(
    pca.explained_variance_ratio_, "--o", linewidth=2,
    label="Explained variance ratio"
)

# Plot the cumulative explained variance
plt.plot(
    pca.explained_variance_ratio_.cumsum(), "--o", linewidth=2,
    label="Cumulative explained variance ratio"
)

# Show legend
plt.ylim([-0.1, 0.7])
plt.legend(loc="best", frameon=True)
plt.show()


# In this case we are not so lucky - with 50 components we still only explain about 55% of the dataset, and with only 2 components we include a little less than 20% of the total variation in the dataset. Still, 10% in only two variables is still significantly better than the about ~2% included in the original features. 
# 
# ## 2.2. PCA Biplots<a class="anchor" id="pca_biplot"></a>
# Even though it's a quite small part of the variation in the dataset, let us try to plot the PCA biplots for the first few components to see if we can see any pattern. Since it might be interesting, I'll also color different dev types in different colors, just to see if any pattern emerges.

# In[6]:


_, axes = plt.subplots(2, 2, figsize=(15, 10))

dev_types = [c for c in X.columns if 'DevType' in c]
colors = sns.color_palette('hls', len(dev_types)).as_hex()

for i, dev in enumerate(dev_types):   
    idx = (df[dev] == 1)
    pca_df.loc[idx].plot(kind="scatter", x="PC1", y="PC2", ax=axes[0][0], c=colors[i], alpha=0.1)
    pca_df.loc[idx].plot(kind="scatter", x="PC2", y="PC3", ax=axes[0][1], c=colors[i], alpha=0.1, label=dev)
    pca_df.loc[idx].plot(kind="scatter", x="PC3", y="PC4", ax=axes[1][0], c=colors[i], alpha=0.1)
    pca_df.loc[idx].plot(kind="scatter", x="PC4", y="PC5", ax=axes[1][1], c=colors[i], alpha=0.1)
    
axes[0][1].legend(loc='best', bbox_to_anchor=(1, 0.5))
plt.show()


# Seems like there are no very clear patterns in first few components of the PCA, and they do not seem to clearly separate the dataset when it comes to developer types at least.

# # 3. Clustering<a class="anchor" id="clustering"></a>
# Now we'll play with clustering of the dataset using different approaches
# 
# ## 3.1. HDBSCAN & T-SNE<a class="anchor" id="hdbscan_tsne"></a>
# First let's try HDBSCAN for clustering algorithm and for dimensionality reduction I'll use TSNE. The reasons for chosing HDBSCAN of other typical clustering algorithms (K-Means, etc.) are nicely summarized [here](http://hdbscan.readthedocs.io/en/latest/comparing_clustering_algorithms.html). Let's see if any patterns emerge by this approach.
# 
# Due to the size of the dataset, I've chosen to take a smaller sample for this visualization; otherwise it took too long to comfortably run during a cup of coffee.

# In[7]:


# Create a sample dataset
sample = X.sample(30000)


# In[8]:


get_ipython().run_cell_magic('time', '', 'print(">> Clustering using HDBSCAN")\nclusterer = hdbscan.HDBSCAN(min_cluster_size=500)\nclusterer.fit(sample)')


# In[9]:


get_ipython().run_cell_magic('time', '', 'print(">> Dimensionality reduction using TSNE")\nprojection = TSNE(init=\'pca\', random_state=42).fit_transform(sample)')


# In[11]:


def get_cluster_colors(clusterer, palette='Paired'):
    """Create cluster colors based on labels and probability assignments"""
    n_clusters = len(np.unique(clusterer.labels_))
    color_palette = sns.color_palette(palette, n_clusters)
    cluster_colors = [color_palette[x] if x >= 0 else (0.5, 0.5, 0.5) for x in clusterer.labels_]
    if hasattr(clusterer, 'probabilities_'):
        cluster_colors = [sns.desaturate(x, p) for x, p in zip(cluster_colors, clusterer.probabilities_)]
    return cluster_colors

# Create the plot on the TSNE projection with HDBSCAN colors
_, ax = plt.subplots(1, figsize=(20, 10))
ax.scatter(
    *projection.T, 
    s=50, linewidth=0, 
    c=get_cluster_colors(clusterer), 
    alpha=0.25
)
plt.show()


# Interestingly, TSNE creates a very clear pattern with lots of clusters, yet the HDBSCAN clusterer does not seem to clearly agree that all these groups of people are actually belonging to a specific cluster. It is important to remember that TSNE **does not** preserve distances or density from the original dataset, rather it tries to preserve nearest neighbors, and as a result can create "fake" patterns, which do not neccesarily have any easily intepretable meaning. Some of the identified clusters by HDBSCAN, however, do align with observed groups from the TSNE. Most of the clusters, however, fall into one big group in the TSNE, and are not that clearly separable in the 2D TSNE embedding. 
# 
# We'll inspect the clusters more closely in a later section.

# ## 3.2. KMeans & T-SNE<a class="anchor" id="kmeans_tsne"></a>
# For KMeans we need to determine how many clusters we need. We'll keep working with the subset of data, so that processing won't take too long.

# In[12]:


from sklearn.cluster import KMeans
kmeans = KMeans(random_state=42)
skplt.cluster.plot_elbow_curve(kmeans, sample, cluster_ranges=[1, 5, 10, 50, 100, 200])
plt.show()


# It seems like there's a kink around 6 clusters, so let's go with that for now. As before we'll plot the colors of the KMeans clusters onto the TSNE plot.

# In[13]:


# Create the plot on the TSNE projection with HDBSCAN colors
_, ax = plt.subplots(1, figsize=(20, 10))
kmeans = KMeans(n_clusters=6).fit(sample)
ax.scatter(
    *projection.T, 
    s=50, linewidth=0, 
    c=get_cluster_colors(kmeans, 'hls'), 
    alpha=0.25
)
plt.show()


# Beyond looking slighly psychedelic the KMeans clusters seem paint a pretty clear picture together with TSNE - seems like a lot of the clusers identifier by KMeans are co-located at the groups of points generated by TSNE. Now that we have cluster identifications with two clustering methods, HDBSCAN and KMeans, we should play a bit with the TSNE settings to see how they can influence the plots.

# ## 3.3. Playing with TSNE perplexity<a class="anchor" id="perplexity"></a>
# The perplexity setting of TSNE can quite often lead to significant variations in the output result, so it could be interesting to see how different values affect our results.

# In[12]:


perplexities = [5, 30, 50, 100]

_, axes = plt.subplots(2, 4, figsize=(20, 10))
for i, perplexity in tqdm(enumerate(perplexities)):
    
    # Create projection
    projection = TSNE(init='pca', perplexity=perplexity).fit_transform(sample)
    
    # Plot for HDBSCAN clusters
    axes[0, i].set_title("Perplexity=%d" % perplexity)
    axes[0, i].scatter(
        *projection.T, 
        s=50, linewidth=0, 
        c=get_cluster_colors(clusterer), 
        alpha=0.25
    )
    
    # Plot for KMeans clusters
    axes[1, i].scatter(
        *projection.T, 
        s=50, linewidth=0, 
        c=get_cluster_colors(kmeans, 'hls'), 
        alpha=0.25
    )

plt.show()


# This looks very interesting; on the first row we have the HDBSCAN clusters, and the second row are the KMeans clusters. We can clearly see how changing the perplexity changes how and how well the classes / clusters are separated in the TSNE embedding.
# 
# ## Cluster Correlations<a class="anchor" id="cluster_correlation"></a>
# Having played enough with the funny TSNE visualizations, let's try investigate what is actually inside the clusters identified by HDBSCAN. We'll do that by going through each cluster, and finding the features that are highly correlated within those clusters.

# In[18]:


# Get the data for each cluster (not noise, aka -1)
unique_clusters = [c for c in np.unique(clusterer.labels_) if c > -1]
        
# Create a figure for holding the correlation plots
cols = 2
rows = np.ceil(len(unique_clusters) / cols).astype(int)
_, axes = plt.subplots(rows, cols, figsize=(20, 10*rows))
if rows > 1:
    axes = [x for l in axes for x in l]

# Calculate sample means
sample_mean = sample.median()

# Go through clusters identified by HDBSCAN
for i, label in enumerate(unique_clusters):
    
    # Get index of this cluster
    idx = clusterer.labels_ == label
    
    # Identify feature where the median differs significantly
    median_diff = (sample.median() - sample[idx].median()).abs().sort_values(ascending=False)
    
    # Create boxplot of these features for all vs cluster
    top = median_diff.index[0:20]
    temp_concat = pd.concat([sample.loc[:, top], sample.loc[idx, top]], axis=0).reset_index(drop=True)
    temp_concat['Cluster'] = 'Cluster {}'.format(i+1)
    temp_concat.loc[0:len(sample),'Cluster'] = 'All respondees'
    temp_long = pd.melt(temp_concat, id_vars='Cluster')
    
    sns.boxplot(x='variable', y='value', hue='Cluster', data=temp_long, ax=axes[i])
    for tick in axes[i].get_xticklabels():
        tick.set_rotation(90)
    axes[i].set_title(f'Cluster #{i+1} - {idx.sum()} respondees')    

# Tight layout    
plt.tight_layout()
plt.show()


# In[ ]:




