#!/usr/bin/env python
# coding: utf-8

# ### This notebook addresses this goal: Create summary tables that address relevant factors related to COVID-19
# Task Details
# 
# Create summary tables that address relevant factors related to COVID-19
# 
# Specifically, we want to know what the literature reports about:
# 
# * How does temperature and humidity affect the transmission of 2019-nCoV?
# 
# 
# 

# ### Table of contents
# 1. Load in necessary libraries
# 2. Load in the metadata table for all available papers
# 3. Load in the CORD-19 embeddings for the papers
# 4. Load in the example output table "How does temperature and humidity affect the transmission of 2019-nCoV.csv" from target_tables
# 5. Identifying example paper IDs
# 6. Loading in the CORD-19 embeddings
# 7. Exploring the full text embedding features
# 8. Run t-tests to see if mean of our examples is significantly different from the population mean for each embedding feature
# 9. Run unsupervised clustering using the informative features
# 10. Identify the cluster of papers we are interested in
# 
# ### Next steps:
# - identify paper type, (review, meta-analysis, etc.)
# - identify the factor (temperature? humidity?) the study is investigating
# - identify supporting excerpts supporting or rejecting the role of the factor in COVID transmission 
# - generate table with the required information.

# ### 1. Load in necessary libraries

# In[ ]:


import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.spatial.distance import cdist
import collections
import pickle
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


# ### 2. Load in the metadata table for all available papers

# In[ ]:


metadata_df = pd.read_csv('../input/CORD-19-research-challenge/metadata.csv', index_col='cord_uid')


# In[ ]:


metadata_df


# ### 3. Load in the example output table "How does temperature and humidity affect the transmission of 2019-nCoV.csv" from target_tables

# In[ ]:


example_df = pd.read_csv('../input/CORD-19-research-challenge/Kaggle/target_tables/2_relevant_factors/How does temperature and humidity affect the transmission of 2019-nCoV.csv')


# In[ ]:


example_df


# - Here we have 63 examples of relevant papers on the topic of temperature and humidity on COVID 19 transmission
# - This is probably not enough examples for applying ML yet
# - But let's take a look at the embedding features for these papers to see if there is something that we can use to differentiate them from the other papers
# - We need the cord_uid in order to find the embeddings for the papers, the example output table is missing the cord_uid for the papers
# - No problem, we can use the titles to find the corresponding cord_uid values and also the hashes for the full texts from the metadata table

# ### 5. Identifying example paper IDs

# In[ ]:


# find the example sha and uids for the example papers
example_shas = []
example_uids = []
for index, row in example_df.iterrows():
    study_title = row['Study']
    study_metadata = metadata_df[metadata_df['title'] == study_title]
    if len(study_metadata) != 0:
        sha = study_metadata.iloc[0]['sha']
        uid = study_metadata.iloc[0].name
        if str(sha) != 'nan':
            example_shas.append(sha)
            example_uids.append(uid)


# Some papers are included multiple times in our list of examples

# In[ ]:


example_uids


# Let's get all the unique papers from this list
# 

# In[ ]:


unique_example_uids = set(example_uids)
len(unique_example_uids)


# ### 6. Loading in the CORD-19 embeddings

# In[ ]:


embeddings_df = pd.read_csv('../input/CORD-19-research-challenge/cord_19_embeddings_4_24/cord_19_embeddings_4_24.csv', header=None, index_col=0)


# In[ ]:


available_uids = unique_example_uids.intersection(embeddings_df.index) # select example uids with an available embedding
example_embeddings_df = embeddings_df.loc[available_uids]


# In[ ]:


example_embeddings_df


# ### 7. Exploring the full text embedding features

# In[ ]:


# first lets see some plots of the embeddings features for the examples vs the rest of the papers
for i in range(1, 21, 2):
    plt.scatter(embeddings_df[i], embeddings_df[i+1])
    plt.scatter(example_embeddings_df[i], example_embeddings_df[i+1])
    plt.show()


# - Here we notice some interesting observations
# - A brief look at the embeddings for our examples reveal that they seem to have a bias and cluster close together in some embedding dimensions
# - We can run some simple t-tests to see which embedding dimensions have the most significant difference for our examples and are most relevant to use in identifying papers on the topic of how temperature and humidity affect the transmission of 2019-nCoV.

# ### 8. Run t-tests to see if mean of our examples is significantly different from the population mean for each embedding feature

# In[ ]:


# First, lets get the population mean for each embedding feature
feature_pop_means = embeddings_df.mean(0)


# In[ ]:


# Now run the t-tests
t_stats, p_vals = stats.ttest_1samp(example_embeddings_df, feature_pop_means)


# In[ ]:


# here we show some visualizations of feature significance
plt.bar(range(len(p_vals)), -np.log(p_vals)) # we plot the negative log of the p-values for ease of visualization purposes
plt.hlines(-np.log(0.05), 0, 800) # line representing p-value of 0.05
plt.hlines(-np.log(0.05/len(p_vals)), 0, 800) # line representing a bonferroni adjusted p-value cutoff


# - There seem to be many embedding features that could be informative of the specific type of paper we are looking for
# - Let's use a p-value threshold of 0.05/len(p_vals) to define a feature as informative 
# - We can try a clustering approach using these informative embedding features to identify all the potentially relevant paper

# ### 9. Run unsupervised clustering using the informative features

# In[ ]:


# select the subset of informative features from original dataframe (selecting p-values < 0.05/len(p_vals))
informative_embeddings_df = embeddings_df.loc[:, p_vals < 0.05/len(p_vals)]


# In[ ]:


informative_embeddings_df


# We can set the number of clusters a bit higher to find papers that are highly similar to our examples. Let's use 10 to see what we find

# In[ ]:


clustering = KMeans(n_clusters=10, random_state=0).fit(informative_embeddings_df.values)


# In[ ]:


# get the cluster labels
labels = clustering.labels_


# In[ ]:


collections.Counter(labels)


# ### 10. Identify the cluster of papers we are interested in

# In[ ]:


uid_cluster_map = dict(zip(informative_embeddings_df.index, labels)) # map the uids to the cluster labels


# In[ ]:


example_clusters = [uid_cluster_map[uid] for uid in example_embeddings_df.index] # find the cluster assigned to each of the example papers


# In[ ]:


# looks like cluster 4 contains all the papers from our examples.
example_clusters


# we can take a further look at the titles of all the papers from cluster 4 vs. other clusters to see whether the example cluster captured the most similar papers

# In[ ]:


for i in range(1, 11):
    print('cluster', i)
    cluster_ids = set([k for k, v in uid_cluster_map.items() if v == i])
    cluster_ids = cluster_ids.intersection(metadata_df.index)
    for ele in metadata_df.loc[cluster_ids, 'title']:
        if isinstance(ele, str):
            # use this rule to filter out non-relevant papers and focus on the coronavirus related content
            if ('corona' in ele.lower() or 'cov' in ele.lower()) and ('humid' in ele.lower() or 'temperature' in ele.lower()):
                print('\t', ele)
    print('#'*25)


# - We can see from clustering that all of the example papers are grouped into cluster 4
# - Cluster 4 also contains papers that were not in the example papers, but are relevant papers nonetheless
# - While we do see other papers containing the keywords that we searched for picked up in other clusters, they are not specifically relevant to COVID-19/transmission etc.
# - Generally, the cluster 4 papers seem more relevant to our examples compared to those papers from other clusters
