#!/usr/bin/env python
# coding: utf-8

# ## Features clustering and visualization
# This notebook attempts to cluster the features and visualize them for better understanding so an effective feature selection/enginering can be designed.

# In[ ]:


# the required imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AffinityPropagation
from tqdm import tqdm
import gc


# ### Loading, Transforming and Scaling the data
# - The train and test set are loaded and concatenated into a single dataframe, it is important to get the total feature distribution not only from the train dataset.
# - the ID and target fields are dropped and log transformation is applied due the data's order of magnitudes.
# - StandardScale without the mean is applied, this is since for the visualization we'd like to better see the diference between the features.
# 

# In[ ]:


input_dir = '../input'

train_df = pd.read_csv(input_dir + '/train.csv')
train_df = train_df.drop(['ID', 'target'], axis = 1)
test_df = pd.read_csv(input_dir + '/test.csv')
test_df = test_df.drop(['ID'], axis = 1)
features = pd.concat([train_df, test_df], ignore_index=True)


data = np.log1p(features)

scaler = StandardScaler(with_mean=False)
data = scaler.fit_transform(data)
scaled_features = pd.DataFrame(data = data, columns=features.columns)

del train_df, test_df, data, features
gc.collect()


# ### Generating histograms for each one of the features
# In this notebook the features are clustered based on their histograms, the main idea is to group those features with the most similar statistical distributions, this could be done also based on their descriptives (mean, std, kurtosis, etc) but doing directly by their histogram might be more effective in terms of their visualization.
# 
# The 'bins' parameter establishes the bins of the histograms being generated for each feature, also indicates the granularity and how many the histograms will differ from each other.
# 
# A new dataframe with the histogram is generated, it's was implemented using a for loop, however this could be done with the apply method.

# In[ ]:


bins=300

features = scaled_features.columns
ranges = np.linspace(np.min(np.min(scaled_features,axis=0)), np.max(np.max(scaled_features,axis=0)), bins+1)

feat_hist_df =  pd.DataFrame(columns=ranges[:-1])

for feat in tqdm(features, ncols=110):
    hist = pd.DataFrame(np.histogram(scaled_features[feat], bins=ranges)[0]                        .reshape(1,-1), columns=ranges[:-1], index= [feat])
    feat_hist_df = feat_hist_df.append(hist)


# ### Clustering the histograms
# The zeros are removed and the features are clusterred using Affinity Propagation with default parameters, Affinity Propagation was used since it seems to be more effective with time series type data, although histograms are not time series, their shape is the property that is required to be kept.

# In[ ]:


feat_hist_nozero_df = feat_hist_df.drop([0],axis=1)
af = AffinityPropagation().fit(feat_hist_nozero_df)
feat_hist_nozero_df['cluster'] = af.labels_
print('Using Affinity Propagation resulted in total of : {} clusters'.format(len(af.cluster_centers_indices_)))


# ### Exploring the clusters by visualization
# From the resulting clusters the top N cluster(s) will be selected and visualized. This might lead to some insight.

# In[ ]:


N = 10
cluster_count = np.unique(af.labels_, return_counts=True)
# as a sorted DataFrame
cluster_count = pd.DataFrame({'cluster':cluster_count[0],'count':cluster_count[1]})                  .sort_values(by=['count'], ascending=False).reset_index(drop = True)
# obtaining the top and bottom clusters
top_clusters = cluster_count.head(N)['cluster'].tolist()
top_counts = cluster_count.head(N)['cluster'].sum()
bottom_clusters = cluster_count.tail(N)['cluster'].tolist()
top_accounts_percent = np.around(100 * top_counts / len(features),2)
print('Top {} cluster(s) accounts for {}% of total features'.format(N,top_accounts_percent))
print(cluster_count.head(N).T)
print('')
print('Bottom {} cluster(s)'.format(N))
print(cluster_count.tail(N).T)

# the data will be converted into "long" format so it could be visualize using sns.tsplot
feat_hist_nozero_long = feat_hist_nozero_df.reset_index().melt(id_vars=['index','cluster'],
                                                               var_name='bins', value_name='count')
intop = np.in1d(feat_hist_nozero_long.cluster, top_clusters)
inbottom = np.in1d(feat_hist_nozero_long.cluster, bottom_clusters)


# ### Visualizing the clusters
# When plotting the histogram for each of the {{N}} top cluster using seaborn's tsplot, it can be observed that the Affinity Propagation does a nice job grouping these features, the confidence interval (68%) shadow is very narrow, which most of the feaures within a cluster have very similar histograms.
# 

# In[ ]:


plt.figure(figsize=(16,8))
sns.tsplot(data=feat_hist_nozero_long[intop], time='bins',value='count',unit='index',condition ='cluster')
plt.title('Histograms for top N cluster(s)')
plt.grid()


# #### Some observations and thoughts
# - Between the top2 clusters, they group a total of 944 features, and they only have around 3 counts of values around their modes, indicating that those features have a lot of values that are zero. While the cluster 345 (with 99 features) have higher counts around its mode.
# - Histograms reveal that feature distributions are grouped around a mean/mode and follow a normal distribution-like shape. If normality can be confirmed. Then the original features would follow a log-normal distribution.
# - Most of the clusters histograms indicate *negative skewness* for the feature distributions. Can this be a consecuence of the log transformation?
# - If these features can be proved useful, using their distribution, can they be **sampled** so data augmentation can be implemented for the training set?

# In[ ]:


plt.figure(figsize=(16,8))
sns.tsplot(data=feat_hist_nozero_long[inbottom], time='bins',value='count',unit='index',condition ='cluster')
plt.title('Histograms for bottom N cluster(s)')
plt.grid()


# - The bottom clusters include only one feature and they group much higher counts of values around their mean/mode (higher than 100 counts). This indicates that there is strong relationship between the cluster and the number of zeros per feature.
# - The intuition might be lead us to think that the cluster with less features might be more relevant since they carry more information for the algorithms to make the predictions.
# 
# #### The relationship between the features per cluster and numer of zeros per feature
# 
# Following up on the intuition mentioned above, it is possible to select features based on their uniqueness of distribution and their count of zeros, following the visualization.

# In[ ]:


zeros_count = feat_hist_df[[0]].astype(int)
zeros_count.columns = ['zero_count']
zeros_count['cluster'] = af.labels_
zeros_count = zeros_count.reset_index(drop=True).groupby('cluster')                         .agg([np.size, np.mean])['zero_count'].sort_values('mean')
    
x = zeros_count['mean'].values
y1 = zeros_count['size'].values
y2 = np.cumsum(y1)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,figsize = (16,8))
plt.setp((ax1, ax2),xticks=np.arange(0,55000,500))
fig.tight_layout()
ax1.scatter(x, y1, alpha = 0.2)
ax1.grid()
ax1.set_title('Features per Cluster (y) vs. number of zeros per feature (x)')
ax1.set_ylabel('features per cluster')
ax2.plot(x,y2, c = 'g')
ax2.set_xlabel('Mean number of zeros per feature')
ax2.set_ylabel('features')
ax2.set_title('Cumulative Features per Cluster (y) vs. number of zeros per feature (x)')
ax2.grid()


# #### Additional Observations and thoughts
# - For example, if it is defined a "cut value" of 5200 zeros per feature, we will end of with 1000 features (~80% reduction of features), if we decide to include most of the cluster with few features, the number of features will increase exponentially, experiments with a model could be done in order to determine the added value of the rest of the features to the model's predictions.
# - Reducing the number of features might not improve the prediction score but might help the model to better generalize when predicting unseen data (test set).
