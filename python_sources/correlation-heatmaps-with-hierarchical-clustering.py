#!/usr/bin/env python
# coding: utf-8

# The purpose of this notebook is to present the benefits of hierarchical clustering for data analysis and for improving the representation of correlation heatmaps. For this notebook, we use the *breast-cancer-wisconsin* dataset. 
# 
# # Correlation coefficient

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data 
data = pd.read_csv("../input/breast-cancer-wisconsin-data/data.csv")

# Show dataframe
data


# We will get rid of some of the columns, including `id`, `diagnosis` and `Unnamed: 32`. `id` won't apport any important information to the model, `diagnosis` is the target and `Unnamed: 32` just contains *NaN* values.

# In[ ]:


data = data.drop(["id"], axis=1)
data = data.drop(["diagnosis"], axis=1)
data = data.drop(["Unnamed: 32"], axis=1)


# To measure the interplay between the different features, we will calculate the correlation coefficient between pairs. The correlation between two features gives us a degree of their relation. This relation can be linear or nonlinear. For this notebook, we will use linear correlation calculating the [Pearson Correlation Coefficient](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) ($\rho$). 
# 
# Correlation ranges from -1 to 1, where -1 indicates anticorrelation (or negative correlation), 0 no correlation and 1 correlation (or positive correlation). The following plot shows a pair of signals with their respective correlation.

# In[ ]:


import numpy as np

plt.figure(figsize=(14,3))

line1 = np.array([2, 1, 1, 2])
line2 = np.array([1, 2, 2, 1])

plt.subplot(131)
plt.plot(line1, color='royalblue')
plt.plot(line2, color='lightcoral')
plt.title(np.corrcoef(line1, line2)[1,0], fontsize=18)

line1 = np.array([2, 1, 1, 2])
line2 = np.array([1, 2, 2, 3])

plt.subplot(132)
plt.plot(line1, color='royalblue')
plt.plot(line2, color='lightcoral')
plt.title(round(np.corrcoef(line1, line2)[1,0],2), fontsize=18)

line1 = np.array([2, 1, 1, 2])
line2 = 2*line1

plt.subplot(133)
plt.plot(line1, color='royalblue')
plt.plot(line2, color='lightcoral')
plt.title(np.corrcoef(line2, 2*line2)[1,0], fontsize=18);


# The three plots show a negative correlation, no correlation, and positive correlation, respectively. 
# 
# Linear correlation is helpful in gaining quick intuition about the relation between two signals. However, it has some drawbacks. For instance, it won't account for sequential displacements (we would need to use lags) or non-linearities. Also, keep in mind that [correlation doesn't necessary mean causation](https://www.tylervigen.com/spurious-correlations). 
# 
# In the next figure, we plot a heatmap with all the correlations between features:

# In[ ]:


plt.figure(figsize=(15,10))
correlations = data.corr()
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);


# At first sight, we see many positive correlations (blue). However, this heatmap is messy. For visualization purposes, it would be better to group features that are highly correlated together. To do so, we will do [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering). 
# 
# # Hierarchical Clustering
# 
# Hierarchical clustering is a method to find hierarchy within our data. This hierarchy allows ordering the data in clusters. It arranges the data using a dissimilarity matrix (also called distance matrix), which gives information on how far are two features. The distance can be computed in many different ways. Since we're using the Pearson Correlation Coefficient, the distance matrix will be calculated as follows:
# 
# $$
# d(X, Y) = 1 - \big | \ \rho_{X, Y} \ \big |
# $$ 
# 
# For negative and positive correlations, the distance will be close to zero. If there is no correlation whatsoever, the distance will be $\approx 0$.
# 
# After computing the distance matrix, we have to group features hierarchically according to their distances. Then we can visualize the relationship between features in a tree diagram called dendrogram.. 

# In[ ]:


from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

plt.figure(figsize=(12,5))
dissimilarity = 1 - abs(correlations)
Z = linkage(squareform(dissimilarity), 'complete')

dendrogram(Z, labels=data.columns, orientation='top', 
           leaf_rotation=90);


# Initially, before starting the algorithm, each feature is a cluster. The algorithms take close features and combine them into a brand new cluster. Iteratively, the algorithm keeps grouping clusters until there is only one. Each leaf in the dendrogram represents a feature and each node a cluster. The *y-axis* shows the distance between points (ranging from 0 to 1). The number of clusters in our data will depend on which distance we take as a threshold. If we select a small distance, more clusters will be formed. Conversely, if we choose a large distance as a threshold, we would less clusters. 
# 
# 

# In[ ]:


# Clusterize the data
threshold = 0.8
labels = fcluster(Z, threshold, criterion='distance')

# Show the cluster
labels


# `label` shows which cluster each of the features belongs to. Finally, to observe the clusters in the correlation plot, we have to rearrange the features in the dataframe according to the cluster output.

# In[ ]:


import numpy as np

# Keep the indices to sort labels
labels_order = np.argsort(labels)

# Build a new dataframe with the sorted columns
for idx, i in enumerate(data.columns[labels_order]):
    if idx == 0:
        clustered = pd.DataFrame(data[i])
    else:
        df_to_append = pd.DataFrame(data[i])
        clustered = pd.concat([clustered, df_to_append], axis=1)


# Finally, we plot the clustered correlation plot:

# In[ ]:


plt.figure(figsize=(15,10))
correlations = clustered.corr()
sns.heatmap(round(correlations,2), cmap='RdBu', annot=True, 
            annot_kws={"size": 7}, vmin=-1, vmax=1);


# Since our threshold was set at 0.7, we will be able to see five different clusters (grouped in the main diagonal). The biggest one corresponds to the red tree in the dendrogram. Since these features are related to the geometry of cancer, they form a robust cluster. Another distinct cluster is formed just by `texture_mean` and `texture_worst`, in green in the dendrogram. Note that in order for this cluster to disappear, we'd have to decrease the threshold considerably. The remaining clusters are less recognizable, given that their distances are higher.
# 
# The following plot shows the different clusters using a different threshold.

# In[ ]:


plt.figure(figsize=(15,10))

for idx, t in enumerate(np.arange(0.2,1.1,0.1)):
    
    # Subplot idx + 1
    plt.subplot(3, 3, idx+1)
    
    # Calculate the cluster
    labels = fcluster(Z, t, criterion='distance')

    # Keep the indices to sort labels
    labels_order = np.argsort(labels)

    # Build a new dataframe with the sorted columns
    for idx, i in enumerate(data.columns[labels_order]):
        if idx == 0:
            clustered = pd.DataFrame(data[i])
        else:
            df_to_append = pd.DataFrame(data[i])
            clustered = pd.concat([clustered, df_to_append], axis=1)
            
    # Plot the correlation heatmap
    correlations = clustered.corr()
    sns.heatmap(round(correlations,2), cmap='RdBu', vmin=-1, vmax=1, 
                xticklabels=False, yticklabels=False)
    plt.title("Threshold = {}".format(round(t,2)))


# Seaborn also includes a function *clustermap* to plot the correlation heatmats with dendrograms.

# In[ ]:


sns.clustermap(correlations, method="complete", cmap='RdBu', annot=True, 
               annot_kws={"size": 7}, vmin=-1, vmax=1, figsize=(15,12));


# 
# # Conclusions
# 
# Hierarchical clustering can be helpful in understanding our data better. It also improves the visual representation of correlation heatmaps, making it easier to find groups of correlated features.

# # References
# 1. [Hierarchical Clustering with Python and Scikit Learn](https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/)
# 2. [Scipy Hierarchical Clustering](https://docs.scipy.org/doc/scipy/reference/cluster.hierarchy.html)
