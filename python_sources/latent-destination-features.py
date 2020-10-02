#!/usr/bin/env python
# coding: utf-8

# #Latent Destination Features
# ##Let's have a look at the latent search region features.
# ##It won't help you to boost your score immediately although you might gain a few ideas how to apply dimensionality reduction.
# 
# ##I just wanted to play with seaborn a bit.
# 

# In[ ]:





# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
sns.set(color_codes=True)


# Destinations.csv has 62K rows. Let's keep only the frequent search destinations. 
# Removing 50K records we could still keep 97% of the test bookings.

# In[ ]:


destination_features = pd.read_csv("../input/destinations.csv")
print(destination_features.shape)
test_destinations = pd.read_csv("../input/test.csv", usecols=['srch_destination_id'])
srch_destinations, count = np.unique(test_destinations, return_counts=True)
fig, ax = plt.subplots(ncols=2, sharex=True)
ax[0].semilogy(sorted(count))
ax[1].plot(1.0 * np.array(sorted(count)).cumsum()/count.sum())
ax[0].set_xticks(range(0, len(srch_destinations), 10000))
ax[1].set_ylabel('Cumulative sum')
ax[0].set_ylabel('Search destination counts in test set (log scale)')
frequent_destinations = srch_destinations[count >= 10]
print (1. * count[count >= 10].sum() / count.sum())


# We show the correlations among the 149 latent features.

# In[ ]:


frequent_destinations = srch_destinations[count >= 10]
frequent_destination_features = destination_features[destination_features['srch_destination_id'].isin(frequent_destinations)]
frequent_destination_features = frequent_destination_features.drop('srch_destination_id', axis=1)
print(frequent_destination_features.shape)
correlations = frequent_destination_features.corr()
f = plt.figure()
ax = sns.heatmap(correlations)
ax.set_xticks([])
ax.set_yticks([])
plt.title('Tartan or correlation matrix')
f.savefig('tartan.png', dpi=300)
plt.show()


# It looks like a nice tartan! It is easy to see that we have many strong correlations and the column order seems to be randomized.
# 

# In[ ]:


fig=plt.figure()
sns.distplot(correlations.values.reshape(correlations.size), bins=50, color='g')
plt.title('Correlation values')
plt.show()
fig.savefig('CorrelationHist')


# Using hierarchical clustering we try to reorder the features.

# In[ ]:


g = sns.clustermap(correlations)
g.ax_heatmap.set_xticks([])
g.ax_heatmap.set_yticks([])
g.savefig('clustermap.png', dpi=300)


# Use dendogram_col.reordered_ind to get the index of the original columns.

# In[ ]:


print(g.dendrogram_col.reordered_ind)


# Select a few features from the beginning and check their distributions.

# In[ ]:


a = [8, 102, 120, 127, 74]
to_plot = frequent_destination_features[frequent_destination_features.columns[a]].sample(1000)
g = sns.PairGrid(to_plot, size=3)
g.map_upper(plt.scatter, s=5, alpha=0.3)
g.map_lower(sns.kdeplot, cmap="Blues_d")
g.map_diag(sns.kdeplot, legend=False, shade=True)
plt.suptitle('A few features')
g.savefig('cluster_1.png', dpi=300)  


# Select a few correlated features from the middle and check their distributions.

# In[ ]:


b = [89, 69, 115, 105, 71]
def green_kde_hack(x, color, **kwargs):
    sns.kdeplot(x, color='g', **kwargs)
to_plot = frequent_destination_features[frequent_destination_features.columns[b]].sample(1000)
g = sns.PairGrid(to_plot, size=3)
g.map_upper(plt.scatter, s=5, alpha=0.3, color='g')
g.map_lower(sns.kdeplot, cmap="Greens_d")
g.map_diag(green_kde_hack, legend=False, shade=True)
plt.suptitle('A few correlated features')
g.savefig('cluster_2.png', dpi=300)

