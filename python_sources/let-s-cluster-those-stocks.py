#!/usr/bin/env python
# coding: utf-8

# ***A brief tutorial on clustering the stocks***

# I keep seeing people truncate the Market_train_df to the last couple of years to manage the memory. Not sure if that's their actual solution or they are keeping the big hammer in their private kernels. Anyways, I just wanted to contribute to the community by showing how to properly manage the training data by clustering the similar stocks together through the following steps.
# * **Data Preparation**: We'll start off with the usual suspects.

# In[ ]:


import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[ ]:


from kaggle.competitions import twosigmanews
# You can only call make_env() once, so don't lose it!
env = twosigmanews.make_env()


# In[ ]:


market_df, _ = env.get_training_data()
market_df['time'] = market_df.time.dt.date
market_df.shape


# So we are starting off with 4 million rows, which is way too much for the given memory! But there are lots tickers that either don't have enough entries to train or their "universe" label is mostly off, so they wouldn't have any significant effect on the final score anyways! Besides, we are interested only in the stocks that are still alive by the end of 2016.

# In[ ]:


df = market_df.pivot(index='time', columns='assetCode', values='universe')
toRemove = df.count()[df.count().values < 100].index.values
toRemove = np.append(toRemove, df.mean()[df.mean().values < .05].index.values)
df = market_df.groupby('assetCode').agg({'time': 'max'}).reset_index()
toRemove = np.append(toRemove, df[df.time < datetime.date(2016, 12, 1)].assetCode.values)
toRemove = np.unique(toRemove)
market_df = market_df[~market_df.assetCode.isin(toRemove)]
market_df.shape


# As the scientists say, "without the loss of generality" we have reduced the size of the dataframe by 35%. But that's not enough! 
# * **Correlation:**
# There are many stocks that have very similar trends. So we can use hierarchical clustering (or whatever you are comfortable with) to combine these stocks together, based on the correlation of their 5 day moving average for the last year:

# In[ ]:


market_df = market_df.sort_values(['assetCode', 'time'])
market_df['sma5'] = market_df['close'].rolling(5, min_periods=1).mean()
df = market_df.pivot(index='time', columns='assetCode', values='sma5').reset_index()
df = df[df.time > datetime.date(2016,1,1)]
df = df.set_index('time')
df = df.dropna(axis=1)
corr = df.corr()
df.index.name = 'time'
df


# Let's take a look at the correlation matrix:

# In[ ]:


size = 7
fig, ax = plt.subplots(figsize=(size, size))
ax.matshow(corr,cmap=cm.get_cmap('coolwarm'), vmin=0,vmax=1)


# * **Clustering:**
# There are lots of red spots that represent high correlation. Now we can use this correlation function as a distance metrics to cluster the stocks.

# In[ ]:


from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(corr, 'average')
from scipy.cluster.hierarchy import cophenet
from scipy.spatial.distance import pdist
import pylab
c, coph_dists = cophenet(Z, pdist(corr))
c


# I have used "average" as my metric, seen it's showing one of the highest cophentic correlation distance. But feel free to explore your options!
# Now let's take a look at the last 100 iterations:

# In[ ]:


fig = plt.figure(figsize=(25, 10))
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=100,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show()


# The values of the x-axis show the number of stocks clustered together. Now it's time to pick a cut of value and be done with our clustering. Using the maximum distance, I'm clustering the stocks with distance of 10 or lower, which is too extreme but it's a good start for our training purposes.

# In[ ]:


from scipy.cluster.hierarchy import fcluster
max_d = 10
clusters = fcluster(Z, max_d, criterion='distance')
clusters.max()


# So, now we have only 18 different indices! Compare it to the original 3000+ stocks!
# Finally, let's see how the stocks in the same cluster look!

# In[ ]:


df = df.transpose()
df['cluster'] = clusters
def norm(x):
    y = x.transpose()
    for col in y.columns:
        y[col] = (y[col] - y[col][:-1].min()) / (y[col][:-1].max() - y[col][:-1].min())
    return y.transpose()
normed_df = norm(df)

data = normed_df[df['cluster'] == 1]
data = data.transpose()
f, (ax1) = plt.subplots(1, 1, figsize=(20,8), sharey=True)
plt.subplots_adjust(wspace=0.05)

t = data.drop('cluster')
tickers = data.columns.values
for tix in tickers:
    ax1.plot(t[tix],label=tix)
ax1.set_title('Stock Correlation')
ax1.set_ylabel('Close prices')
ax1.legend(loc='upper left',prop={'size':8})
plt.setp(ax1.get_xticklabels(), rotation=70);


# Now, let's take a look at the correlation matrix after sorting the data based on the clusters

# In[ ]:


dff = df.sort_values('cluster').transpose()
# dff = df.groupby('cluster').mean().transpose()
new_corr = dff.corr()
size = 7
fig, ax = plt.subplots(figsize=(size, size))
ax.matshow(new_corr,cmap=cm.get_cmap('coolwarm'), vmin=0,vmax=1)


# In[ ]:




