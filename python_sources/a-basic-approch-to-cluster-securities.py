#!/usr/bin/env python
# coding: utf-8

# # A Basic Approach to Cluster Securities

# There is a large diversity of securities in the dataset. This probably makes building a single model for all id's very difficult. On the other hand, building a model for each of the securities individually is out of reach. Therefore it might be a good idea to seperate the securities into a couple of groups on which we can train a model jointy.
# In the following, I am going to present a simple approach to find such a grouping.

# **Table of Contents**
# 
# * Preparation
# * Visualisation
# * Finding Clusters
# * Exploring Cluster Characteristics
# * Conclusion

# **Description**
# 
# I do the clustering soley based on the first four moments of the asset returns (y) as I assume that these contain almost all of the relevant information that characterize different asset types. 
# 
# I neglect that certain securities are only available for a subperiod and may therefore have characteristics dominated by the temporary regime. In a further study the temporal component may be included in the clustering process, e.g. recluster every n timestep using the last m observations.

# ## Preparation

# In[ ]:


import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


with pd.HDFStore('../input/train.h5') as train:
    df = train.get('train')


# In[ ]:


moments = df[['id', 'y']].groupby('id').agg([np.mean, np.std, stats.kurtosis, stats.skew]).reset_index()


# In[ ]:


moments.head()


# In[ ]:


sec = moments['id']
dat = moments['y']


# In[ ]:


# Scale the data
from sklearn.preprocessing import StandardScaler
scal = StandardScaler()
dat = scal.fit_transform(dat)


# ## Visualisation

# In[ ]:


dat = pd.DataFrame(dat)
dat.columns = ['mean', 'std', 'kurtoris', 'skew']


# In[ ]:


sns.pairplot(dat, size=1.5);


# There are no well seperated clusters visible, but the plots nevertheless promise some kind diversity which we might be able to exploit.

# ## Finding Clusters

# In[ ]:


from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[ ]:


# determine suitable number of clusters with silhouette score
k_max = 20
k_range = range(2,k_max)
scores = []
for k in k_range:
    model = KMeans(n_clusters=k, n_init=20).fit(dat)
    scores.append(silhouette_score(dat, model.labels_))


# In[ ]:


plt.figure()
plt.plot(k_range, scores);
plt.title('Results KMeans')
plt.xlabel('n_clusters');
plt.ylabel('Silhouette Score');


# This plot suggests that we should choose a k between 2 and 6. I will continue with a k=3.

# In[ ]:


k = 3


# In[ ]:


model = KMeans(n_clusters=k).fit(dat)


# In[ ]:


labels = pd.DataFrame([sec,model.labels_]).transpose()
labels.columns = ['sec', 'label']


# In[ ]:


labels.head()


# In[ ]:


# Create list of list such that members[i] contains id's with label i
members = []
for i in range(k):
    members.append(list(labels.sec[labels.label==i]))


# In[ ]:


print("Size of each group: ", [len(_) for _ in members])


# The small group is probably made up by the outliers we saw in the previous plot.

# ## Exploring Cluster Characteristics

# If the different clusters would indeed correspond to different asset classes, I would suspect that the available information for the different id's would reflect that, as bonds have different fundamentals as stocks or certain small caps might not report on all fundamentals that large caps do.

# In[ ]:


prop_nan = pd.DataFrame(index=range(k), columns=df.columns[2:-1])
for i in range(k): #go through labels
    dfi = df.loc[df['id'].isin(members[i]),:]
    n = len(dfi)
    for col in dfi.columns[2:-1]: #go through feature cols
        prop_nan.set_value(i, col, dfi[col].isnull().sum() / n)
for col in df.columns[2:-1]: #go through feature cols
    prop_nan.set_value('mean', col, df[col].isnull().sum() / len(df))
    


# In[ ]:


# only look at the features with significant missing data (>10%)
prop_nan_sel = prop_nan.loc[:,prop_nan.loc['mean',:] > 0.1].reset_index()


# In[ ]:


prop_nan_sel


# In[ ]:


n = len(prop_nan_sel.columns[1:])
cols = 3
f, axarr = plt.subplots(int(n/cols)+1, cols, figsize=(8,30))
i = 0
for col in prop_nan_sel.columns[1:]:
    sns.barplot(x='index', y=col, data=prop_nan_sel,ax=axarr[int(i/cols),i%cols]);
    axarr[int(i/cols),i%cols].set_ylabel('')
    axarr[int(i/cols),i%cols].set_xlabel('')
    i += 1


# Apparently our 'outlier class' has a significant amount of missing data. A reasonable explanation would be that these securities are penny stocks or the like which are highly volatile and don't have to publish the same data as blue chips do. 
# We can also observe that there are features for which either one of the two main classes, 0 or 1, have significantly more missing data than the other one. Maybe just confirmation bias (?) but is probably still worth digging deeper...

# ## Conclusion

# The previous elaboration showed a simple approach to using clustering techniques on the Two Sigma dataset. Despite its rudimentariness we could already observe some visible patterns which give enough motivation to do some more work down this path. I think that clustering could be a very powerful weapon to crack this dataset.
# 
# Thanks for reading! Any feedback is very much appreciated.
# 
# 
# 
# **Potential further work:**
# 
# * find better features to build clusters
#  * maybe based on missing data
# * experiment with different number of clusters
# * try different models, e.g. Gaussian Mixture
# * do time dependent clustering to avoid influence of regime shifts
