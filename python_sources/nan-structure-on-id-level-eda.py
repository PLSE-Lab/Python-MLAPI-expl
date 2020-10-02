#!/usr/bin/env python
# coding: utf-8

# ## Overview
# 
# This notebook focuses on the exploratory analysis of the NaNs in the data. There are three parts:
# 
#  1. *Is the NaN structure on id level random?*
#  2. *Can that NaN structure be used to cluster id's?*
#  3. *Are there any gaps in any id time series?*

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


data = pd.read_hdf("../input/train.h5")


# How many id's are in the dataset?

# In[ ]:


len(pd.unique(data.id))


# ##1. Is the NaN Structure on Id Level Random?

# **Short answer**: No it isn't. Let's dig in deeper!

# ## Time Structure for First Id

# In[ ]:


# First ID
data[data.id == data.id[0]].plot(x = 'timestamp', y = 'y',kind='scatter');


# The following plot shows the NaNs for the first id (it is basically a picture of the dataframe). Keep in mind that the data is already sorted.
# 
# Black: No NaN
# 
# White: NaN

# In[ ]:


import matplotlib.pyplot as plt

data_binary = data[data.id == data.id[0]].isnull()

fig = plt.figure()
ax = fig.add_subplot(111)
ax.imshow(data_binary, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
plt.title("NaN structure for first id. White: NaN Black: No NaN")
plt.xlabel("Features")
plt.ylabel("Timestamp");


# The data is **not** missing at random.
# 
# Now let's do the same analysis for more financial instruments (chosen randomly).

# ## Time Structure for Randomly Chosen Id's

# In[ ]:


np.random.seed(352)
ids = np.random.choice(data.id,20)

for i in ids:
    
    data_binary = data[data.id == i].isnull()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(data_binary, aspect='auto', cmap=plt.cm.gray, interpolation='nearest')
    plt.title("NaN structure for id %i" % i)
    plt.xlabel("Features")
    plt.ylabel("Timestamp");


# It seems that the measurements are only missing at the beginning of the time series! They seem to kick in later on (some never kick in).
# 
# Another thing that is interesting: Look at the features 70 - 111 (x-axis). The NaN structure is often similar for different financial instruments. But in general each instrument has a different NaN structure (at least those we looked at).
# 
# **Hypothesis**: *The "noisy" NaN structure is not observable when drilling down on id level. The data is either always missing or only for a specific time period. After that time period it can always be observed. The NaN structure is not necessarily 100% unique for each id. id's can share aspects of that structure.* 
# 
# (Still need to work out how to generalize the analysis above without looking at each id plot myself :P)

# ## 2. Can That NaN Structure Be Used to Cluster Id's?

# **Short answer**: There are at least two different groups of id's when looking at NaNs.

# Let's have an aggregated look at the NaNs for each feature for each id.

# In[ ]:


unique_ids = pd.unique(data.id)

NaN_vectors = np.zeros(shape=(1424, data.shape[1]))

for i, i_id in enumerate(unique_ids):
    
    data_sub = data[data.id == i_id]
    NaN_vectors[i, :] = np.sum(data_sub.isnull(),axis=0) / float(data_sub.shape[0])


# Now we have the relative number of NaNs for each label and id. The columns of this matrix correspond to the labels in the dataset and the rows correspond to the unique id's (i.e. shape = 1424x111).
# 
# This is what the NaN vector looks like for the first id. Every entry gives the ratio of NaNs vs all recorded timestamps for that id (i.e. 0 means no NaNs for that label and 1 means that all recorded measurements are NaN).

# In[ ]:


NaN_vectors[0, :]


# Next, we apply hierarchical clustering on this data (applied to rows not columns, i.e. we sort the id's and not the labels). The color scheme is now inverted! I.e. NaNs are dark and non-NaNs are bright.

# In[ ]:


import seaborn as sns
g = sns.clustermap(NaN_vectors,col_cluster=False,method='average',metric='euclidean')


# Just by looking at the NaN patterns, there might be two or three groups of id's that are similar w.r.t their NaN structure.

# ## 3. Are There Any Gaps in Any Id Time Series?

# **Short answer**: Yes there are!

# Let's see if we can find any gaps of measurements in the time series. Since the measurement frequency is the same across all id's (timestamp always +1) we can simply look for gaps by using the following formula:
# 
# ratio = total_count_measurements / (timestamp_max - timestamp_min + 1)
# 
# If there is no gap, we expect the ratio to be 1. If there is a gap, then it should be <1.

# In[ ]:


unique_id = pd.unique(data.id)

result = pd.DataFrame(np.zeros(shape=(len(unique_id),5)))

for i, i_id in enumerate(unique_id):

    data_sub = data[data.id == i_id]

    count = data_sub.timestamp.count()

    time_min = np.min(data_sub.timestamp)
    time_max = np.max(data_sub.timestamp)

    ratio = count / float(time_max - time_min + 1)
    
    result.loc[i, :] = [i_id, ratio, time_min, time_max, count]
    result.columns = ["id", "ratio", "time_min", "time_max", "count"]


# In[ ]:


print(pd.unique(np.round(result.ratio,2)))


# Interesting! There are indeed values <1! Let's have a closer look.

# In[ ]:


print(result[result.ratio < 0.99])


# In[ ]:


ids = result[result.ratio < 0.99].id
ratios = result[result.ratio < 0.99].ratio

for i, r in zip(ids,ratios):

    data[data.id == i].plot(x = "timestamp", y = "y", kind = "scatter", 
                            title = "ratio:" + str("{0:.2f}".format(r)) + " id: " + str(int(i)));


# We have found id's where gaps in the data exist. Not clear if these gaps are meaningful or actual measurement errors.

# Hope this is useful! :)

# In[ ]:




