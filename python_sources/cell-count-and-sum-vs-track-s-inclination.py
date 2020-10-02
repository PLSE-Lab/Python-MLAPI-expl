#!/usr/bin/env python
# coding: utf-8

# 
# ## Now, we expect that particles that hit a detector along the detector's normal axis will traverse fewer cells, but let's see if we can prove this
# ## Also, what can we learn about true vs spurious hits by looking at the cells that were hit

# ### load the libraries

# In[132]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import glob
import seaborn as sns
plt.style.use('seaborn-deep')
FILE_FORMAT = "../input/train_1/event000001000-{}.csv"


# ### read the first training event

# In[133]:


truth = pd.read_csv(FILE_FORMAT.format("truth"))
hits = pd.read_csv(FILE_FORMAT.format("hits"))
particles = pd.read_csv(FILE_FORMAT.format("particles"))
cells = pd.read_csv(FILE_FORMAT.format("cells"))


# ### count the number of cells per hit and the total charge deposited per hit, add this to the hits table

# In[134]:


cellagg = cells.groupby("hit_id").agg({"value" : ["sum", "count"]})
cellagg.columns = cellagg.columns.droplevel()
cellagg.reset_index(inplace=True)
cellagg.columns = ["hit_id", "cell_sum", "cell_count"]
hits = pd.merge(hits, cellagg, on="hit_id")


# ### compute the true momentum inclination (theta), and add this to the hits
# Note: theta = 0 implies that the particle is moving along the positive z-axis, while theta=180 implies that it's moving along the negative z-axis. Theta = 90 implies that the particle is moving perpendicular to the z-axis in a radially outward direction.

# In[135]:


truth['theta'] = np.degrees(np.arctan2(np.sqrt(truth.tpy.values ** 2 + truth.tpx.values ** 2), truth.tpz.values))
truth['phi'] = np.degrees(np.arctan2(truth.tpy.values, truth.tpx.values))
hits = pd.merge(hits, truth, on="hit_id")


# ## let's look at the distribution of cell_count and cell_sum for real vs spurious hits

# In[136]:


plt.figure(figsize=(18,6))
plt.subplot(121)
buckets = np.arange(60)
plt.hist(hits.query("weight==0")["cell_count"].values, label="spurious", alpha=.5, normed=True, bins=buckets)
plt.hist(hits.query("weight>0")["cell_count"].values, label="real", alpha=.5, normed=True, bins=buckets)
plt.xlabel("cell count")
plt.grid()
plt.legend()
plt.subplot(122)
buckets = np.linspace(0, 10, 100)
plt.hist(hits.query("weight==0")["cell_sum"].values, label="spurious", alpha=.5, normed=True, bins=buckets)
plt.hist(hits.query("weight>0")["cell_sum"].values, label="real", alpha=.5, normed=True, bins=buckets)
plt.xlabel("cell value sum")
plt.grid()
_ = plt.legend()


# ## the discrete values are coming from some of the volumes, so let's plot the above figure for each volume

# In[150]:


for volume_id in sorted(set(hits.volume_id.values)):
    plt.figure(figsize=(18,6))
    plt.suptitle("Volume {}".format(volume_id))
    plt.subplot(121)
    buckets = np.arange(60)
    plt.hist(hits.query("(volume_id=={}) and (weight==0)".format(volume_id))["cell_count"].values, label="spurious", alpha=.5, normed=True, bins=buckets)
    plt.hist(hits.query("(volume_id=={}) and (weight>0)".format(volume_id))["cell_count"].values, label="real", alpha=.5, normed=True, bins=buckets)
    plt.xlabel("cell count")
    plt.grid()
    plt.legend()
    plt.subplot(122)
    buckets = np.linspace(0, 10, 100)
    plt.hist(hits.query("(volume_id=={}) and (weight==0)".format(volume_id))["cell_sum"].values, label="spurious", alpha=.5, normed=True, bins=buckets)
    plt.hist(hits.query("(volume_id=={}) and (weight>0)".format(volume_id))["cell_sum"].values, label="real", alpha=.5, normed=True, bins=buckets)
    plt.xlabel("cell value sum")
    plt.grid()
    _ = plt.legend()
    plt.show()


# ## let's see the distribution of cell_count for spurious hits in each volume

# In[155]:


plt.figure(figsize=(18,6))
buckets = np.arange(60)
for volume_id in sorted(set(hits.volume_id.values)):
    plt.hist(hits.query("(volume_id=={}) and (weight==0)".format(volume_id))["cell_count"].values, label=str(volume_id), alpha=.5, normed=True, bins=buckets)
plt.xlabel("cell count")
plt.grid()
_ = plt.legend(title="volume")


# ## is there any difference in the spurious hits' cell count for volume 8 across the various layers?

# In[154]:


plt.figure(figsize=(18,6))
buckets = np.arange(60)
for layer_id in sorted(set(hits.query("volume_id==8").layer_id.values)):
    plt.hist(hits.query("(volume_id==8) and (layer_id=={}) and (weight==0)".format(layer_id))["cell_count"].values, label=str(layer_id), alpha=.5, normed=True, bins=buckets)
plt.xlabel("cell count")
plt.grid()
_ = plt.legend(title="layer")


# ## Finally, let's take a look at the distribution of theta as a function of the cell_count for real hits

# In[152]:


plt.figure(figsize=(18, 12))
buckets = np.arange(181)
for cnt in range(1, 10):
    plt.hist(hits.query("(weight > 0) and (cell_count == {})".format(cnt)).theta.values, normed=True, alpha=.5, label=str(cnt), bins=buckets)
plt.hist(hits.query("(weight > 0) and (cell_count > 9)").theta.values, normed=True, alpha=.25, label=">9", bins=buckets, fc='k')
plt.legend(title="cell count")
_ = plt.xlabel("theta")


# ## So, there is clearly some signal here; let's repeat the above analysis on a per-volume basis

# In[153]:


for vol in sorted(set(hits.volume_id.values)):
    plt.figure(figsize=(18, 12))
    plt.title("Volume {}".format(vol))
    buckets = np.arange(181)
    for cnt in range(1, 10):
        plt.hist(hits.query("(weight > 0) and (cell_count == {} and (volume_id == {}))".format(cnt, vol)).theta.values, normed=True, alpha=.5, label=str(cnt), bins=buckets)
    plt.hist(hits.query("(weight > 0) and (cell_count > 9) and (volume_id == {})".format(vol)).theta.values, normed=True, alpha=.25, label=">9", bins=buckets, fc='k')
    plt.legend(title="cell count")
    plt.xlabel("theta")
    plt.show()


# # does the same thing hold for phi? (it shouldn't)

# In[151]:


plt.figure(figsize=(18, 12))
buckets = np.arange(-180, 181)
for cnt in range(1, 10):
    plt.hist(hits.query("(weight > 0) and (cell_count == {})".format(cnt)).phi.values, normed=True, alpha=.5, label=str(cnt), bins=buckets)
plt.hist(hits.query("(weight > 0) and (cell_count > 9)").phi.values, normed=True, alpha=.25, label=">9", bins=buckets, fc='k')
plt.xlabel("phi")
_ = plt.legend(title="cell count")


# # Conclusions
# - The distribution of cell counts for false hits is very similar across all volumes and layers
# - The cell counts for real hits depends very closely on the inclination angle of the particle motion
# 
# 
