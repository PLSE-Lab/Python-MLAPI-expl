#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from astropy.time import Time

import os
print(os.listdir("../input"))


# In[ ]:


train_df = pd.read_csv("../input/training_set.csv", dtype={"object_id": "object"})
train_meta_df = pd.read_csv("../input/training_set_metadata.csv", dtype={"object_id": "object"})

test_df = pd.read_csv("../input/test_set.csv", iterator=True, dtype={"object_id": "object"}) # Large size hence loading as iterator
test_meta_df = pd.read_csv("../input/test_set_metadata.csv", iterator=True, dtype={"object_id": "object"})

sample_submission_df = pd.read_csv("../input/sample_submission.csv", dtype={"object_id": "object"})


# In[ ]:


train_df.head()


# In[ ]:


test_df.get_chunk(size=5)


# In[ ]:


train_meta_df.head()


# In[ ]:


test_meta_df.get_chunk(size=5)


# In[ ]:


print("--------------Shape------------------")
print(" Train: {}\n Train meta: {}".format(train_df.shape, train_meta_df.shape))


# In[ ]:


ID_col = "object_id"
target_col = "target"
ts_index_col = "mjd"
ts_cols = ["passband", "flux", "flux_err", "detected"]
static_cols = ["ra", "decl", "gal_l", "gal_b", "ddf", "hostgal_specz", 
               "hostgal_photoz", "hostgal_photoz_err", "distmod", "mwebv"]


# In[ ]:


train_df[ts_index_col] =train_df[ts_index_col].apply(lambda x : )


# ## Select object_id from dropdown to view all time series variable for that object (train). Only in edit mode

# In[ ]:


@interact(object_id=train_df.object_id.unique())
def plot_ts(object_id):
    fig, axs = plt.subplots(4,sharex=True, figsize=(20,10))
    object_df = train_df[train_df.object_id == object_id]
    for i, col in enumerate(ts_cols):
        axs[i].plot(object_df[col], label=col)
        axs[i].grid()
        axs[i].set_title(col)
    plt.show()


# ## Following inferneces can be drawn from time series plots
# *  "flux" variable follows periodic trend 

# ## Analyzing length of time series for each object id (train). Adjust slider to change number of bins. Only in edit mode

# In[ ]:


@interact(bins= (5, 40))
def plt_series_len(bins):
    plt.figure(figsize=(10, 4))
    plt.hist(train_df[ID_col].value_counts(), bins=bins, color='purple', align= 'left')
    plt.xlabel("Length of Series")
    plt.ylabel("Number of objects")
    plt.show()


# ## Observations
# * It can be seen that most of objects have time series length between 100 and 150 
# * Series length can be used as a feature while modelling. As it can be the case that some astronomical objects are easier to track as compared to others hence have more observations. Also, some astronomical objects can be observed only during a fixed window of time.
# 

# In[ ]:


### Distribution of train labels


# In[ ]:


train_lbls = train_meta_df[target_col].value_counts()

plt.figure(figsize=(20,5))
plt.barh(train_lbls.index.astype("str"), width= train_lbls.values)
plt.ylabel("Class")
plt.xlabel("Number of objects")
plt.title("Class wise object distribution")
plt.show()


# In[ ]:





# In[ ]:




