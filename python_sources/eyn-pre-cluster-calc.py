#!/usr/bin/env python
# coding: utf-8

# In[1]:


# place all the points on the map of last-seen-nonstat, excluding itself
# folds information actually not necessary
import os
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
print(os.listdir("../input/eyn-original"))
x_min, x_max = -1., 1.
y_min, y_max = -.3, .3


# In[2]:


# to load
df_train_cluster = pd.read_pickle("../input/eyn-pre-cluster-unif/df_train_cluster.pickle")
df_test_cluster = pd.read_pickle("../input/eyn-pre-cluster-unif/df_test_cluster.pickle")
print(df_train_cluster.shape, df_test_cluster.shape)


# In[3]:


df_train_cluster.head()


# In[4]:


df_train_cluster_values = df_train_cluster.values
df_test_cluster_values = df_test_cluster.values
df_original_columns = df_train_cluster.columns
print(df_train_cluster_values.shape, df_test_cluster_values.shape)


# In[5]:


QUERY_SIZE = df_train_cluster_values[:,0::4].shape[1]

df_train_entry_target = df_train_cluster_values[:,0::4]
df_train_exit_target = df_train_cluster_values[:,2::4]
df_train_entry_dist = df_train_cluster_values[:,1::4]
df_train_exit_dist = df_train_cluster_values[:,3::4]
print(df_train_entry_target.shape, df_train_exit_target.shape, 
      df_train_entry_dist.shape, df_train_exit_dist.shape)

df_test_entry_target = df_test_cluster_values[:,0::4]
df_test_exit_target = df_test_cluster_values[:,2::4]
df_test_entry_dist = df_test_cluster_values[:,1::4]
df_test_exit_dist = df_test_cluster_values[:,3::4]
print(df_test_entry_target.shape, df_test_exit_target.shape, 
      df_test_entry_dist.shape, df_test_exit_dist.shape)


# In[6]:


plt.figure(figsize=(24,4))
for i in range(1,10):
    plt.hist(list(df_train_entry_dist[::21,i]), density=True, bins='scott', histtype='step')
    plt.axvline(x=np.mean(df_train_entry_dist[::21,i]))
plt.show()


# In[7]:


DECAY_CONST = 0.001
df_train_entry_dist_w = np.exp(-np.array(df_train_entry_dist, dtype=np.float32) / 0.001)
df_train_exit_dist_w = np.exp(-np.array(df_train_exit_dist, dtype=np.float32) / 0.001)
df_test_entry_dist_w = np.exp(-np.array(df_test_entry_dist, dtype=np.float32) / 0.001)
df_test_exit_dist_w = np.exp(-np.array(df_test_exit_dist, dtype=np.float32) / 0.001)


# In[8]:


df_train_entry_w_dot = df_train_entry_target-0.5 * df_train_entry_dist_w
df_train_exit_w_dot = df_train_exit_target-0.5 * df_train_entry_dist_w
df_test_entry_w_dot = df_test_entry_target-0.5 * df_test_entry_dist_w
df_test_exit_w_dot = df_test_exit_target-0.5 * df_test_entry_dist_w
print(df_train_entry_w_dot.shape, df_train_exit_w_dot.shape, 
      df_test_entry_w_dot.shape, df_test_exit_w_dot.shape)


# In[9]:


entry_col_names_weighted = ["c{}".format(i) for i in range(QUERY_SIZE)]
exit_col_names_weighted = ["d{}".format(i) for i in range(QUERY_SIZE)]

for i,(a,b) in enumerate(zip(entry_col_names_weighted, exit_col_names_weighted)):
    df_train_cluster[a] = df_train_entry_w_dot[:,i].ravel()
    df_train_cluster[b] = df_train_exit_w_dot[:,i].ravel()
    df_test_cluster[a] = df_test_entry_w_dot[:,i].ravel()
    df_test_cluster[b] = df_test_exit_w_dot[:,i].ravel()


# In[10]:


df_train_cluster = df_train_cluster.drop(columns=df_original_columns)
df_test_cluster = df_test_cluster.drop(columns=df_original_columns)


# In[11]:


df_train_cluster.to_pickle("df_train_cluster_w.pickle")
df_test_cluster.to_pickle("df_test_cluster_w.pickle")


# In[20]:


get_ipython().run_line_magic('reset', '-sf')


# In[43]:


# place all the points on the map of last-seen-nonstat, excluding itself
# folds information actually not necessary
import os
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
print(os.listdir("../input/eyn-original"))
x_min, x_max = -1., 1.
y_min, y_max = -.3, .3


# In[44]:


# try loading
df_train_cluster_w = pd.read_pickle("df_train_cluster_w.pickle")
df_test_cluster_w = pd.read_pickle("df_test_cluster_w.pickle")
print(df_train_cluster_w.shape, df_test_cluster_w.shape)
df_train_cluster_w.head()


# In[45]:


df_train_cluster_w_values = df_train_cluster_w.values
df_test_cluster_w_values = df_test_cluster_w.values
df_original_columns = df_test_cluster_w.columns


# In[46]:


df_train_cluster_w_sum_entry = np.sum(df_train_cluster_w_values[:,0::2], axis=1)
df_train_cluster_w_sum_exit = np.sum(df_train_cluster_w_values[:,1::2], axis=1)
df_test_cluster_w_sum_entry = np.sum(df_test_cluster_w_values[:,0::2], axis=1)
df_test_cluster_w_sum_exit = np.sum(df_test_cluster_w_values[:,1::2], axis=1)

df_train_cluster_w_sum_entry[::21] = np.nansum(df_train_cluster_w_values[::21,0::2], axis=1)
df_test_cluster_w_sum_entry[::21] = np.nansum(df_test_cluster_w_values[::21,0::2], axis=1)


# In[47]:


df_train_cluster_w["sum_entry"] = df_train_cluster_w_sum_entry
df_train_cluster_w["sum_exit"] = df_train_cluster_w_sum_exit
df_test_cluster_w["sum_entry"] = df_test_cluster_w_sum_entry
df_test_cluster_w["sum_exit"] = df_test_cluster_w_sum_exit


# In[48]:


df_train_cluster_w.head(7)


# In[49]:


df_train_cluster_w = df_train_cluster_w.drop(columns=df_original_columns)
df_test_cluster_w = df_test_cluster_w.drop(columns=df_original_columns)


# In[50]:


df_train_cluster_w.to_pickle("df_train_cluster_s.pickle")
df_test_cluster_w.to_pickle("df_test_cluster_s.pickle")


# In[52]:


df_test_cluster_w.head(7)


# In[ ]:




