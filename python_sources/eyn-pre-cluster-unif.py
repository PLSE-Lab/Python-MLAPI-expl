#!/usr/bin/env python
# coding: utf-8

# In[67]:


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

QUERY_SIZE = 100


# In[68]:


train_data = np.load("../input/eyn-original/train_data.npy")
test_data = np.load("../input/eyn-original/test_data.npy")
train_targets = np.load("../input/eyn-original/train_targets.npy")
train_targets_inside = np.load("../input/eyn-original/train_targets_inside.npy")
train_targets_inside_indexes = np.argwhere(train_targets_inside == 1)
print(train_data.shape, train_targets.shape)
print(train_targets_inside.shape, train_targets_inside_indexes.shape)
print(test_data.shape)
# 't_entry','t_exit','x_entry','y_entry','x_exit','y_exit','vmax','vmin','vmean','tid_0','tid_1'


# In[69]:


# the queries
train_entry_loc = np.concatenate((train_data[:,:,2:4], 
                                  train_data[:,:,0:2]*0.00001 + 
                                  np.random.randn(*train_data[:,:,0:2].shape)*0.0000001), axis=2)
train_exit_loc = np.concatenate((train_data[:,:,4:6], train_data[:,:,0:2]*0.0001), axis=2)
test_entry_loc = np.concatenate((test_data[:,:,2:4], test_data[:,:,0:2]*0.0001), axis=2)
test_exit_loc = np.concatenate((test_data[:,:,4:6], test_data[:,:,0:2]*0.0001), axis=2)
print(train_entry_loc.shape, train_exit_loc.shape)
print(test_entry_loc.shape, test_exit_loc.shape)


# In[73]:


# building the reference
train_last_nonstat_indexes = np.load("../input/eyn-folds/train_last_not_stationary.npy")
print(train_last_nonstat_indexes.shape)
train_last_nonstat_loc = train_entry_loc[train_last_nonstat_indexes, 0, :] 
train_last_nonstat_target = train_targets_inside[train_last_nonstat_indexes]
print(train_last_nonstat_loc.shape, train_last_nonstat_target.shape)


# In[74]:


from scipy import spatial

reference = train_last_nonstat_loc
reference_tree = spatial.KDTree(reference)


# In[75]:


train_entry_query_dist, train_entry_query_indexes = reference_tree.query(train_entry_loc,k=QUERY_SIZE)
print(train_entry_query_dist.shape, train_entry_query_indexes.shape)


# In[ ]:


train_exit_query_dist, train_exit_query_indexes = reference_tree.query(train_exit_loc,k=QUERY_SIZE)
print(train_exit_query_dist.shape, train_exit_query_indexes.shape)


# In[ ]:


test_entry_query_dist, test_entry_query_indexes = reference_tree.query(test_entry_loc,k=QUERY_SIZE)
print(test_entry_query_dist.shape, test_entry_query_indexes.shape)


# In[ ]:


test_exit_query_dist, test_exit_query_indexes = reference_tree.query(test_exit_loc,k=QUERY_SIZE)
print(test_exit_query_dist.shape, test_exit_query_indexes.shape)


# In[ ]:


train_entry_query_target = np.take(train_last_nonstat_target, train_entry_query_indexes, mode = 'clip')
train_exit_query_target = np.take(train_last_nonstat_target, train_exit_query_indexes, mode = 'clip')
test_entry_query_target = np.take(train_last_nonstat_target, test_entry_query_indexes, mode = 'clip')
test_exit_query_target = np.take(train_last_nonstat_target, test_exit_query_indexes, mode = 'clip')
print(train_entry_query_target.shape, train_exit_query_target.shape)
print(test_entry_query_target.shape, test_exit_query_target.shape)


# In[ ]:


# final exit values should be nans
train_exit_query_target[:,0,:] = np.nan
train_exit_query_dist[:,0,:] = np.nan
test_exit_query_target[:,0,:] = np.nan
test_exit_query_dist[:,0,:] = np.nan

# should exclude the last-seen-position
train_entry_query_target[:,0,0] = np.nan
train_entry_query_dist[:,0,0] = np.nan


# In[ ]:


print(train_entry_query_target.shape, train_exit_query_target.shape)
print(train_entry_query_dist.shape, train_exit_query_dist.shape)
print(test_entry_query_target.shape, test_exit_query_target.shape)
print(test_entry_query_dist.shape, test_exit_query_dist.shape)


# In[ ]:


df_train = pd.read_pickle("../input/eyn-pre-unravel-df/df_train.pickle")
df_test = pd.read_pickle("../input/eyn-pre-unravel-df/df_test.pickle")
df_original_columns = df_train.columns


# In[ ]:


entry_col_names_target = ["at{}".format(i) for i in range(QUERY_SIZE)]
entry_col_names_dist = ["ad{}".format(i) for i in range(QUERY_SIZE)]
exit_col_names_target = ["bt{}".format(i) for i in range(QUERY_SIZE)]
exit_col_names_dist = ["bd{}".format(i) for i in range(QUERY_SIZE)]


# In[ ]:


for i,(a,b,c,d) in enumerate(zip(entry_col_names_target, exit_col_names_target,
                                 entry_col_names_dist, exit_col_names_dist)):
    df_train[a] = train_entry_query_target[:,:,i].ravel()
    df_train[c] = train_entry_query_dist[:,:,i].ravel()
    df_train[b] = train_exit_query_target[:,:,i].ravel()
    df_train[d] = train_exit_query_dist[:,:,i].ravel()
    
    df_test[a] = test_entry_query_target[:,:,i].ravel()
    df_test[c] = test_entry_query_dist[:,:,i].ravel()
    df_test[b] = test_exit_query_target[:,:,i].ravel()
    df_test[d] = test_exit_query_dist[:,:,i].ravel()


# In[ ]:


# reinforce nan values for nan entries
df_train.loc[np.isnan(train_data[:,:,2].ravel()), :] = np.nan
df_test.loc[np.isnan(test_data[:,:,2].ravel()), :] = np.nan


# In[ ]:


df_train.head(8)


# In[ ]:


df_test.head(6)


# In[ ]:


df_train[entry_col_names_target] = df_train[entry_col_names_target].astype('category')
df_train[exit_col_names_target] = df_train[exit_col_names_target].astype('category')
df_test[entry_col_names_target] = df_test[entry_col_names_target].astype('category')
df_test[exit_col_names_target] = df_test[exit_col_names_target].astype('category')
df_train = df_train.drop(columns=df_original_columns)
df_test = df_test.drop(columns=df_original_columns)


# In[ ]:


df_train.head(10)


# In[ ]:


df_train.to_pickle("df_train_cluster.pickle")
df_test.to_pickle("df_test_cluster.pickle")


# In[ ]:


# to load
df_train = pd.read_pickle("df_train_cluster.pickle")
df_test = pd.read_pickle("df_test_cluster.pickle")
print(df_train.shape, df_test.shape)


# In[76]:


str_list = []
for i in range(len(train_entry_query_dist)):
    str_list.append(str([train_entry_query_dist[i,0,:]]))
print(len(str_list))
print(len(set(str_list)))
print(len(np.array(str_list)[train_last_nonstat_indexes]))
print(len(set(np.array(str_list)[train_last_nonstat_indexes])))


# In[78]:


a = np.argwhere((train_entry_query_indexes[:,0,0][train_last_nonstat_indexes] - np.arange(len(train_last_nonstat_indexes)))!=0)[:,0]
a


# In[79]:


train_entry_query_indexes[:,0,0][train_last_nonstat_indexes][a]


# In[80]:


train_entry_query_indexes[:,0,1][train_last_nonstat_indexes][a]


# In[81]:


train_entry_query_indexes[:,0,2][train_last_nonstat_indexes][a]


# In[ ]:




