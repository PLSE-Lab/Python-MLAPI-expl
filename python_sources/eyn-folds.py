#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
print(os.listdir("../input"))
np.random.seed(42)


# In[ ]:


train_targets_inside = np.load("../input/eyn-original/train_targets_inside.npy")
df_train = pd.read_pickle("../input/eyn-pre-unravel-df/df_train.pickle")
df_test = pd.read_pickle("../input/eyn-pre-unravel-df/df_test.pickle")
print(df_train.shape, df_test.shape)
df_train


# In[ ]:


df_train_values = df_train.values
df_test_values = df_test.values
df_train_columns = df_train.columns
df_test_columns = df_test.columns


# ### original 4 folds

# In[ ]:


# standardised 4-fold train-test split for clustering purposes
from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
trn_index_list = []
val_index_list = []
for trn_index, val_index in skf.split(np.arange(len(df_train_values[::21])),
                                      train_targets_inside.astype(int)):
    trn_index_list.append(trn_index)
    val_index_list.append(val_index)
    
np.save("trn_index_list",trn_index_list)
np.save("val_index_list",val_index_list)


# ### 10 folds

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
trn_index_list = []
val_index_list = []
for trn_index, val_index in skf.split(np.arange(len(df_train_values[::21])),
                                      train_targets_inside.astype(int)):
    trn_index_list.append(trn_index)
    val_index_list.append(val_index)
    
np.save("trn_index_list_10f",trn_index_list)
np.save("val_index_list_10f",val_index_list)


# ### 15 folds

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
trn_index_list = []
val_index_list = []
for trn_index, val_index in skf.split(np.arange(len(df_train_values[::21])),
                                      train_targets_inside.astype(int)):
    trn_index_list.append(trn_index)
    val_index_list.append(val_index)
    
np.save("trn_index_list_15f",trn_index_list)
np.save("val_index_list_15f",val_index_list)


# ## Inside and stationary indexes

# In[ ]:


train_last_is_stationary = np.argwhere(df_train_values[::21,df_train_columns.get_loc("dur")] == 0)[:,0]
train_last_not_stationary = np.argwhere(df_train_values[::21,df_train_columns.get_loc("dur")] != 0)[:,0]
test_last_is_stationary = np.argwhere(df_test_values[::21,df_test_columns.get_loc("dur")] == 0)[:,0]
test_last_not_stationary = np.argwhere(df_test_values[::21,df_test_columns.get_loc("dur")] != 0)[:,0]
print(train_last_is_stationary.shape, train_last_not_stationary.shape)
print(test_last_is_stationary.shape, test_last_not_stationary.shape)

train_last_seen_is_inside = np.argwhere(df_train_values[::21,df_train_columns.get_loc("entry_in")] == 1)[:,0]
train_last_seen_not_inside = np.argwhere(df_train_values[::21,df_train_columns.get_loc("entry_in")] == 0)[:,0]
test_last_seen_is_inside = np.argwhere(df_test_values[::21,df_test_columns.get_loc("entry_in")] == 1)[:,0]
test_last_seen_not_inside = np.argwhere(df_test_values[::21,df_test_columns.get_loc("entry_in")] == 0)[:,0]
print(train_last_seen_is_inside.shape, train_last_seen_not_inside.shape)
print(test_last_seen_is_inside.shape, test_last_seen_not_inside.shape)

np.save("train_last_is_stationary", train_last_is_stationary)
np.save("train_last_not_stationary", train_last_not_stationary)
np.save("test_last_is_stationary", test_last_is_stationary)
np.save("test_last_not_stationary", test_last_not_stationary)
np.save("train_last_seen_is_inside", train_last_seen_is_inside)
np.save("train_last_seen_not_inside", train_last_seen_not_inside)
np.save("test_last_seen_is_inside", test_last_seen_is_inside)
np.save("test_last_seen_not_inside", test_last_seen_not_inside)

last_nonstat_indexes = train_last_not_stationary


# ## Other number of folds
# 
# Now we exclude points that end up to be stationary

# ### 10 folds unreserved stationary-excluded
# without reserve for ensembling

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
trn_index_list = []
val_index_list = []
for trn_index, val_index in skf.split(np.arange(len(df_train_values[::21])), train_targets_inside.astype(int)):
    trn_index = np.intersect1d(trn_index, last_nonstat_indexes)
    val_index = np.intersect1d(val_index, last_nonstat_indexes)
    trn_index_list.append(trn_index)
    val_index_list.append(val_index)
print(trn_index.shape, val_index.shape)
print(trn_index, val_index)

np.save("trn_index_list_10f_nostat",trn_index_list)
np.save("val_index_list_10f_nostat",val_index_list)


# ### 15 folds unreserved 

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=15, shuffle=True, random_state=42)
trn_index_list = []
val_index_list = []
for trn_index, val_index in skf.split(np.arange(len(df_train_values[::21])), train_targets_inside.astype(int)):
    trn_index = np.intersect1d(trn_index, last_nonstat_indexes)
    val_index = np.intersect1d(val_index, last_nonstat_indexes)
    trn_index_list.append(trn_index)
    val_index_list.append(val_index)
print(trn_index.shape, val_index.shape)
print(trn_index, val_index)

np.save("trn_index_list_15f_nostat",trn_index_list)
np.save("val_index_list_15f_nostat",val_index_list)


# ## Now with reserve for emsemble
# 
# 

# In[ ]:


np.random.shuffle(last_nonstat_indexes)

trn_val_index, ems_index = np.split(last_nonstat_indexes, [int(.9*len(last_nonstat_indexes))])
trn_val_index = np.sort(trn_val_index)
ems_index = np.sort(ems_index)
print(ems_index.shape, trn_val_index.shape)


# ### 10 folds reserved

# In[ ]:


from sklearn.model_selection import StratifiedKFold, KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
trn_index_list = []
val_index_list = []
for trn_index, val_index in skf.split(np.arange(len(df_train_values[::21])), train_targets_inside.astype(int)):
    trn_index = np.intersect1d(np.intersect1d(trn_index, last_nonstat_indexes), trn_val_index)
    val_index = np.intersect1d(np.intersect1d(val_index, last_nonstat_indexes), trn_val_index)
    trn_index_list.append(trn_index)
    val_index_list.append(val_index)
print(trn_index.shape, val_index.shape, ems_index.shape)
print(trn_index, val_index)
    
np.save("trn_index_list_e10_10f_nostat",trn_index_list)
np.save("val_index_list_e10_10f_nostat",val_index_list)
np.save("ems_index_e10_nostat",ems_index)


# In[ ]:


get_ipython().system('ls')


# In[ ]:




