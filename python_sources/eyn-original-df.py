#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook as tqdm
print(os.listdir("../input"))
print(os.listdir("../input/eyn-original/"))


# In[2]:


dims = ['traj', 'seq', 'info']
info_col = ['t_entry','t_exit','x_entry','y_entry','x_exit','y_exit','vmax','vmin','vmean','tid_0','tid_1']


# In[3]:


train_data = np.load("../input/eyn-original/train_data.npy")
print(train_data.shape)
train_data = np.moveaxis(train_data,0,1)  # should be done on pandas
index = pd.MultiIndex.from_product([range(s)for s in train_data.shape], names=dims)
df_train = pd.DataFrame({'A': train_data.flatten()}, index=index)['A']
df_train = df_train.unstack(level='info').swaplevel().sort_index()
df_train.columns = info_col
df_train[['tid_0','tid_1']] = df_train[['tid_0','tid_1']].astype('category')
df_train.head(10)


# In[4]:


test_data = np.load("../input/eyn-original/test_data.npy")
print(test_data.shape)
test_data = np.moveaxis(test_data,0,1)  # should be done on pandas
index = pd.MultiIndex.from_product([range(s)for s in test_data.shape], names=dims)
df_test = pd.DataFrame({'A': test_data.flatten()}, index=index)['A']
df_test = df_test.unstack(level='info').swaplevel().sort_index()
df_test.columns = info_col
df_test[['tid_0','tid_1']] = df_test[['tid_0','tid_1']].astype('category')
df_test.head(10)


# In[5]:


df_train.to_pickle("df_train.pickle")
df_test.to_pickle("df_test.pickle")


# In[6]:


# to load
df_train = pd.read_pickle("df_train.pickle")
df_test = pd.read_pickle("df_test.pickle")
print(df_train.shape, df_test.shape)


# # BASELINE SUBMISSION

# In[7]:


x_min, x_max = -1., 1.
y_min, y_max = -.3, .3

test_ids = np.load("../input/eyn-original/test_ids.npy")
df_test_notnull = df_test[df_test['t_entry'].notnull()]
df_test_1st_traj_only = df_test_notnull[df_test_notnull['x_exit'].isnull()]

# helper function to determine if point is inside
def is_inside(arr_x, arr_y):
    return ((arr_x > x_min) & 
            (arr_x < x_max) & 
            (arr_y > y_min) & 
            (arr_y < y_max)).astype(float)

df_submit = pd.DataFrame()
df_submit["id"] = test_ids
df_submit['target'] = is_inside(np.array(df_test_1st_traj_only['x_entry']),
                                np.array(df_test_1st_traj_only['y_entry']))
df_submit.to_csv('submission.csv', index=False)
df_submit.head()


# In[8]:


df_submit.tail()


# In[9]:


# to document: specifications for all if not clear enough
# might not care to do: make the pivot table to 3D array faster - omg need to do this


# In[ ]:




