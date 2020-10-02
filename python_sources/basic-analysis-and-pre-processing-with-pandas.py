#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# # Read the data

# In[ ]:


get_ipython().run_line_magic('time', "train = pd.read_csv('../input/train.csv')")
train.shape, train.columns


# In[ ]:


get_ipython().run_line_magic('time', "test = pd.read_csv('../input/test.csv')")
test.shape, test.columns


# # Look at the data

# ## Overview

# ### 3147 integer columns, 1845 float columns (including target) and 1 string column (ID)

# In[ ]:


train.dtypes.value_counts()


# ### No missing values; all values are nonnegative

# In[ ]:


features = train.drop(columns=['ID', 'target'])
features.min().min(), features.max().max(), features.isnull().any().any()


# In[ ]:


test_features = test.drop(columns='ID').sample(n=features.shape[0], random_state=123)
test_features.min().min(), test_features.max().max(), test_features.isnull().any().any()


# ### Almost all entries are zeros (very sparse data)

# In[ ]:


plt.figure(figsize=(5,5))
plt.spy((features > 0).values);


# In[ ]:


(features == 0).sum().sum() / features.size * 100


# In[ ]:


plt.figure(figsize=(5,5))
plt.spy((test_features > 0).values);


# In[ ]:


(test_features == 0).sum().sum() / test_features.size * 100


# ### Columns carrying no information: 256 => can remove them right away

# In[ ]:


nunique = features.nunique()
no_info = nunique == 1
no_info.sum()


# In[ ]:


to_drop = nunique[no_info].index.values
train.drop(columns=to_drop, inplace=True)
features.drop(columns=to_drop, inplace=True)
test.drop(columns=to_drop, inplace=True)
test_features.drop(columns=to_drop, inplace=True)


# ### Duplicate rows: only two and with different targets

# In[ ]:


train.loc[features.duplicated(keep=False), ['ID', 'target']]


# ### Duplicate columns: in training, but not in test

# In[ ]:


trans = features.T
all_duplicates = trans[trans.duplicated(keep=False)].index
last_duplicates = trans[trans.duplicated()].index
all_duplicates, last_duplicates


# In[ ]:


test_sample = test_features.sample(n=features.shape[0], random_state=123)
trans_test = test_sample.T
trans_test[trans_test.duplicated(keep=False)].index


# In[ ]:


for i in range(len(all_duplicates)):
    for j in range(i + 1, len(all_duplicates)):
        col1, col2 = all_duplicates[i], all_duplicates[j]
        print(col1, col2, 'train:', sum(train[col1] != train[col2]), ' test:', sum(test_sample[col1] != test_sample[col2]))


# ## Target

# ### No zeros: starts from 30,000.

# In[ ]:


train.target.describe()


# ### Approximately linear on a log scale

# In[ ]:


fig, ax = plt.subplots()
plt.scatter(range(train.shape[0]), np.sort(train.target.values));
ax.set_yscale('log')


# ## Integer columns

# In[ ]:


int_cols = features.columns[features.dtypes == np.int64].values
int_train = features[int_cols]


# ### Very sparse data

# In[ ]:


plt.figure(figsize=(5,10))
plt.spy((int_train > 0).values);


# In[ ]:


(int_train == 0).sum().sum() / int_train.size * 100


# ### Number of unique values (could be lots of categorical data)

# In[ ]:


nunique_int = int_train.nunique()
fig, ax = plt.subplots()
nunique_int.hist(bins=300, bottom=0.1)
ax.set_xscale('log')


# ## Float columns

# In[ ]:


float_cols = features.columns[features.dtypes == np.float64].values
float_train = features[float_cols]


# ### Slightly less sparse than int

# In[ ]:


float_train = train[float_cols]
plt.figure(figsize=(5,10))
plt.spy((float_train > 0).values);


# In[ ]:


(float_train == 0).sum().sum() / float_train.size * 100


# ### The distribution of unique counts is notably different between floats and ints

# In[ ]:


nunique_float = float_train.nunique()
fig, ax = plt.subplots()
nunique_float.hist(bins=300, bottom=0.1)
ax.set_xscale('log')


# > # Save the data in the feather format

# In[ ]:


train.target = np.log1p(train.target)
get_ipython().run_line_magic('time', "train.to_feather('train.feather')")
get_ipython().run_line_magic('time', "train = pd.read_feather('train.feather')")


# In[ ]:


# No space left in Kaggle, but can be done locally
# %time test.to_feather('test.feather')
# %time test = pd.read_feather('test.feather')


# In[ ]:




