#!/usr/bin/env python
# coding: utf-8

# ## This Kernel is a start point to analyze attributes of events
# 
# The `event_data` column is informative, but hard to break down, especially with the time and memory limitation.
# This kernel gives a chunk-wise data loading and the distribution of attributes in both train and test data.

# In[ ]:


import numpy as np
import pandas as pd
import json
import gc
import ast
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max_columns', 100)
pd.set_option('max_rows', 100)


# In[ ]:


get_ipython().run_cell_magic('time', '', "train_csv = '/kaggle/input/data-science-bowl-2019/train.csv'\ntest_csv = '/kaggle/input/data-science-bowl-2019/test.csv'\nspecs_csv = '/kaggle/input/data-science-bowl-2019/specs.csv'\n\nall_train_event_attributes, all_test_event_attributes = [], []\ntrain_count, test_count = 0, 0\nfor chunk in pd.read_csv(train_csv,chunksize=10000):\n    chunk_attributes = chunk['event_data'].apply(lambda x: list(json.loads(x).keys()))\n    all_train_event_attributes.extend([y for x in chunk_attributes.to_list() for y in x])\n    train_count += chunk.shape[0]\n    \nfor chunk in pd.read_csv(test_csv,chunksize=10000):\n    chunk_attributes = chunk['event_data'].apply(lambda x: list(json.loads(x).keys()))\n    all_test_event_attributes.extend([y for x in chunk_attributes.to_list() for y in x])\n    test_count += chunk.shape[0]")


# In[ ]:


get_ipython().run_cell_magic('time', '', "count_train = Counter(all_train_event_attributes)\ncount_test = Counter(all_test_event_attributes)\n\ndef get_count_df(count_dict, total):\n    df = pd.DataFrame.from_dict(count_dict, orient='index')\n    df['attribute']=df.index\n    df.columns = ['count', 'attribute']\n    df.sort_values(by=['count'], axis=0, ascending=False, inplace=True)\n    df['pct'] = df['count'] / total\n    return df\n\ncount_train_df = get_count_df(count_train, train_count)\ncount_test_df = get_count_df(count_test, test_count)")


# In[ ]:


plt.figure(figsize=(10, 30))
sns.set(style='whitegrid')
ax = sns.barplot(x='pct', y='attribute', data=count_train_df.head(50))


# In[ ]:


plt.figure(figsize=(10, 30))
sns.set(style='whitegrid')
ax = sns.barplot(x='pct', y='attribute', data=count_test_df.head(50))


# ### to check the attributes to which events, we can check spec table

# In[ ]:


specs = pd.read_csv(specs_csv)
specs_parse = lambda _col:str([x['name'] for x in json.loads(_col)])
specs['attribute_list'] = specs['args'].apply(lambda _col: specs_parse(_col))
specs.head()

