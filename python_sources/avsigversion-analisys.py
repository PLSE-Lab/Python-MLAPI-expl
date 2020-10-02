#!/usr/bin/env python
# coding: utf-8

# > "the dataset provided here has been roughly split by time"
# 
# So we need carefull feature engineering.

# In[ ]:


import pandas as pd
from matplotlib import pyplot as plt

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


TRAIN_FILE = "../input/train.csv"
TEST_FILE = "../input/test.csv"
f = 'AvSigVersion'


# In[ ]:


train_df = pd.read_csv(TRAIN_FILE, usecols=[f])
test_df = pd.read_csv(TEST_FILE, usecols=[f])
df = pd.concat([train_df,test_df])


# Let's convert to serial number.

# In[ ]:


unq_vals = df[f].value_counts().reset_index()
unq_vals = pd.concat([unq_vals, unq_vals['index'].str.replace('&#x17;','').str.split('.', expand=True).astype(int)], axis=1)
sorted_vals = unq_vals.sort_values([0,1,2,3]).reset_index(drop=True).reset_index()
vals_dict = sorted_vals.set_index('index')['level_0'].to_dict()
df[f] = df[f].map(vals_dict)


# In[ ]:


train_len = len(train_df)
train_df = df[:train_len]
test_df = df[train_len:]


# In[ ]:


_ = plt.hist([train_df[f],test_df[f]], stacked=False)


# Most of values in training data has < 9000. And most of values in test data has > 9000.
# 
# So if we use AvSigVersion for training, we would get better CV score and worse or a little better LB score.
# 
# I just used AvSigVersion as an example, but there would be many other similar features.

# In[ ]:




