#!/usr/bin/env python
# coding: utf-8

# # Cleaning useless data to load train.csv faster
# We can split `train.csv` in 3 groups of installation ids:
# 1. Children who tried at least one assessment throughout their game play history (useful)
# 2. Children who never tried any assessment (useless)
# 3. Children who tried at least one assessment, but for whom we do not have labeled data to train a model (useful in later stages)
# 
# Group 1 is very useful, in what we should focus on. Group 2 is completely useless (if you disagree, please leave a comment!).
# Group 3 might be useful in a more advanced way to clustering, in later stages of the competition (unsupervised learning/autoencoders).
# Let's check each group's size and get rid of the useless.

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


get_ipython().run_cell_magic('time', '', "train = pd.read_csv('/kaggle/input/data-science-bowl-2019/train.csv')\ntrain_labels = pd.read_csv('/kaggle/input/data-science-bowl-2019/train_labels.csv')")


# In[ ]:


train.shape[0], train['installation_id'].unique().shape[0]


# As we can see, over **11 million rows**, from **17,000 unique installation ids**. However, most of those ids are useless:

# In[ ]:


t1 = train.groupby(['installation_id', 'type']).agg({'game_session': 'count'}).reset_index()
t2 = t1.pivot(index='installation_id', columns='type', values='game_session').reset_index()
t2


# The table above is the **pivot table** that aggregates the count of **activity types** (children either do an activity, do an assessment, watch a clip or play a game) by installation id. From that it's easy to count how many installation ids never did a single assessment:

# In[ ]:


t2['Assessment'].isna().sum(), t2['Assessment'].isna().sum() / t2.shape[0]


# There it is: **almost 13 thousand installation ids, 75% of them, never did a single assignment**. They are useless. Let's see how much we gain by removing all rows that belong to these ids from the main table:

# In[ ]:


useful_installation_ids = t2[~t2['Assessment'].isna()]['installation_id'].values
clean_train = train[train['installation_id'].isin(useful_installation_ids)]
clean_train.shape[0], clean_train.shape[0] / train.shape[0]


# Removing these useless records brings the train DataFrame **down to 8 million rows, 73% of its original size**.
# This, imho, is the useful dataset. Now, assuming that (at least for now), we won't use any unsupervised learning technique (i.e. we need labels), we can reduce it even further:

# In[ ]:


useful_installation_ids_sl = train_labels['installation_id'].unique()
clean_train_sl = train[train['installation_id'].isin(useful_installation_ids_sl)]
clean_train_sl.shape[0], clean_train_sl.shape[0] / train.shape[0]


# Removing all unlabelled data from the train DataFrame reduces its size **down to 7.7 million rows, 68% of its original size**.
# It's much faster to work with this than it is to work with the original file.
# 
# Cheers!

# In[ ]:




