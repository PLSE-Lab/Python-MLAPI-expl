#!/usr/bin/env python
# coding: utf-8

# Main topic of this kernel is how to improve your score with features and settigs what you already have.
# 
# Let's look down...

# # Some Imports

# In[ ]:


import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool

from sklearn.model_selection import train_test_split


# In[ ]:


SEED = 1981
index_col = 'match_id_hash'
PATH_TO_DATA = '../input/mlcourse-dota2-win-prediction/'


# # Data loading

# In[ ]:


df_train = pd.read_csv(PATH_TO_DATA + 'train_features.csv', index_col=index_col)
df_targets = pd.read_csv(PATH_TO_DATA + 'train_targets.csv', index_col=index_col)


# # CatBoost Baseline

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(df_train, df_targets['radiant_win'].astype('int'), test_size=0.2, random_state=SEED)


# In[ ]:


cat_features = []
params_cb = {
    'loss_function': 'Logloss',
    'eval_metric': 'AUC',
    'verbose': 100,
    'random_seed': SEED,
    'thread_count': -1,
    'iterations':1000,
}

model_cb = CatBoostClassifier(**params_cb)


# In[ ]:


model_cb.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    logging_level='Verbose',
    plot=True)


# # What do we have?

# We used CatBoost and unmodified dataset to get some baseline score. We got 0.8010972357 like this.
# I hope no one will keep this features in real models.
# 
# I wanna show how to get better score with features what you have.

# # Target Upsample

# **Let's make dataset two times bigger by swap dire and radiants with changing target to opposite of course.**

# In[ ]:


df_targets_upsample = df_targets.reset_index()
df_targets_upsample['match_id_hash'] = df_targets_upsample['match_id_hash'] + '_up'
df_targets_upsample.set_index('match_id_hash', inplace=True)
df_targets_upsample['radiant_win'] = df_targets_upsample['radiant_win'].map({False: True, True: False})

df_targets = df_targets.append(df_targets_upsample, sort=False)


# # Train Upsample

# In[ ]:


df_train_upsample = df_train.reset_index()
df_train_upsample['match_id_hash'] = df_train_upsample['match_id_hash'] + '_up'
df_train_upsample.set_index('match_id_hash', inplace=True)


# **Let's rename columns. Dire will become radiants and Radiants will become Dire**

# In[ ]:


columns_names = df_train_upsample.columns
col_dict = {}
for col in columns_names:
    new_col = col
    if col[0] == 'd':
        new_col = col.replace('d', 'r', 1)

    elif col[0] == 'r':
        new_col = col.replace('r', 'd', 1)
    col_dict[col] = new_col


# ** Let's mirror coordinates**

# In[ ]:


for i in range(1, 6):
    df_train_upsample[f'd{i}_y'] = 186 - df_train_upsample[f'd{i}_y'] + 70
    df_train_upsample[f'r{i}_y'] = 186 - df_train_upsample[f'r{i}_y'] + 70
    
    df_train_upsample[f'd{i}_x'] = 186 - df_train_upsample[f'd{i}_x'] + 66
    df_train_upsample[f'r{i}_x'] = 186 - df_train_upsample[f'r{i}_x'] + 66    


# In[ ]:


df_train_upsample.rename(columns=col_dict, inplace=True)


# In[ ]:


df_train = df_train.append(df_train_upsample, sort=False)


# # Time to check what we got

# In[ ]:


X_train, X_valid, y_train, y_valid = train_test_split(df_train, df_targets['radiant_win'].astype('int'), test_size=0.2, random_state=SEED)


# In[ ]:


model_cb.fit(
    X_train,
    y_train,
    cat_features=cat_features,
    eval_set=(X_valid, y_valid),
    logging_level='Verbose',
    plot=True)


# # What do we have?

# * We got + 0.01 score
# * More good features you have more you can get from it

# # Found it interesting? Wanna More Secrets?

# ## Upvote this kernel then
