#!/usr/bin/env python
# coding: utf-8

# # Dota 2 Winner Prediction: How to make use of hero ids

# This kernel attempts to show how to process hero ids to make them useful for prediction.

# In[1]:


from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

PATH_TO_DATA = Path('../input')
N_ESTIMATORS = 1500
SEED = 42


# Let's load the data and combine into one dataframe. We need `train_size` to split it back into train and test later.

# In[2]:


y_train = pd.read_csv(PATH_TO_DATA / 'train_targets.csv', index_col='match_id_hash')['radiant_win']
y_train = y_train.map({True: 1, False: 0})
train_df = pd.read_csv(PATH_TO_DATA / 'train_features.csv', index_col='match_id_hash')
test_df = pd.read_csv(PATH_TO_DATA / 'test_features.csv', index_col='match_id_hash')
full_df = pd.concat([train_df, test_df], sort=False)
train_size = train_df.shape[0]


# We will use only hero columns in this kernel:

# In[3]:


hero_columns = [c for c in full_df.columns if '_hero_' in c]
full_df = full_df[hero_columns]
full_df.head()


# ## Raw ids

# Is there any sense in raw ids? Let's see what score these features can give as numbers.
# If we look at correlation with target, we see that they don't correlate well with it:

# In[4]:


train_df = full_df.iloc[:train_size, :]
test_df = full_df.iloc[train_size:, :]
train_df.corrwith(y_train).abs().sort_values(ascending=False).head(12)


# Build a classifier and see the score for raw features. 
# If you run this kernel yourself, you will see nice interactive charts after each `evaluate()`.

# In[5]:


def evaluate():
    from catboost import CatBoostClassifier, Pool
    train_df_part, valid_df, y_train_part, y_valid =         train_test_split(train_df, y_train, test_size=0.25, random_state=SEED)
    cat_features_idx = np.where(train_df.dtypes == 'object')[0].tolist()
    catboost_dataset = Pool(train_df_part, label=y_train_part, cat_features=cat_features_idx)
    catboost_dataset_valid = Pool(valid_df, label=y_valid, cat_features=cat_features_idx)
    catboost_classifier = CatBoostClassifier(
        eval_metric='AUC', depth=5, learning_rate=0.02,
        random_seed=17, verbose=False, n_estimators=N_ESTIMATORS, task_type='GPU')
    catboost_classifier.fit(catboost_dataset, eval_set=catboost_dataset_valid, plot=True)
    valid_pred = catboost_classifier.predict_proba(valid_df)[:, 1]
    score = roc_auc_score(y_valid, valid_pred)
    print('Score:', score)
    return catboost_classifier


# In[ ]:


classifier = evaluate()


# ## Categorical features

# Now let's convert raw ids to `str` so that they are processes as categorical ones, not as numerical.

# In[8]:


full_df = full_df.astype(str)
train_df = full_df.iloc[:train_size, :]
test_df = full_df.iloc[train_size:, :]
classifier = evaluate()


# The score has increased, but we can do it better.

# ## Dummies

# Now let's think what hero ids mean. Some heroes might be stronger than other.
# But what matters for victory is what heroes each team has, no matter what player.
# Let's build dummies for each column and sum them for each team.
# This way we get a table with a column per hero. Each row will contain `1` if a corresponding hero was in this match.

# In[6]:


for team in 'r', 'd':
    players = [f'{team}{i}' for i in range(1, 6)]
    hero_columns = [f'{player}_hero_id' for player in players]
    d = pd.get_dummies(full_df[hero_columns[0]])
    for c in hero_columns[1:]:
        d += pd.get_dummies(full_df[c])
    full_df = pd.concat([full_df, d.add_prefix(f'{team}_hero_')], axis=1)
    full_df.drop(columns=hero_columns, inplace=True)
    
train_df = full_df.iloc[:train_size, :]
test_df = full_df.iloc[train_size:, :]


# Let's check correlation of new dummy features with target:

# In[7]:


train_df.corrwith(y_train).abs().sort_values(ascending=False).head(12)


# It can be seen from the above output that heroes 32, 22, 19, 91 and 92 play an important role for winning for both teams.
# Let's look up their names in `train_matches.jsonl`:

# In[ ]:


get_ipython().system('grep -oEm1 \'"hero_id":32,"hero_name":"[^"]+?"\' $PATH_TO_DATA/train_matches.jsonl')
get_ipython().system('grep -oEm1 \'"hero_id":22,"hero_name":"[^"]+?"\' $PATH_TO_DATA/train_matches.jsonl')
get_ipython().system('grep -oEm1 \'"hero_id":19,"hero_name":"[^"]+?"\' $PATH_TO_DATA/train_matches.jsonl')
get_ipython().system('grep -oEm1 \'"hero_id":91,"hero_name":"[^"]+?"\' $PATH_TO_DATA/train_matches.jsonl')
get_ipython().system('grep -oEm1 \'"hero_id":92,"hero_name":"[^"]+?"\' $PATH_TO_DATA/train_matches.jsonl')


# Dota players, do you have something to say about the above heroes?
# Probably combining ids of these heroes with some other feature can give a boost? Please share your thoughts in comments.
# 
# Since the features now correlate better with target, this approach should increase the score:

# In[ ]:


classifier = evaluate()


# If we submit this, we get 0.59017

# In[ ]:


submission_df = pd.read_csv(PATH_TO_DATA / 'sample_submission.csv', index_col='match_id_hash')
submission_df['radiant_win_prob'] = classifier.predict_proba(test_df)[:, 1]
submission_df.to_csv('submission.csv')


# Do you know a better approach? Please share in comments.
# And good luck with feature engineering!
