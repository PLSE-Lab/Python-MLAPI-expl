#!/usr/bin/env python
# coding: utf-8

# ## More DATA for DOTA
# 
# Every player knows that game balance is very important, and for wordlwide recognized cybersport discipline, like Dota, balance is crucial. In Dota balance is achived by symmetry. Both teams pick their heroes from the same pool and the map is almost symmetric. Good player should play equally well for both sides. To test this hypothesis we can plot target variable distribution.
# 
# If this assumption is true, we can effectively double our train dataset simply by flipping the teams.

# In[7]:


import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

PATH_TO_DATA = '../input/'


# Let's use the functions from Yury's kernel (https://www.kaggle.com/kashnitsky/dota-2-win-prediction-random-forest-starter) to read data from raw JSON files and slightly modify them.

# In[3]:


import collections

MATCH_FEATURES = [
    ('game_time', lambda m: m['game_time']),
    ('game_mode', lambda m: m['game_mode']),
    ('lobby_type', lambda m: m['lobby_type']),
    ('objectives_len', lambda m: len(m['objectives'])),
    ('chat_len', lambda m: len(m['chat'])),
]


PLAYER_FIELDS = [
    'hero_id',
    
    'kills',
    'deaths',
    'assists',
    'denies',
    
    'gold',
    'lh',
    'xp',
    'health',
    'max_health',
    'max_mana',
    'level',

    'x',
    'y',
    
    'stuns',
    'creeps_stacked',
    'camps_stacked',
    'rune_pickups',
    'firstblood_claimed',
    'teamfight_participation',
    'towers_killed',
    'roshans_killed',
    'obs_placed',
    'sen_placed',
]

def sum_features(feature_log):
    sum = 0
    for key in feature_log:
        sum += feature_log[key]
    return sum

def extract_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash']),
    ]
    
    for field, f in MATCH_FEATURES:
        row.append((field, f(match)))
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'r%d' % (slot + 1)
        else:
            player_name = 'd%d' % (slot - 4)

        for field in PLAYER_FIELDS:
            column_name = '%s_%s' % (player_name, field)
            row.append((column_name, player[field]))

    # Counting ruined towers for both teams
    radiant_tower_kills = 0
    dire_tower_kills = 0
    for objective in match['objectives']:
        if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
            if objective['team'] == 2:
                radiant_tower_kills += 1
            if objective['team'] == 3:
                dire_tower_kills += 1

    # Write new features
    row.append(('radiant_tower_kills', radiant_tower_kills))
    row.append(('dire_tower_kills', dire_tower_kills))
    row.append(('diff_tower_kills', radiant_tower_kills - dire_tower_kills))
    
    return collections.OrderedDict(row)
    
def extract_targets_csv(match, targets):
    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [
        (field, targets[field])
        for field in ['radiant_win']
    ])


# 
# We need to modify this two functions to flip the teams over.

# In[4]:


def extract_inverse_features_csv(match):
    row = [
        ('match_id_hash', match['match_id_hash'][::-1]),
    ]
    
    for field, f in MATCH_FEATURES:
        row.append((field, f(match)))
        
    for slot, player in enumerate(match['players']):
        if slot < 5:
            player_name = 'd%d' % (slot + 1)
        else:
            player_name = 'r%d' % (slot - 4)

        for field in PLAYER_FIELDS:
            column_name = '%s_%s' % (player_name, field)
            row.append((column_name, player[field]))

    # Counting ruined towers for both teams
    radiant_tower_kills = 0
    dire_tower_kills = 0
    for objective in match['objectives']:
        if objective['type'] == 'CHAT_MESSAGE_TOWER_KILL':
            if objective['team'] == 3:
                radiant_tower_kills += 1
            if objective['team'] == 2:
                dire_tower_kills += 1

    # Write new features
    row.append(('radiant_tower_kills', radiant_tower_kills))
    row.append(('dire_tower_kills', dire_tower_kills))
    row.append(('diff_tower_kills', radiant_tower_kills - dire_tower_kills))
    
    return collections.OrderedDict(row)

def extract_inverse_targets_csv(match, targets):
    return collections.OrderedDict([('match_id_hash', match['match_id_hash'][::-1])] + [
        (field,  not targets[field])
        for field in ['radiant_win']
    ])


# In[5]:


import os
import pandas as pd
import numpy as np


import os

try:
    import ujson as json
except ModuleNotFoundError:
    import json
    print ('Please install ujson to read JSON oblects faster')
    
try:
    from tqdm import tqdm_notebook
except ModuleNotFoundError:
    tqdm_notebook = lambda x: x
    print ('Please install tqdm to track progress with Python loops')

def read_matches(matches_file):
    
    MATCHES_COUNT = {
        'test_matches.jsonl': 10000,
        'train_matches.jsonl': 39675,
    }
    _, filename = os.path.split(matches_file)
    total_matches = MATCHES_COUNT.get(filename)
    
    with open(matches_file) as fin:
        for line in tqdm_notebook(fin, total=total_matches):
            yield json.loads(line)


# Read the JSON files and extract features.

# In[8]:


get_ipython().run_cell_magic('time', '', "\ndf_features = []\ndf_targets = []\n\ndf_inverse_features = []\ndf_inverse_targets = []\n\nfor match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):\n    features = extract_features_csv(match)\n    inverse_features = extract_inverse_features_csv(match)\n    targets = extract_targets_csv(match, match['targets'])\n    inverse_targets = extract_inverse_targets_csv(match, match['targets'])\n    \n    df_features.append(features)\n    df_inverse_features.append(features)\n    df_inverse_features.append(inverse_features)\n    df_targets.append(targets)\n    df_inverse_targets.append(targets)\n    df_inverse_targets.append(inverse_targets)")


# Build dataframe from extracted data. If everyhing is correct, this dataframe should be identical to the one used in Yury's kernel (https://www.kaggle.com/kashnitsky/dota-2-win-prediction-random-forest-starter).

# In[ ]:


train_df = pd.DataFrame.from_records(df_features).set_index('match_id_hash')
y_train = pd.DataFrame.from_records(df_targets).set_index('match_id_hash')
y_train = y_train['radiant_win'].map({True: 1, False: 0})


# Now we can test our hypothesis that the side the team played for ('radiant' or 'dire') is not important. Plot the histogram and see, that the 'radiant' and 'dire' wins count is quite close.

# In[ ]:


y_train.hist()


# Now we are ready to build our augmented training dataset.

# In[ ]:


train_df_inverse = pd.DataFrame.from_records(df_inverse_features).set_index('match_id_hash')
y_train_inverse = pd.DataFrame.from_records(df_inverse_targets).set_index('match_id_hash')
y_train_inverse = y_train_inverse['radiant_win'].map({True: 1, False: 0})


# Check the shape of the arrays.

# In[ ]:


print(train_df.shape)
print(train_df_inverse.shape)


# In[ ]:


train_df.head()


# In[ ]:


train_df_inverse.head()


# Train simple LogisticRegression model to see if we got any improvement.

# In[ ]:


logit_pipe = Pipeline([('scaler', MinMaxScaler(feature_range=(0, 1))),
                       ('logit', LogisticRegression(C=0.5, random_state=17, solver='liblinear'))])

logit_res = cross_val_score(logit_pipe, train_df, y_train, scoring='roc_auc', cv = 5, n_jobs=6)
logit_res.mean()


# In[ ]:


logit_pipe = Pipeline([('scaler', MinMaxScaler(feature_range=(0, 1))),
                       ('logit', LogisticRegression(C=0.5, random_state=17, solver='liblinear'))])

logit_res = cross_val_score(logit_pipe, train_df_inverse, y_train_inverse, scoring='roc_auc', cv = 5, n_jobs=6)
logit_res.mean()


# ### Profit!

# In[ ]:




