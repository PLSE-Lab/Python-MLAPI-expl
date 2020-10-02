#!/usr/bin/env python
# coding: utf-8

# ## In this kernel i will describe some features from json files and some ideas how use this features.

# In[2]:


import os
import json

import pandas as pd

PATH_TO_DATA = '../input/'


# In[3]:


df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_features.csv'), index_col='match_id_hash')

df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 'train_targets.csv'), index_col='match_id_hash')
df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), index_col='match_id_hash')


# In[ ]:


df_train_features.head(2)


# Read some data from json

# In[ ]:


json_list = []
number_of_rows = 50 # Number of readed json rows

with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as fin:
    for i in range(number_of_rows):
        line = fin.readline()
        json_list.append(json.loads(line))


# In[ ]:


json_list[0].keys()


# We can see features, which exists in json and dataframe (**'game_time', 'game_mode', 'lobby_type', 'match_id_hash', 'targets'**).
# 
# Let investigate other features.
# 
# I'm using different index in json list, because some features are empty for different matches.
# 

# ## Chat feature

# In[ ]:


json_list[0]['chat']


# Chat is feature which show only players chat.
# 
# **Players slot** means (0, 1, 2, 3, 4) for radiant and (128, 129, 130, 131, 132) for dire.
# 
# **Time** is current time of message. Negative time is time(i'm not sure), when gamers connected to the game and prepearing for match(players in game, but wave of creeps didn't start).
# 
# **Text** feature is encoded and we can't extract some useful information.
# 
# <br>
# *Some ideas of extracting features:*
# 1. If team have a lot of messages they discussing a tactics.
# 
# 2. If team have a lot of messages they not focused on the game.
# 
# 

# ## Objectives feature

# In[ ]:


json_list[1]['objectives']


# Objectives is feature which shows global activity of team.
# 
# **Type** is a type of objectives(tower kill, firsblood, roshan killed and etc).
# 
# **Time** is a time when it happends.
# 
# **Team** is a team(2 is radiant, 3 is dire).
# 
# **Slot and key** i can't understand.
# 
# 
# <br>
# *Some ideas of extracting features:*
# 1. How many roshans killed by the team.
# 
# 2. How fast team destroy towers of opponent (There are exists some strategy, called "push-strategy". The idea of this strategy is destroy a oppent as fast as possible).
# 
# 
# 
# 

# # Teamfights feature

# In[ ]:


print(json.dumps(json_list[1]['teamfights'][1], indent=4, sort_keys=True))


# Teamfights is feature which show information about teamfights(in teamfights participate all players)
# 
# | Feature | Description |
# | ------------- |:-------------| 
# | **Start** | Start time of fights.|
# |**End** | End time of fights.|
# |**last_death**| Time of last death in fights(after last death time players can damage each other, but nobody die).|
# |**deaths**| Number of deaths.|
# |**Players feature**| |
# |**ability_uses**| Name of ability which used by hero and number of using.|
# |**buybacks**| Buyback means that hero can return to game after die immediately, but need to pay gold. (0 not used, 1 - used) |
# |**damage**| Number of damage deal by this player.|
# |**gold_delta**| How many gold receive player in teamfight.|
# |**healing**| Number of restored health.|
# |**item_uses**| Which item used in teamfights.|
# |**killed**| Which heroes kill current hero and how many damage take in last moment.|
# |**xp_delta**| How many xp receive player in teamfight.|
# 
# 
# 
# 
# 

# *Some ideas of extracting features:*
# 1. From one teamfight we can extract info about who winning teamfight(by xp and gold delta). From all teamfights we can see dynamics of teamfights (Maybe some team made comeback?)).

# ## Players feature

# In[ ]:


print(json.dumps(json_list[1]['players'][1], indent=4, sort_keys=True))


# Players is feature which show information about every player
# 
# | Feature | Description |
# | ------------- |:-------------| 
# | **ability_upgrades** | |
# |**ability** | Id of ability |
# |**level** | By which level ability was upgrade |
# |**ability** | Time of upgrade ability |
# | **ability_uses** | Name of ability which used by hero and number of using.|
# |**account_id_hash** | Hash of player account |
# |**damage** | How much damage dealt for which heroes/creeps|
# |**damage_taken** | How much damage taken for which heroes/creeps|
# |**damage_inflictor** | How much damage from skills dealt and for which heroes/creeps|
# |**damage_inflictor_received** | How much damage from skills taken|
# |**death** | Number of death |
# |**denies** | Number of denies of creeps |
# |**firsblood_claimed** | Hero made firstblood?(0 or 1)|
# |**healing** | Number of restored health|
# |**hero_id** | Id of hero|
# |**hero_inventory** | Current items of hero|
# |**hero_name** | Name of hero|
# |**hero_stash** | Stash of hero(Item, which hero buy, but not brought herself|
# |**item_uses** | Item and number of it uses|
# |**kill_streaks** | Type and number of kill streaks|
# |**multi_kills** | Type and number of multi kills|
# |**pred_vict** | Predict hero his victory(true/false). At the beginning of the match players can bet on his winning(Only on win. False means hero not bet)|
# |**purchase** | All items which bought by player|
# |**randomed** | Picked hero by random?(Player can choose hero random)|
# |**purchase** | All items which bought by player|
# |**stuns** | Number of seconds when hero stuns(disable) enemies|
# 
# 
# 
# 
# 

# I think that this feature has a lot of useful information and this feature need more investigation.
# 
# *Some ideas of extracting features:*
# 1.  Number of pred_vict in a team (Maybe team sure in his victory).
# 2.  OHE of hero_inventory for team (Maybe exists some items which has high rate of winrate?)
# 3.  Total damage by hero (damage feature)
# 4.  account_id_hash (Maybe exists players which has high rate of winrate?)

# This is just an ideas, and I didn't researched them all. It also takes a lot of time to read the full JSON and create a new features.
# 

# ## Example with total damage feature

# In[4]:


y = df_train_targets['radiant_win'].values


# In[5]:


try:
    from tqdm import tqdm_notebook
except ModuleNotFoundError:
    tqdm_notebook = lambda x: x
    print ('Please install tqdm to track progress with Python loops')


# In[6]:


#a helper function, we will use it in next cell
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


# In[8]:


def add_new_features(df_features, matches_file):
    
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']

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
        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills
        
        


# In[9]:


get_ipython().run_cell_magic('time', '', "# copy the dataframe with features\ndf_train_features_extended = df_train_features.copy()\n\n# add new features\nadd_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))")


# ### Use simple lightgbm

# In[16]:


from sklearn.model_selection import cross_val_score
from lightgbm import LGBMClassifier


# In[19]:


lgb_classifier = LGBMClassifier(n_estimators=200)


# In[22]:


cv_default = cross_val_score(lgb_classifier, df_train_features_extended, y, cv=5)
cv_default.mean()


# Add feature total damage by team

# In[33]:


def add_new_features(df_features, matches_file):
    
    # Process raw data and add new features
    for match in read_matches(matches_file):
        match_id_hash = match['match_id_hash']

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
        df_features.loc[match_id_hash, 'radiant_tower_kills'] = radiant_tower_kills
        df_features.loc[match_id_hash, 'dire_tower_kills'] = dire_tower_kills
        df_features.loc[match_id_hash, 'diff_tower_kills'] = radiant_tower_kills - dire_tower_kills
        
        # Total damage
        total_damage = 0
        for i in range(1, 6):
            for j in match['players'][i-1]['damage']:
                # Take damage only to hero(not for creeps)
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'r_damage'] = total_damage
        total_damage = 0
        for i in range(6, 11):
            for j in match['players'][i-1]['damage']:
                if j.startswith('npc_dota_hero'):
                    total_damage += match['players'][i-1]['damage'][j]
        df_features.loc[match_id_hash, 'd_damage'] = total_damage

        df_features.loc[match_id_hash, 'diff_damage'] = df_features.loc[match_id_hash, 'r_damage'] - df_features.loc[match_id_hash, 'd_damage'] 

      


# In[34]:


get_ipython().run_cell_magic('time', '', "# copy the dataframe with features\ndf_train_features_extended = df_train_features.copy()\n\n# add new features\nadd_new_features(df_train_features_extended, os.path.join(PATH_TO_DATA, 'train_matches.jsonl'))")


# In[35]:


cv_extended = cross_val_score(lgb_classifier, df_train_features_extended, y, cv=5)
cv_extended.mean()


# See slight improvement
