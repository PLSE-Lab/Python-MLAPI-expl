#!/usr/bin/env python
# coding: utf-8

# # <center> Dota 2 winner prediction competition
# 
# ##### <center> By Artur Kolishenko (@payonear)
#      
# <center> <img src='https://whyigame.files.wordpress.com/2015/05/dota2header.jpg?w=1163&h=364&crop=1.jpeg'>
# 

# ## Data description
# 
# We have the following files:
# 
# - `train_extracted.pkl`, `train_extracted.pkl`: features extracted from json files (details in [
# Data extraction from json- additional features](https://www.kaggle.com/karthur10/data-extraction-from-json-additional-features))
# - `targets.pkl`: results of training games (including the winner)

# In[ ]:


import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")
from sklearn import metrics as sm


# In[ ]:


PATH_TO_DATA = Path('/kaggle/input/data-extraction-from-json-additional-features')


# In[ ]:


train_df = pd.read_pickle(PATH_TO_DATA/'train_extracted.pkl')
test_df = pd.read_pickle(PATH_TO_DATA/'test_extracted.pkl')
target = pd.read_pickle(PATH_TO_DATA/'target.pkl')


# In[ ]:


train_df.head(3)


# In[ ]:


target.head(3)


# ## Exploratory Data Analysis
# Let's look at our data to understand it and generate some ideas for future model.

# In[ ]:


f'Train set\'s shape is {train_df.shape}, of the test set is {test_df.shape} and targer set\'s shape is {target.shape}.'


# In[ ]:


# There are no missed values for the train set
for i in train_df.columns:
    if train_df[i].isnull().sum() > 0:
        print(i, train_df[i].isnull().sum())


# In[ ]:


# So as for the test set
for i in test_df.columns:
    if test_df[i].isnull().sum() > 0:
        print(i, test_df[i].isnull().sum())


# In[ ]:


full_df = pd.concat([train_df, test_df], sort=False)
full_df.shape


# In[ ]:


if all(train_df.columns == test_df.columns):
    print('Train and test features are identical')

if len(full_df.index.unique()) == len(full_df.index):
    print('There is no repeating games in the train and test datasets.')


#  [Gini coefficient](https://en.wikipedia.org/wiki/Gini_coefficient) is a good tool to see how good is the feature in terms of class separation. Often used in Credit Scoring to analyse the power of model for default and non-default accounts discrimination. Coefficient is defined below and used for analysis.

# In[ ]:


def gini(fpr, tpr):
    """
    Function calculates Gini coefficient.
    fpr - the vector of class labels;
    tpr - the vector of feature(s) values.
    """
    return -(2 * sm.roc_auc_score(fpr, tpr) - 1)


# Let's calculate gini for all features before preprocessing.

# In[ ]:


gini_df = {}
for i in [x for x in list(train_df.columns) if x not in list(train_df.filter(like = 'item').columns)]:
    gini_df[i] = gini(target['radiant_win'].values, train_df[i])
gini_df = pd.DataFrame.from_dict(gini_df, orient = 'index',columns = ['gini'])
gini_df['gini_abs'] = abs(gini_df['gini'])
gini_df = gini_df.sort_values('gini_abs', ascending = False)


# In[ ]:


# top 10, baracks_kills and tower_kills are promising
gini_df.head(10)


# In[ ]:


# 10 loosers
gini_df.tail(10).index


# In[ ]:


# Let's plot our data.
import seaborn as sns
from matplotlib import pyplot as plt


# In[ ]:


#Just a slight disbalance
sns.countplot(target['radiant_win'])
plt.title('Result distribution')
plt.show();


# There are 5 non-player features:
# * game_time;
# * game_mode;
# * lobby_type;
# * objectives_len;
# * chat_len.

# In[ ]:


# Seems pretty the same for both samples.
fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
sns.distplot(train_df['game_time'], ax = ax[0][0])
sns.distplot(test_df['game_time'], ax = ax[0][1])
ax[0][0].set_title('Train')
ax[0][1].set_title('Test')
plt.show();


# In[ ]:


sns.distplot(target.duration)
plt.title('Duration')
plt.show();


# In[ ]:


train_avg_time = round(train_df['game_time'].mean(),2)
test_avg_time = round(test_df['game_time'].mean(),2)
train_std_time = round(train_df['game_time'].std(),2)
test_std_time = round(test_df['game_time'].std(),2)
print(f'Average game_time for train - {train_avg_time}, test - {test_avg_time}.')
print(f'Standard deviation for train = {train_std_time}, for test = {test_std_time}.')


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
sns.distplot(train_df[target['radiant_win'] == 1]['game_time'], hist = False, label = 'R_WIN',ax = ax[0][0])
sns.distplot(train_df[target['radiant_win'] == 0]['game_time'], hist = False, label = 'D_WIN', ax = ax[0][0])
sns.distplot(target[target['radiant_win'] == 1]['duration'], hist = False, label = 'R_WIN', ax = ax[0][1])
sns.distplot(target[target['radiant_win'] == 0]['duration'], hist = False, label = 'D_WIN', ax = ax[0][1])
plt.legend()
plt.show();


# Seems like Dire's team more often wins in longer games. Notice, the game starts before 0. So, I assume 0 is the time of first creaps wave. Try to additionally analyse game_time yourself before using it in your model, there are obvious mistakes in data which can spoil it. 

# In[ ]:


# Quite a low value of index
g_game_time = gini(target['radiant_win'].values, train_df['game_time'])
g_duration = gini(target['radiant_win'].values, target['duration'])
print(f'Gini for game_time  = {g_game_time}')
print(f'Gini for duration  = {g_duration}') 


# In[ ]:


# What about game_mode?
fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
a = sns.countplot(train_df['game_mode'], ax = ax[0][0])
for p in a.patches:
    a.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
b = sns.countplot(test_df['game_mode'], ax = ax[0][1])
for p in b.patches:
    b.annotate(format(p.get_height()), (p.get_x() + p.get_width() / 2., p.get_height()),                ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points')
ax[0][0].set_title('Train')
ax[0][1].set_title('Test')
plt.show();


# Worth noticing, there is no 16th game_mode in test sample. Consider dropping. 22nd game_mode is the most popular. Some mergings of game_modes are possible.

# In[ ]:


fig = plt.figure(figsize=(8, 5)) 
sns.countplot(train_df['game_mode'], hue = target['radiant_win'])
plt.show();


# In[ ]:


pd.crosstab(train_df['game_mode'], target['radiant_win'], normalize = 'index').sort_values(1)


# Less numerous game_modes have lower radiant_win rate on average. May be additional reason for game_mode mergings.

# In[ ]:


pd.crosstab(train_df['game_mode'], columns = target['radiant_win'], values = target['duration'], aggfunc = 'mean')


# In[ ]:


pd.crosstab(train_df['game_mode'], columns = target['radiant_win'], values = target['game_time'], aggfunc = 'mean')


# Obviously, Dires win longer games on average. Btw, notice, 23rd game_mode is the shortest game_mode.

# In[ ]:


train_df['lobby_type'].value_counts(normalize = True)


# In[ ]:


test_df['lobby_type'].value_counts(normalize = True)


# In[ ]:


# Radiant_win rate by lobby_type distribution in game_time
fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
sns.distplot(train_df[train_df['lobby_type'] == 0]['game_time'], hist = False, label = '0', ax = ax[0][0])
sns.distplot(train_df[train_df['lobby_type'] == 7]['game_time'], hist = False, label = '7', ax = ax[0][0])
sns.distplot(test_df[test_df['lobby_type'] == 0]['game_time'], hist = False, label = '0', ax = ax[0][1])
sns.distplot(test_df[test_df['lobby_type'] == 7]['game_time'], hist = False, label = '7', ax = ax[0][1])
ax[0][0].set_title('Train')
ax[0][1].set_title('Test')
plt.legend()
plt.show();


# Lobby_types have different time distribution, but still the effect on the result is not obvious. As a result low gini value.

# In[ ]:


# Objective_len
fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
a = sns.countplot(train_df['objectives_len'], ax = ax[0][0])
a.set_xticklabels(a.get_xticklabels(),rotation=90, horizontalalignment='right')
b = sns.countplot(test_df['objectives_len'], ax = ax[0][1])
b.set_xticklabels(b.get_xticklabels(),rotation=90, horizontalalignment='right')
ax[0][0].set_title('Train')
ax[0][1].set_title('Test')
plt.show();


# Objectives_len most often equals 1. Interesting to see what is this objective. Assume this is 'First_blood' message as this message is first almost all the time. First_blood may be useful in predicting. Let's check whether objectives_len correlates with game_time.

# In[ ]:


train_df[train_df.iloc[:,:5].columns].corr()


# Obviously, it does. So, the message here is that objectives_len shows almost the same as game_time. We may check the point in time when the first_blood occured. Earlier first_blood may cause more agressive game, maybe it has impact on the result. Interesting note, chat_len doesn't correlate much with game_time. Which means, that chatting is not quite regular in the game.

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
a = sns.scatterplot(train_df['chat_len'], train_df['chat_len'], ax = ax[0][0])
b = sns.scatterplot(test_df['chat_len'], test_df['chat_len'], ax = ax[0][1])
ax[0][0].set_title('Train')
ax[0][1].set_title('Test')
plt.show();


# Chats are quite long. So, players communicate much. We have few ouliers. I guess, chat_len as a feature is rather noisy here. Just a voise of a common sense. The feature doesn't show us even which team communicates more. That's why in [Data extraction from json- additional features](https://www.kaggle.com/karthur10/data-extraction-from-json-additional-features) new features for chats were created. Let's check them out.

# In[ ]:


train_df.filter(like = 'chat').columns


# In[ ]:


g_chat_len = gini(target['radiant_win'].values, train_df['chat_len'].values)
g_radiant_chat_len = gini(target['radiant_win'].values, train_df['radiant_chat_len'].values)
g_dire_chat_len = gini(target['radiant_win'].values, train_df['dire_chat_len'].values)
g_diff_chat_len = gini(target['radiant_win'].values, train_df['diff_chat_len'].values)
g_radiant_chat_memb = gini(target['radiant_win'].values, train_df['radiant_chat_memb'].values)
g_dire_chat_memb = gini(target['radiant_win'].values, train_df['dire_chat_memb'].values)
g_diff_chat_memb = gini(target['radiant_win'].values, train_df['diff_chat_memb'].values)
print(f'Gini for chat_len = {g_chat_len}, radiant_chat_len = {g_radiant_chat_len}, dire_chat_len = {g_dire_chat_len}, diff_chat_len = {g_diff_chat_len}')
print(f'radiant_chat_memb = {g_radiant_chat_memb}, dire_chat_memb = {g_dire_chat_memb}, diff_chat_memb = {g_diff_chat_memb}')


# Not a huge difference, but 'diff_chat_len' is better feature than simle 'chat_len'. Anyway gini coefficient value is rather low. Consider dropping.

# ### <center> Players' features

# Let's analyse player's features. It's very important to understand, that the sequence of players plays no role. Which means doesn't matter whether the player with the same statistics is r_1 or r_2 etc. So we should consider methods of feature aggregations for the whole team. The very first feature is hero_id. Let's create dummies for hero_ids and analyse what kind of heroes we have.

# In[ ]:


train_size = train_df.shape[0]
hero_columns = [c for c in full_df.columns if '_hero_' in c]
train_hero_id = train_df[hero_columns]
train_hero_id.head(3)


# In[ ]:


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


# In[ ]:


if all(train_df.filter(like = 'hero').columns == test_df.filter(like = 'hero').columns):
    print('hero_ids in the train sample are the same as in the test sample.')


#  Let's see the correlation of each hero_id with radiant_win variable. Seems, that heroes with id: 32,22,19, 91, 92 have quite a high correlation with the result. Maybe these heroes are strong?

# In[ ]:


train_df[train_df.filter(like = 'hero').columns].corrwith(target.radiant_win).abs().sort_values(ascending=False).head(12)


# 
# What about the weakest ones?

# In[ ]:


train_df[train_df.filter(like = 'hero').columns].corrwith(target.radiant_win).abs().sort_values(ascending=False).tail(12)


# In[ ]:


heroes = pd.DataFrame(train_df[train_df.filter(like = 'hero').columns].sum().sort_values(ascending = False)
    , columns = ['Train']).merge(pd.DataFrame(test_df[test_df.filter(like = 'hero').columns].sum()
    , columns = ['Test']), left_index = True, right_index = True)
heroes['train_occ'] = round(heroes['Train']/train_df.shape[0]*100,2)
heroes['test_occ'] = round(heroes['Test']/test_df.shape[0]*100,2)
heroes.head(12)


# As can be seen, heroes 14,11,32,8,74,35 etc. are very popular amongst players. 14 is chosen in more than 40% of games (hero may be chosen just by one team in the same game). Obviously, train and test samples have almost the same hero_id distribution and that's a good news. But, worth noticing, that there is a huge difference in hero_id occurances. 66th hero_id barely appears just in 1% of the games. So, maybe could be useful to use not every hero-id dummy.

# In[ ]:


# Kinda good news
heroes[['train_occ', 'test_occ']].corr()


# In[ ]:


# What about our most succesful hero_ids?
heroes.loc[['d_hero_32','r_hero_32','r_hero_22','r_hero_19',
      'd_hero_22','d_hero_19','d_hero_92','d_hero_91',
      'd_hero_73','r_hero_92','r_hero_91']]


# Look at other player's features. As in case of hero_ids there is no sense in players' sequence. We should consider some ways to avoid 'r1', 'r2' etc. I propose to calculate statistics like max, mean, min, std per team and feature. First of all it'll help to avoid the sequence issue, secondly, reduce dimensionality.

# In[ ]:


train_df.filter(like = 'r1').columns


# In[ ]:


#let's make our life simplier
def combine_numeric_features (df, feature_suffixes):
    for feat_suff in feature_suffixes:
        for team in 'r', 'd':
            players = [f'{team}{i}' for i in range(1, 6)] # r1, r2...
            player_col_names = [f'{player}_{feat_suff}' for player in players] # e.g. r1_gold, r2_gold
            
            df[f'{team}_{feat_suff}_max'] = df[player_col_names].max(axis=1) # e.g. r_gold_max
            df[f'{team}_{feat_suff}_mean'] = df[player_col_names].mean(axis=1) # e.g. r_gold_mean
            df[f'{team}_{feat_suff}_min'] = df[player_col_names].min(axis=1) # e.g. r_gold_min
            df[f'{team}_{feat_suff}_sum'] = df[player_col_names].sum(axis=1)
            df[f'{team}_{feat_suff}_std'] = df[player_col_names].std(axis=1)
            
            df.drop(columns=player_col_names, inplace=True) # remove raw features from the dataset
            
    return df


# In[ ]:


numeric_features = ['kills', 'deaths', 'assists', 'denies', 'gold', 'xp', 'health', 
                    'max_health', 'max_mana', 'level', 'towers_killed', 'stuns', 'creeps_stacked', 
                    'camps_stacked', 'lh', 'rune_pickups', 'firstblood_claimed', 'teamfight_participation', 
                    'roshans_killed', 'obs_placed', 'sen_placed', 'dam_diff', 'ability_upgrades']


# In[ ]:


train_df = combine_numeric_features(train_df, numeric_features)
test_df = combine_numeric_features(test_df, numeric_features)


# In[ ]:


train_df.head(3)


# Be careful, many features are highly correlated. That's important especially in case of linear models. Now, I propose to look at coordinate features, as they've already shown high gini coef values.

# In[ ]:


# Creating vectors of x and y
x_values = []
y_values = []
for team in 'r','d':
    players = [f'{team}{i}' for i in range(1, 6)]
    for i in players:
        x_values += list(train_df[f'{i}_x'])
        y_values += list(train_df[f'{i}_y'])
coord_df = pd.DataFrame(x_values, columns = ['x'])
coord_df['y'] = y_values
coord_df['radiant_win'] = list(target['radiant_win'])*10
coord = pd.pivot_table(data = coord_df, index = 'y', columns = 'x', values = 'radiant_win', aggfunc = 'mean').fillna(0)


# In[ ]:


fig, ax = plt.subplots(figsize=(20,10))
sns.heatmap(coord.sort_index(ascending = False), ax = ax);


# As can be seen, as nearer players to upper right corner, as bigger probability of radiant_win. Vice versa, if players near lower left corner (radiant base), radiant_win prob is lower. But how to aggregate coordiantes? For a good start - [Combine hero features into team ones - basic](https://www.kaggle.com/daemonis/combine-hero-features-into-team-ones-basic).

# In[ ]:


def make_coordinate_features(df):
    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)] # r1, r2...
        for player in players:
            df[f'{player}_distance'] = np.sqrt(df[f'{player}_x']**2 + df[f'{player}_y']**2)
            df.drop(columns=[f'{player}_x', f'{player}_y'], inplace=True)
    return df


# In[ ]:


train_df = make_coordinate_features(train_df)
test_df = make_coordinate_features(test_df)


# In[ ]:


train_df = combine_numeric_features(train_df, ['distance'])
test_df = combine_numeric_features(test_df, ['distance'])


# In[ ]:


# As expected baracks_kills feature is quite strong
baracks = pd.crosstab(train_df['diff_baracks_kills'],target['radiant_win'], normalize = 'index')
sns.lineplot(y = baracks.index, x=baracks[1]);


# In[ ]:


# But baracks are killed mostly in the end of the game, so often diff_baracks_kills = 0
train_df['diff_baracks_kills'].value_counts()


# In[ ]:


towers = pd.crosstab(train_df['diff_tower_kills'],target['radiant_win'], normalize = 'index')
sns.lineplot(y = towers.index, x=towers[1]);


# In[ ]:


# A bit better with towers
train_df['diff_tower_kills'].value_counts()


# In[ ]:


aegis = pd.crosstab(train_df['diff_aegis'],target['radiant_win'], normalize = 'index')
sns.lineplot(y = aegis.index, x=aegis[1]);


# Quite a strange dependance from diff_aegis feature 

# Let's try to analyse inventories. My high recommendation [Hero_Items Guide](https://www.kaggle.com/grazder/hero-items-guide).

# In[ ]:


def add_items_dummies(df_train, df_test):
    
    full_df = pd.concat([df_train, df_test], sort=False)
    train_size = df_train.shape[0]

    for team in 'r', 'd':
        players = [f'{team}{i}' for i in range(1, 6)]
        item_columns = [f'{player}_items' for player in players]

        d = pd.get_dummies(full_df[item_columns[0]].apply(pd.Series).stack()).sum(level=0, axis=0)
        dindexes = d.index.values

        for c in item_columns[1:]:
            d = d.add(pd.get_dummies(full_df[c].apply(pd.Series).stack()).sum(level=0, axis=0), fill_value=0)
            d = d.ix[dindexes]

        full_df = pd.concat([full_df, d.add_prefix(f'{team}_item_')], axis=1, sort=False)
        full_df.drop(columns=item_columns, inplace=True)

    df_train = full_df.iloc[:train_size, :]
    df_test = full_df.iloc[train_size:, :]

    return df_train, df_test


# In[ ]:


def drop_consumble_items(df_train, df_test):
    
    full_df = pd.concat([df_train, df_test], sort=False)
    train_size = df_train.shape[0]

    for team in 'r', 'd':
        consumble_columns = ['tango', 'tpscroll', 
                             'bottle', 'flask',
                            'enchanted_mango', 'clarity',
                            'faerie_fire', 'ward_observer',
                            'ward_sentry']
        
        starts_with = f'{team}_item_'
        consumble_columns = [starts_with + column for column in consumble_columns]
        full_df.drop(columns=consumble_columns, inplace=True)

    df_train = full_df.iloc[:train_size, :]
    df_test = full_df.iloc[train_size:, :]

    return df_train, df_test


# In[ ]:


train_df, test_df = add_items_dummies(train_df, test_df)
train_df, test_df = drop_consumble_items(train_df, test_df)


# In[ ]:


train_df.columns


# In[ ]:


train_df[train_df.filter(like = 'item').columns].corrwith(target.radiant_win).abs().sort_values(ascending=False).head(20)


# Yet, seems like aegis is very important. What's interesting is that by first sight different items are important for different teams. Let's now look at our gini.

# In[ ]:


gini_df = {}
for i in [x for x in list(train_df.columns) if x not in list(train_df.filter(like = 'item').columns)]:
    gini_df[i] = gini(target['radiant_win'].values, train_df[i])
gini_df = pd.DataFrame.from_dict(gini_df, orient = 'index',columns = ['gini'])
gini_df['gini_abs'] = abs(gini_df['gini'])
gini_df = gini_df.sort_values('gini_abs', ascending = False)


# In[ ]:


gini_df.head(40)


# Oh, you reached the end of the kernel. Hope it was useful. Please don't forget to upvote.
