#!/usr/bin/env python
# coding: utf-8

# # <center> Dota 2 winner prediction
# 
# <img src='https://habrastorage.org/webt/ua/vn/pq/uavnpqfoih4zwwznvxubu33ispy.jpeg'>
# 
# #### <center> Originally done by Peter Romov, translated and adapted by Yury Kashnitskiy (@yorko)
#     
# ### Quick start
# 
# Grab features prepared by organizers, train a model and submit. 
# 
# 1. [Data description](#Data-description)
# 2. [Features created by organizers](#Features-created-by-organizers)
# 3. [Training and evaluating a model](#Training-and-evaluating-a-model)
# 4. [Preparing a submission](#Preparing-a-submission)
# 
# ### Now do it as a real Data Scientist
# 
# 5. [Cross-validation](#Cross-validation)
# 6. [Working with all available information on Dota games](#Working-with-all-available-information-on-Dota-games)
# 7. [Feature engineering](#Feature-engineering)
# 8. [How to build initial features from scratch](#How-to-build-initial-features-from-scratch)

# ## Data description
# 
# We have the following files:
# 
# - `sample_submission.csv`: example of a submission file
# - `train_matches.jsonl`, `test_matches.jsonl`: full "raw" training data 
# - `train_features.csv`, `test_features.csv`: features created by organizers
# - `train_targets.csv`: results of training games (including the winner)

# ## Features created by organizers
# 
# These are basic features which include simple players' statistics. Scroll to the end to see how to build these features from raw json files.

# In[ ]:


import os
import pandas as pd

PATH_TO_DATA = '../input/'

df_train_features = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                             'train_features.csv'), 
                                    index_col='match_id_hash')
df_train_targets = pd.read_csv(os.path.join(PATH_TO_DATA, 
                                            'train_targets.csv'), 
                                   index_col='match_id_hash')


# We have ~ 40k games, each described by `match_id_hash` (game id) and 245 features. Also `game_time` is given - time (in secs) when the game was over. 

# In[ ]:


df_train_features.shape


# In[ ]:


df_train_features.head()


# We are interested in the `radiant_win` column in `train_targets.csv`. All these features are not known during the game (they come "from future" as compared to `game_time`), so we have these features only for training data. 

# In[ ]:


df_train_targets.head()


# ## Training and evaluating a model

# #### Let's construct a feature matrix `X` and a target vector `y`

# In[ ]:


X = df_train_features.values
y = df_train_targets['radiant_win'].values


# #### Perform  a train/test split (a simple validation scheme)

# In[ ]:


from sklearn.model_selection import train_test_split

X_train, X_valid, y_train, y_valid = train_test_split(X, y, 
                                                      test_size=0.3, 
                                                      random_state=17)


# #### Train the Random Forest model

# In[ ]:


get_ipython().run_cell_magic('time', '', 'from sklearn.ensemble import RandomForestClassifier\nmodel = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)\nmodel.fit(X_train, y_train)')


# #### Make predictions for the holdout set
# 
# We need to predict probabilities of class 1 - that Radiant wins, thus we need index 1 in the matrix returned by the `predict_proba` method.

# In[ ]:


y_pred = model.predict_proba(X_valid)[:, 1]


# Let's take a look:

# In[ ]:


y_pred


# #### Let's evaluate prediction quality with the holdout set
# 
# We'll calculate ROC-AUC.

# In[ ]:


from sklearn.metrics import roc_auc_score

valid_score = roc_auc_score(y_valid, y_pred)
print('Validation ROC-AUC score:', valid_score)


# Out if curiosiry, we can calculate accuracy of a classifier which predicts class 1 if predicted probability is higher than 50%. 

# In[ ]:


from sklearn.metrics import accuracy_score

valid_accuracy = accuracy_score(y_valid, y_pred > 0.5)
print('Validation accuracy of P>0.5 classifier:', valid_accuracy)


# ## Preparing a submission
# 
# Now the same for test data.

# In[ ]:


df_test_features = pd.read_csv(os.path.join(PATH_TO_DATA, 'test_features.csv'), 
                                   index_col='match_id_hash')

X_test = df_test_features.values
y_test_pred = model.predict_proba(X_test)[:, 1]

df_submission = pd.DataFrame({'radiant_win_prob': y_test_pred}, 
                                 index=df_test_features.index)


# In[ ]:


df_submission.head()


# Save the submission file, it's handy to include current datetime in the filename. 

# In[ ]:


import datetime
submission_filename = 'submission_{}.csv'.format(
    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
df_submission.to_csv(submission_filename)
print('Submission saved to {}'.format(submission_filename))


# ## Cross-validation
# 
# As we already know, cross-validation is a more reliable validation technique than just one train/test split. Here we'll resort to `ShuffleSplit` to create 5 70%/30% splits. 

# In[ ]:


from sklearn.model_selection import ShuffleSplit, KFold
cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=17)


# In[ ]:


from sklearn.model_selection import cross_val_score


# #### Run cross-validation
# 
# We'll train 2 versions of the  `RandomForestClassifier` model - first with default capacity (trees are not limited in depth), second - with `min_samples_leaf`=3, i.e. each leave is obliged to have at least 3 instances. 

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nmodel_rf1 = RandomForestClassifier(n_estimators=100, n_jobs=4,\n                                   max_depth=None, random_state=17)\n\n# calcuate ROC-AUC for each split\ncv_scores_rf1 = cross_val_score(model_rf1, X, y, cv=cv, scoring='roc_auc')")


# In[ ]:


get_ipython().run_cell_magic('time', '', "\nmodel_rf2 = RandomForestClassifier(n_estimators=100, n_jobs=4,\n                                   min_samples_leaf=3, random_state=17)\n\ncv_scores_rf2 = cross_val_score(model_rf2, X, y, cv=cv, \n                                scoring='roc_auc', n_jobs=-1)")


# #### CV results
# 
# The result returned by `cross_val_score` is an array with metric values (ROC-AUC) for each split:

# In[ ]:


cv_scores_rf1


# In[ ]:


cv_scores_rf2


# Let's compare average ROC-AUC among all splits for both models.

# In[ ]:


print('Model 1 mean score:', cv_scores_rf1.mean())
print('Model 2 mean score:', cv_scores_rf2.mean())


# The second model is preferred. Look, there's a caveat here: the second model is actually better for 4 splits out of 5. So if we were to perform only one train/test split, there would've been a 20% probability to make a wrong conclusion that the first model is better.

# In[ ]:


cv_scores_rf2 > cv_scores_rf1


# ## Working with all available information on Dota games
# Raw data descriptions for all games are given in files `train_matches.jsonl` and `test_matches.jsonl`. Each file has one entry for each game in [JSON](https://en.wikipedia.org/wiki/JSON) format. You only need to know that it can be easily converted to Python objects via the `json.loads` method.

# ##### Let's explore a single entry

# In[ ]:


import json

with open(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')) as fin:
    # read the 18-th line
    for i in range(18):
        line = fin.readline()
    
    # read JSON into a Python object 
    match = json.loads(line)


# The `match` object is now a big Python dictionary. In `match['players']` we have a description of each player.
# 
# You might think that this `match` object look ugly. You're right! That's actually the real data. And it's the ability to extract nice features from raw data that makes good Data Scientists stand out. You might even be unfamiliar with Dota (or any other application domain) but still be able to construct a good model via feature engineering. It's art and craftmanship at the same time.   

# In[ ]:


#match


# #### Player description

# In[ ]:


player = match['players'][2]


# KDA: the number of kills, deaths, and assists to alleys.

# In[ ]:


player['kills'], player['deaths'], player['assists']


# Some statistics on player abilities:

# In[ ]:


player['ability_uses']


# #### Example: time series for each player's gold.

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt


# In[ ]:


for player in match['players']:
    plt.plot(player['times'], player['gold_t'])
    
plt.title('Gold change for all players');


# #### Function to read files with game descriptions
# 
# The following function `read_matches(filename)`, can be used to read raw data on Dota 2 games.
# 
# We recommend to install two Python packages: `ujson` and `tqdm`, it'll make the execution faster and 

# In[ ]:


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


# #### Reading data in a loop
# 
# Reading data on all games might take some 2-3 minutes. Thus you'd better stick to the following approach:
# 
# 1. Read a small amount (10-100) of games
# 2. Write code to extract features from these JSON objects
# 3. Make sure the code works fine
# 4. Run the code with all available data
# 5. Save results to a `pickle` file so that you don't need to run all computations from scratch next time 

# In[ ]:


for match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):
    match_id_hash = match['match_id_hash']
    game_time = match['game_time']
    
    # processing each game
    
    for player in match['players']:
        pass  # processing each player


# ## Feature engineering

# In[ ]:


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
        
        # ... here you can add more features ...
        


# In[ ]:


# copy the dataframe with features
df_train_features_extended = df_train_features.copy()

# add new features
add_new_features(df_train_features_extended, 
                 os.path.join(PATH_TO_DATA, 
                              'train_matches.jsonl'))


# We see new features added to the right.

# In[ ]:


df_train_features_extended.head()


# #### Evaluating new features
# 
# Let's run cross-validation with a fixed model but with two different datasets:
# 
# 1. with features built by organizers (base)
# 2. with new features that we've added (extended)

# In[ ]:


get_ipython().run_cell_magic('time', '', "\nfrom sklearn.ensemble import RandomForestClassifier\n\nmodel = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)\n\ncv_scores_base = cross_val_score(model, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)\ncv_scores_extended = cross_val_score(model, df_train_features_extended.values, y, \n                                     cv=cv, scoring='roc_auc', n_jobs=-1)")


# In[ ]:


print('Base features: mean={} scores={}'.format(cv_scores_base.mean(), 
                                                cv_scores_base))
print('Extended features: mean={} scores={}'.format(cv_scores_extended.mean(), 
                                                    cv_scores_extended))


# In[ ]:


cv_scores_extended > cv_scores_base


# As we see, `RandomForestClassifier` shows better cross-validation results in case of the extended dataset. Looks reasonable, that's what we build features for.

# #### New submission

# In[ ]:


get_ipython().run_cell_magic('time', '', "# Build the same features for the test set\ndf_test_features_extended = df_test_features.copy()\nadd_new_features(df_test_features_extended, \n                 os.path.join(PATH_TO_DATA, 'test_matches.jsonl'))")


# In[ ]:


model = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)
model.fit(X, y)
df_submission_base = pd.DataFrame(
    {'radiant_win_prob': model.predict_proba(df_test_features.values)[:, 1]}, 
    index=df_test_features.index,
)
df_submission_base.to_csv('submission_base_rf.csv')


# In[ ]:


model_extended = RandomForestClassifier(n_estimators=100, n_jobs=4, random_state=17)
model_extended.fit(df_train_features_extended.values, y)
df_submission_extended = pd.DataFrame(
    {'radiant_win_prob': model_extended.predict_proba(df_test_features_extended.values)[:, 1]}, 
    index=df_test_features.index,
)
df_submission_extended.to_csv('submission_extended_rf.csv')


# In[ ]:


# this one will be used as a final submission in this kernel
get_ipython().system('cp submission_extended_rf.csv submission.csv')


# ## How to build initial features from scratch
# 
# Now we diclose the code that we used to build initial features `train_features.csv` and `test_features.csv`. You can modify the following code to add more features.
# 
# In a nutshell:
# 
# 1. the  `extract_features_csv(match)` function extracts features from game descriptions and writes them into a dictionary
# 2. the `extract_targets_csv(match, targets)` function extracts the target variable `radiant_win`
# 3. iterating through the file with raw data, we collect all features
# 4. with `pandas.DataFrame.from_records()` we create dataframes with new features

# In[ ]:


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
            
    return collections.OrderedDict(row)
    
def extract_targets_csv(match, targets):
    return collections.OrderedDict([('match_id_hash', match['match_id_hash'])] + [
        (field, targets[field])
        for field in ['game_time', 'radiant_win', 'duration', 'time_remaining', 'next_roshan_team']
    ])


# In[ ]:


get_ipython().run_cell_magic('time', '', "\ndf_new_features = []\ndf_new_targets = []\n\nfor match in read_matches(os.path.join(PATH_TO_DATA, 'train_matches.jsonl')):\n    match_id_hash = match['match_id_hash']\n    features = extract_features_csv(match)\n    targets = extract_targets_csv(match, match['targets'])\n    \n    df_new_features.append(features)\n    df_new_targets.append(targets)\n    ")


# In[ ]:


df_new_features = pd.DataFrame.from_records(df_new_features).set_index('match_id_hash')
df_new_targets = pd.DataFrame.from_records(df_new_targets).set_index('match_id_hash')


# In[ ]:


df_new_features.head()


# ## Go on!
# 
# - Discuss new ideas in Slack 
# - Create new features
# - Try new models and ensembles
# - Submit predictions
# - Go and win!
