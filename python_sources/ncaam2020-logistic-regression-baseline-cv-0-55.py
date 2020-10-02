#!/usr/bin/env python
# coding: utf-8

# # NCAAM2020 - Logistic Regression Baseline [CV 0.55]

# In this notebook, I trained a baseline Logistic Regression model using seed data (seed number and seed difference) from tourney games before 2015. Before that, I explored the data a bit, simulated the test set, and prepared a train set (without leakage and has opposite team perspectives).
# 
# References:
# - [Delete Leaked from Training NCAAM/NCAAW - Stage1](https://www.kaggle.com/catadanna/delete-leaked-from-training-ncaam-ncaaw-stage1) by [Catadanna](https://www.kaggle.com/catadanna)
# - [How to score 0 and why you should not do this](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/discussion/131539) by [Roman](https://www.kaggle.com/nroman)
# - [Jump Shot to Conclusions - March Madness EDA](https://www.kaggle.com/headsortails/jump-shot-to-conclusions-march-madness-eda) by [Heads or Tails](https://www.kaggle.com/headsortails)

# ## Import Libraries

# In[ ]:


import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression


# # Explore Data
# - for now we only explore and use Section 1 - The Basics

# In[ ]:


data_dir = '../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/'


# ## Teams

# In[ ]:


team = pd.read_csv(data_dir + 'MTeams.csv')

print(team.shape)
display(team.head(3))
display(team.describe().transpose())


# - __we have TeamIDs from 1101 to 1467__
# - __we have D1Seasons from 1985 to 2020__

# ## Seasons

# In[ ]:


season = pd.read_csv(data_dir + 'MSeasons.csv')

print(season.shape)
display(season.head(3))
display(season.describe().transpose())


# - __we have Seasons from 1985 to 2020__

# In[ ]:


# format date
season['DayZero'] = pd.to_datetime(season['DayZero'])
display(season[['DayZero']].describe())


# ## Seeds

# In[ ]:


seed = pd.read_csv(data_dir + 'MNCAATourneySeeds.csv')

print(seed.shape)
display(seed.head(3))
display(seed.describe().transpose())


# - __not all TeamIDs have seed data (max TeamID 1467 from above)__

# ## Season Results

# In[ ]:


season_result = pd.read_csv(data_dir + 'MRegularSeasonCompactResults.csv')

print(season_result.shape)
display(season_result.head(3))
display(season_result.describe().transpose()[['min', 'max']])


# In[ ]:


print('seasons without results:')

season.loc[~season['Season'].isin(season_result['Season'])]


# - __the current season has no results yet__

# In[ ]:


# add game date (we can use DayZero + DayNum to get the actual GameDate)

def apply_column_offset(df, date_col, offset_col):
    
    # source: https://stackoverflow.com/questions/48210892/pandas-date-difference-using-column-as-offset
    
    df = df.copy()
    df['NewDate'] = df[date_col].values.astype('datetime64[D]') +                     df[offset_col].add(1).values.astype('timedelta64[D]') -                     np.array([1], dtype='timedelta64[D]')
    return df['NewDate']

season_result = pd.merge(season_result,
                         season[['Season', 'DayZero']],
                         on='Season',
                         how='inner')
            
season_result['GameDate'] = apply_column_offset(season_result, 'DayZero', 'DayNum')
season_result.head(3)


# ## Tourney Results

# In[ ]:


tourney_result = pd.read_csv(data_dir + 'MNCAATourneyCompactResults.csv')

print(tourney_result.shape)
display(tourney_result.head(3))
display(tourney_result.describe().transpose()[['min', 'max']])


# In[ ]:


print('seasons without results:')

season.loc[~season['Season'].isin(tourney_result['Season'])]


# - __the current tourney has no results yet of course__

# In[ ]:


# add game date (we can use DayZero + DayNum to get the actual GameDate)

tourney_result = pd.merge(tourney_result,
                          season[['Season', 'DayZero']],
                          on='Season',
                          how='inner')

tourney_result['GameDate'] = apply_column_offset(tourney_result, 'DayZero', 'DayNum')
tourney_result.head(3)


# ## Sample Submission
# - let's see how many possible matchups there are per year

# In[ ]:


sample_submission = pd.read_csv(data_dir + '../MSampleSubmissionStage1_2020.csv')

print(sample_submission.shape)
display(sample_submission.head())


# In[ ]:


sample_submission['Season'] = sample_submission['ID'].apply(lambda x: x.split('_')[0]).astype(int)

sample_submission['Season'].value_counts().sort_index()


# In[ ]:


68 * 67 / 2


# - as mentioned in [data](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/data), each year has "68\*67/2=2,278" possible tourney matchups

# # Simulate Test Data
# - for tourney games since 2015, let's create game IDs with the same format in the sample submission and look at the actual matchups that happened

# In[ ]:


test = tourney_result.loc[tourney_result['Season'] >= 2015].copy()

print(test.shape)
test.head()


# - we have 335 tourney games since 2015

# In[ ]:


def create_ID(row):
    # source: https://www.kaggle.com/catadanna/delete-leaked-from-training-ncaam-ncaaw-stage1
    if row['WTeamID'] < row['LTeamID']:
        ID = str(row['Season'])+"_"+str(row['WTeamID'])+"_"+str(row['LTeamID'])
    else:
        ID = str(row['Season'])+"_"+str(row['LTeamID'])+"_"+str(row['WTeamID'])
    return ID

test['ID'] = test.apply(create_ID, axis = 1)
test.head()


# In[ ]:


print('number of rows with true labels (had actual games):')
sample_submission['ID'].isin(test['ID']).sum()


# As mentioned by [Roman](https://www.kaggle.com/nroman) in [How to score 0 and why you should not do this](https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/discussion/131539**), only 335 rows in the test set matter as they are the only ones with true labels (had actual games).
# 
# All 335 tourney games since 2015 are in the test set, so that means that we can't use any data present since this time (except perhaps 2015 regular season data) if we don't want any leaks.

# # Test Submission
# - later on we can prepare the test data in the same way as the train data is prepared below, but for now I just used the sample submission

# In[ ]:


test = sample_submission.copy()

test['Team1'] = test['ID'].apply(lambda x: x.split('_')[1]).astype(int)
test['Team2'] = test['ID'].apply(lambda x: x.split('_')[2]).astype(int)

print(test.shape)
test.head()


# # Prepare Train Data
# - for this baseline I will only be using tourney (seed) data, so I will drop all data since 2015 to avoid leakage

# In[ ]:


# remove future data
train = tourney_result.loc[tourney_result['Season'] < 2015].copy()

# assign Team1 and Team2
rename_dict = {
    'WTeamID': 'Team1',
    'LTeamID': 'Team2',
    'WScore': 'Score1',
    'LScore': 'Score2',
    'WLoc': 'Loc1'
}
train = train.rename(columns=rename_dict)

# target is 1 when Team1 is the winner
train['target'] = 1

print(train.shape)
train.head(3)


# In [Data](http://https://www.kaggle.com/c/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/data) > Data Section 1 file: MSampleSubmissionStage1.csv, it was mentioned that:
# 
# "However, our evaluation utility will automatically infer the winning percentage in the other direction, so a 47% prediction for Arizona to win also means a 53% prediction for Duke to win."
# 
# To simulate this in the train set, we can add a flipped train set (like in [Jump Shot to Conclusions - March Madness EDA](https://www.kaggle.com/headsortails/jump-shot-to-conclusions-march-madness-eda) > Baseline Model) for the perspective of the other team:

# In[ ]:


# add flipped train set for perspective of other team

train_flipped = train.copy()

rename_dict = {
    'Team1': 'Team2',
    'Team2': 'Team1',
    'Score1': 'Score2',
    'Score2': 'Score1'
}

train_flipped = train_flipped.rename(columns=rename_dict)
train_flipped['target'] = 0

train = pd.concat([train, train_flipped], sort=True)
print(train.shape)
train.head(3)


# In[ ]:


print('train samples:', train.shape[0])
train['target'].value_counts()


# # Feature Engineering

# In[ ]:


train_orig = train.copy()
train = train_orig.copy()
team_nums = ['Team1', 'Team2']


# ## Seed
# - we can use the Season and TeamId to extract Seed data from MNCAATourneySeeds.csv

# In[ ]:


# Team1 and Team2 seeds

for i, team_num in enumerate(team_nums):

    train = pd.merge(train, 
                     seed, 
                     left_on=['Season', team_num ], 
                     right_on=['Season', 'TeamID'], 
                     how='left')

    train['var_seed{}'.format(i + 1)] = train['Seed'].apply(lambda x: x[1:3]).astype(int)
    train = train.drop(['Seed', 'TeamID'], axis=1)

print(train.shape)
train.head()


# In[ ]:


# do the same for test

for i, team_num in enumerate(team_nums):

    test = pd.merge(test, 
                     seed, 
                     left_on=['Season', team_num ], 
                     right_on=['Season', 'TeamID'], 
                     how='left')

    test['var_seed{}'.format(i + 1)] = test['Seed'].apply(lambda x: x[1:3]).astype(int)
    test = test.drop(['Seed', 'TeamID'], axis=1)

print(test.shape)
test.head()


# In[ ]:


# seed difference

train['var_seed_diff'] = train['var_seed1'] - train['var_seed2']
train.head()


# In[ ]:


# do the same for test

test['var_seed_diff'] = test['var_seed1'] - test['var_seed2']
test.head()


# # Logistic Regression
# - here we apply train a simple Logistic Regresison model and evaluate via 5-fold cross validation

# In[ ]:


features = [x for x in train.columns if 'var_' in x]
features


# In[ ]:


features = ['var_seed1',
 # 'var_seed2',
 'var_seed_diff'
]

# using either one of var_seed1 and var_seed_diff produces the same CV log_loss


# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# define model\nclf = LogisticRegression(random_state=2020)\n\n# define K-folds\nskf = StratifiedKFold(n_splits=5, \n                      shuffle=True, \n                      random_state=2020)\n\n# perform cross validation\nprint(\'training w/ {} features...\'.format(len(features)))\nscores = cross_val_score(clf,\n                         train[features],\n                         train[\'target\'],\n                         scoring=\'neg_log_loss\',\n                         cv=skf,\n                         n_jobs=2)\n\n# print roc_auc\nprint(\'\\n\', scores, \'\\n\')\n\nprint("avg log_loss:", round(-scores.mean(), 4))\nprint("std log_loss:", round(scores.std(), 4,), \'\\n\')')


# In[ ]:


# fit on all train data

clf.fit(train[features], train['target'])


# - we hope to get a log_loss close to 0.5533 in the LB (leaderboard)

# # Create Submission

# In[ ]:


submission = pd.DataFrame()
submission['ID'] = test['ID']
submission['pred'] = clf.predict_proba(test[features])[:, 1:]

submission.to_csv('submission.csv', index=False)

print(submission.shape)
submission.head()

