#!/usr/bin/env python
# coding: utf-8

# # About
# This notebook's about making predictions. Exploratory Data Analysis is nice and important, but 1) that horse's been beaten a few times and 2) in my opinion, developing models is a lot harder, more interesting, and in the real world typically more lucrative than just analyzing data. 
# 
# In this notebook I hope to
# 
#   1) Illustrate a robust data cleaning *pipeline* and its value  
#   2) Develop a simple validation strategy with scores that generalize to unseen data  
#   3) Train a neural network, gradient boosting model, and logistic regression model, and show how ensembling them provides extra lift

# ### Why the name Professor?
# I like giving my projects names that give them life, so I tend to name things after people, pets, and places local to me. Professor is the name of my dog :)
# 
# [@professor.walt](https://www.instagram.com/professor.walt/)
# ![professor](https://i.imgur.com/Dx0mr1Cm.png)

# # Imports

# In[ ]:


import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression

import lightgbm as lgb

import matplotlib.pyplot as plt


# # Helper Functions
# ---
# 
# Helper functions are key to a good data wrangling pipeline. Typically I'd store these into a *helpers_generic.py* file and a *helpers_project.py* file where:
# 
# - *helpers_generic.py* stores **really** generic functions that you'd use across different projects (e.g. *log_loss()*)
# - *helpers_project.py* stores functions that you'd only use for the current project (e.g. *rank_teams()*)
# 
# Below I present three generic helper functions which are critial to my data wrangling pipeline. I've tried to explain them in the docstrings, but the gist is
# 
# 1. *join_insertion(into_df, from_df, ...)*  
#     this is useful if you want to insert values from a dataframe into another dataframe via some matching columns  
#     
# 2. *dataframe_cj(df1, df2)*  
#     performs a cartesian join between two dataframes. So if df1 has n rows and df2 has k rows, the result has n*k rows
#     
# 3. *array_cj(arrays, out=None)*  
#     performs a cartesian join between a sequence of numpy 1d arrays

# In[ ]:


def join_insertion(into_df, from_df, cols, on, by=None, direction=None, mult='error'):
    """
    Suppose A and B are dataframes. A has columns {foo, bar, baz} and B has columns {foo, baz, buz}
    This function allows you to do an operation like:
    "where A and B match via the column foo, insert the values of baz and buz from B into A"
    Note that this'll update A's values for baz and it'll insert buz as a new column.
    This is a lot like DataFrame.update(), but that method annoyingly ignores NaN values in B!

    Optionally, direction can be given as 'backward', 'forward', or nearest to implement a rolling join
    insertion. forward means 'roll into_df values forward to match from_df values', etc. Additionally,
    when doing a rolling join, 'on' should be the roll column and 'by' should be the exact-match columns.
    See pandas.merge_asof() for details.

    Note that 'mult' gets ignored when doing a rolling join. In the case of a rolling join, the first
    appearing record is kept, even if two records match a key from the same distance. Perhaps this
    can be improved...

    :param into_df: dataframe you want to modify
    :param from_df: dataframe with the values you want to insert
    :param cols: list of column names (values to insert)
    :param on: list of column names (values to join on), or a dict of {into:from} column name pairs
    :param by: same format as on; when doing a rolling join insertion, what columns to exact-match on
    :param direction: 'forward', 'backward', or 'nearest'. forward means roll into_df values to match from_df
    :param mult: if a key of into_df matches multiple rows of from_df, how should this be handled?
    an error can be raised, or the first matching value can be inserted, or the last matching value
    can be inserted
    :return: a modified copy of into_df, with updated values using from_df
    """

    # Infer left_on, right_on
    if (isinstance(on, dict)):
        left_on = list(on.keys())
        right_on = list(on.values())
    elif(isinstance(on, list)):
        left_on = on
        right_on = on
    elif(isinstance(on, str)):
        left_on = [on]
        right_on = [on]
    else:
        raise Exception("on should be a list or dictionary")

    # Infer left_by, right_by
    if(by is not None):
        if (isinstance(by, dict)):
            left_by = list(by.keys())
            right_by = list(by.values())
        elif (isinstance(by, list)):
            left_by = by
            right_by = by
        elif (isinstance(by, str)):
            left_by = [by]
            right_by = [by]
        else:
            raise Exception("by should be a list or dictionary")
    else:
        left_by = None
        right_by = None

    # Make cols a list if it isn't already
    if(isinstance(cols, str)):
        cols = [cols]

    # Setup
    A = into_df.copy()
    B = from_df[right_on + cols + ([] if right_by is None else right_by)].copy()

    # Insert row ids
    A['_A_RowId_'] = np.arange(A.shape[0])
    B['_B_RowId_'] = np.arange(B.shape[0])

    # Merge
    if(direction is None):
        A = pd.merge(
            left=A,
            right=B,
            how='left',
            left_on=left_on,
            right_on=right_on,
            suffixes=(None, '_y'),
            indicator=True
        ).sort_values(['_A_RowId_', '_B_RowId_'])

        # Check for rows of A which got duplicated by the merge, and then handle appropriately
        if (mult == 'error'):
            if (A.groupby('_A_RowId_').size().max() > 1):
                raise Exception("At least one key of into_df matched multiple rows of from_df.")
        elif (mult == 'first'):
            A = A.groupby('_A_RowId_').first().reset_index()
        elif (mult == 'last'):
            A = A.groupby('_A_RowId_').last().reset_index()

    else:
        A.sort_values(left_on, inplace=True)
        B.sort_values(right_on, inplace=True)
        A = pd.merge_asof(
            left=A,
            right=B,
            direction=direction,
            left_on=left_on,
            right_on=right_on,
            left_by=left_by,
            right_by=right_by,
            suffixes=(None, '_y')
        ).sort_values(['_A_RowId_', '_B_RowId_'])

    # Insert values from new column(s) into pre-existing column(s)
    mask = A._merge == 'both' if direction is None else np.repeat(True, A.shape[0])
    cols_in_both = list(set(into_df.columns.to_list()).intersection(set(cols)))
    for col in cols_in_both:
        A.loc[mask, col] = A.loc[mask, col + '_y']

    # Drop unwanted columns
    A.drop(columns=list(set(A.columns).difference(set(into_df.columns.to_list() + cols))), inplace=True)

    return A

def dataframe_cj(df1, df2):
    """
    given two dataframes, compute their cartesian product

    :param df1: dataframe
    :param df2: dataframe
    :return: cartesian product of input dataframes
    """

    df1copy = df1.copy()
    df2copy = df2.copy()
    df1copy['_Temp_'] = 1
    df2copy['_Temp_'] = 1
    result = pd.merge(df1copy, df2copy, on='_Temp_').drop(columns = '_Temp_')

    return result

def array_cj(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    cj(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = int(n / arrays[0].size)
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        array_cj(arrays[1:], out=out[0:m,1:])
        for j in range(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


# # Load Data
# Here we load the stage-1 input files. Nothing special.

# In[ ]:


# Load the data
rsgames = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MRegularSeasonCompactResults.csv")
seasons = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MSeasons.csv")
teams = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MTeams.csv")
trnygames = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv")
trnyseeds = pd.read_csv("../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv")


# ***Pro Tip*******  
# Load data objects in alphabetical order so its easier to see if something's missing or being loaded twice.

# # Wrangle
# ---
# 
# NOT SHOWN HERE is me inspecting the input data, understanding its format, and validating certain assumptions about it.
# 
# Trying to build a model directly from those input files is impractical. Our lives will be a lot easier if we reorganize the data into a more useful format. In practice, I whip out a pen and a notebook and draw the ideal datasets I want to work with, and then I make it happen.
# 
# My argument is that the following reoganized datasets would be a lot easier to work with:
# 
# 1. **games**  
# One row per game. Includes columns: {Season, DayNum, Round, Team1ID, Team2ID, Team1Score, Team2Score, Team1Won} where Team1ID < Team2ID and Round = -1 for regular season, 0 for play-in games, 1 for 1st round of the tournament, ... 6 for championship game
# 
# 2. **gameteams**  
# One row per (game, team). So, one game will correspond to two (game, team)s. Includes columns: {Season, DayNum, Round, TeamID, OppID, Score, OppScore, Won}
# 
# 3. **seasonteams**  
# One row per (season, team). Includes columns: {Season, DayNum, Round, TeamID, Games, Wins, PPG (Points Per Game)} where stats like Games and Wins are based on regular-season play
# 
# Additionally, we'll improve some of the input datasets like **trnyseeds** and **trnygames**.
# 

# ### rd_pairs

# In[ ]:


# Determine all possible (Region1, Region2, SeedNum1, SeedNum2, Round)s excluding play-in games
# The purpose of this is to develop a key such that, for any tournament game, if we know the Region and Seed
# of the teams playing, we can lookup the Round (-1 = regular season, 0 = play-in, 1 = 1st round, 2 = 2nd round, ... 6 = championship)

rd1_seeds = [np.array([i]) for i in [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]]
rd2_seeds = [np.concatenate((rd1_seeds[i], rd1_seeds[i + 1])) for i in range(0, len(rd1_seeds), 2)]
rd3_seeds = [np.concatenate((rd2_seeds[i], rd2_seeds[i + 1])) for i in range(0, len(rd2_seeds), 2)]
rd4_seeds = [np.concatenate((rd3_seeds[i], rd3_seeds[i + 1])) for i in range(0, len(rd3_seeds), 2)]
rd1_matchups = [array_cj((rd1_seeds[i], rd1_seeds[i+1])) for i in range(0, len(rd1_seeds), 2)]
rd2_matchups = [array_cj((rd2_seeds[i], rd2_seeds[i+1])) for i in range(0, len(rd2_seeds), 2)]
rd3_matchups = [array_cj((rd3_seeds[i], rd3_seeds[i+1])) for i in range(0, len(rd3_seeds), 2)]
rd4_matchups = [array_cj((rd4_seeds[i], rd4_seeds[i+1])) for i in range(0, len(rd4_seeds), 2)]
rd1_pairs = pd.DataFrame(np.concatenate(rd1_matchups), columns = ['SeedNum1', 'SeedNum2'])
rd2_pairs = pd.DataFrame(np.concatenate(rd2_matchups), columns = ['SeedNum1', 'SeedNum2'])
rd3_pairs = pd.DataFrame(np.concatenate(rd3_matchups), columns = ['SeedNum1', 'SeedNum2'])
rd4_pairs = pd.DataFrame(np.concatenate(rd4_matchups), columns = ['SeedNum1', 'SeedNum2'])
rd1_pairs['Round'] = 1
rd2_pairs['Round'] = 2
rd3_pairs['Round'] = 3
rd4_pairs['Round'] = 4
rd1234_pairs_forward = pd.concat((rd1_pairs, rd2_pairs, rd3_pairs, rd4_pairs), sort=True)
rd1234_pairs_reverse = rd1234_pairs_forward.rename(columns={'SeedNum2':'SeedNum1', 'SeedNum1':'SeedNum2'})
rd1234_pairs = pd.concat((rd1234_pairs_forward, rd1234_pairs_reverse), sort=True)
rd1234_pairs = dataframe_cj(rd1234_pairs, pd.DataFrame({'Region1':['W','X','Y','Z'], 'Region2':['W','X','Y','Z']}))
rd1234_pairs = rd1234_pairs[['Region1', 'Region2', 'SeedNum1', 'SeedNum2', 'Round']]
seed_combos = dataframe_cj(pd.DataFrame({'SeedNum1':np.arange(16) + 1}), pd.DataFrame({'SeedNum2':np.arange(16) + 1}))
rd5_region_combos = pd.DataFrame({'Region1':['W','X','Y','Z'], 'Region2':['X','W','Z','Y']})
rd5_pairs = dataframe_cj(seed_combos, rd5_region_combos)[['Region1', 'Region2', 'SeedNum1', 'SeedNum2']]
rd5_pairs['Round'] = 5
rd6_region_combos = pd.DataFrame({'Region1':['W','W','X','X','Y','Y','Z','Z'], 'Region2':['Y','Z','Y','Z','W','X','W','X']})
rd6_pairs = dataframe_cj(seed_combos, rd6_region_combos)[['Region1', 'Region2', 'SeedNum1', 'SeedNum2']]
rd6_pairs['Round'] = 6
rd_pairs = pd.concat((rd1234_pairs, rd5_pairs, rd6_pairs), sort=True)

# Print snippet
rd_pairs.head()


# ### trnyseeds
# 
# ***Pro Tip***  
# Use regular expressions to extract information from strings. It's way more reliable than extracting substrings by character *position*. For a good tutorial on regular expressions, check out [RegexOne](https://regexone.com/).

# In[ ]:


trnyseeds['SeedNum'] = trnyseeds.Seed.str.extract(r'(\d+)').astype('int64')
trnyseeds['Region'] = trnyseeds.Seed.str.extract(r'(^[A-Z])')
trnyseeds['PlayIn'] = trnyseeds.Seed.str.extract(r'([a-z]$)')

# Print snippet
trnyseeds.head()


# ### games

# In[ ]:


# Determine Team1, Team2
rsgames['Team1ID'] = np.where(rsgames.WTeamID.to_numpy() < rsgames.LTeamID.to_numpy(), rsgames.WTeamID.to_numpy(), rsgames.LTeamID.to_numpy())
rsgames['Team2ID'] = np.where(rsgames.WTeamID.to_numpy() > rsgames.LTeamID.to_numpy(), rsgames.WTeamID.to_numpy(), rsgames.LTeamID.to_numpy())
trnygames['Team1ID'] = np.where(trnygames.WTeamID.to_numpy() < trnygames.LTeamID.to_numpy(), trnygames.WTeamID.to_numpy(), trnygames.LTeamID.to_numpy())
trnygames['Team2ID'] = np.where(trnygames.WTeamID.to_numpy() > trnygames.LTeamID.to_numpy(), trnygames.WTeamID.to_numpy(), trnygames.LTeamID.to_numpy())

# Insert Seed into trnygames
trnygames = join_insertion(into_df=trnygames, from_df=trnyseeds, on={'Season':'Season', 'Team1ID':'TeamID'}, cols=['SeedNum', 'Region', 'PlayIn'])
trnygames.rename(columns={'SeedNum':'SeedNum1', 'Region':'Region1', 'PlayIn':'PlayIn1'}, inplace=True)
trnygames = join_insertion(into_df=trnygames, from_df=trnyseeds, on={'Season':'Season', 'Team2ID':'TeamID'}, cols=['SeedNum', 'Region', 'PlayIn'])
trnygames.rename(columns={'SeedNum':'SeedNum2', 'Region':'Region2', 'PlayIn':'PlayIn2'}, inplace=True)

# Insert Round
trnygames = join_insertion(
    into_df=trnygames,
    from_df=rd_pairs,
    on=['Region1', 'Region2', 'SeedNum1', 'SeedNum2'],
    cols='Round'
)
trnygames.loc[(trnygames.Region1 == trnygames.Region2) & (trnygames.SeedNum1 == trnygames.SeedNum2), 'Round'] = 0
rsgames['Round'] = -1

# Make games
trnygames['Team1Won'] = np.where(trnygames.Team1ID == trnygames.WTeamID, True, False)
rsgames['Team1Won'] = np.where(rsgames.Team1ID == rsgames.WTeamID, True, False)
trnygames['Team1Score'] = np.where(trnygames.Team1ID == trnygames.WTeamID, trnygames.WScore, trnygames.LScore)
rsgames['Team1Score'] = np.where(rsgames.Team1ID == rsgames.WTeamID, rsgames.WScore, rsgames.LScore)
trnygames['Team2Score'] = np.where(trnygames.Team2ID == trnygames.WTeamID, trnygames.WScore, trnygames.LScore)
rsgames['Team2Score'] = np.where(rsgames.Team2ID == rsgames.WTeamID, rsgames.WScore, rsgames.LScore)
games = pd.concat((rsgames, trnygames[rsgames.columns]), axis=0)
games['Team1MOV'] = games.Team1Score - games.Team2Score

# Print snippet
games.head()


# ### gameteams

# In[ ]:


gameteams1 = games.rename(columns={
    'Team1ID':'TeamID',
    'Team1Score':'Score',
    'Team2ID':'OppID',
    'Team2Score':'OppScore'
}).drop(columns=['Team1Won', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc'])
gameteams2 = games.rename(columns={
    'Team2ID':'TeamID',
    'Team2Score':'Score',
    'Team1ID':'OppID',
    'Team1Score':'OppScore'
}).drop(columns=['Team1Won', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc'])

# Combine gameteams1 and gameteams2
gameteams = pd.concat((gameteams1, gameteams2), sort=True)

# Insert fields, Won, MOV (Margin Of Victory)
gameteams['Won'] = gameteams.Score > gameteams.OppScore
gameteams['MOV'] = gameteams.Score - gameteams.OppScore

# Print snippet
gameteams.head()


# ### seasonteams

# In[ ]:


# Aggregate regular seaon gameteams by (Season, TeamID) and calculate basic stats
seasonteams = gameteams.loc[gameteams.Round == -1].groupby(['Season', 'TeamID']).agg(
    Games=pd.NamedAgg(column='TeamID', aggfunc=np.size),
    Wins=pd.NamedAgg(column='Won', aggfunc='sum'),
    Points=pd.NamedAgg(column='Score', aggfunc='sum'),
    OppPoints=pd.NamedAgg(column='OppScore', aggfunc='sum'),
    AvgMOV=pd.NamedAgg(column='MOV', aggfunc='mean')
).reset_index()
seasonteams['WinPct'] = seasonteams.Wins/seasonteams.Games
seasonteams['PPG'] = seasonteams.Points/seasonteams.Games
seasonteams['OppPPG'] = seasonteams.OppPoints/seasonteams.Games

# For each (Season, Team), identify the team as making the tournament or not
trnyteams = gameteams.loc[gameteams.Round >= 0, ['Season', 'TeamID']].drop_duplicates()
trnyteams['InTourney'] = True
seasonteams['InTourney'] = False
seasonteams = join_insertion(into_df=seasonteams, from_df=trnyteams, cols='InTourney', on=['Season', 'TeamID'])

# Print snippet
seasonteams.head()


# # Professor
# ----
# 
# Before we start building models, let's construct one more dataset - **modeldata**. This dataset will have every conceivable tournament game; (Season, Team1, Team2) tuples including tournament games that were never played but *could* have been played. For example, the 2019 tournament started with 68 teams. Pick any two teams from that tournament. As long as they keep winning, they'll eventually match up against each other (or perhaps they matched up at the very start). This means that tournaments that start with 68 teams will generate 68\*67/2 conceivable matchups - each represented as a row in our **modeldata** dataset.
# 
# The purpose of this dataset is to have training, validation, and test records to draw from for developing our model.

# ## Model Data

# In[ ]:


# Build every possible (Team1ID vs Team2ID) tournament game. Most of these will have never been played
modeldata = trnyteams[['Season', 'TeamID']]
modeldata = modeldata.groupby('Season').apply(lambda x: dataframe_cj(x[['TeamID']], x[['TeamID']]))
modeldata = modeldata.reset_index().drop(columns=['level_1']).rename(columns={'TeamID_x':'Team1ID', 'TeamID_y':'Team2ID'})
modeldata = modeldata.loc[modeldata.Team1ID < modeldata.Team2ID]

# Insert team seeds and regions
modeldata = join_insertion(
    into_df=modeldata,
    from_df=trnyseeds,
    cols=['SeedNum', 'Region', 'PlayIn'],
    on={'Season':'Season', 'Team1ID':'TeamID'}
).rename(columns={'SeedNum':'SeedNum1', 'Region':'Region1', 'PlayIn':'PlayIn1'})
modeldata = join_insertion(
    into_df=modeldata,
    from_df=trnyseeds,
    cols=['SeedNum', 'Region', 'PlayIn'],
    on={'Season':'Season', 'Team2ID':'TeamID'}
).rename(columns={'SeedNum':'SeedNum2', 'Region':'Region2', 'PlayIn':'PlayIn2'})

# Insert Round
modeldata = join_insertion(
    into_df=modeldata,
    from_df=rd_pairs,
    on=['Region1', 'Region2', 'SeedNum1', 'SeedNum2'],
    cols='Round'
)
modeldata.loc[(modeldata.Region1 == modeldata.Region2) & (modeldata.SeedNum1 == modeldata.SeedNum2), 'Round'] = 0

# Insert season-team stats
modeldata = join_insertion(
    into_df=modeldata,
    from_df=seasonteams,
    cols=['WinPct', 'AvgMOV', 'PPG', 'OppPPG'],
    on={'Season':'Season', 'Team1ID':'TeamID'}
)
modeldata.rename(columns={'WinPct':'WinPct1', 'AvgMOV':'AvgMOV1', 'PPG':'PPG1', 'OppPPG':'OppPPG1'}, inplace=True)
modeldata = join_insertion(
    into_df=modeldata,
    from_df=seasonteams,
    cols=['WinPct', 'AvgMOV', 'PPG', 'OppPPG'],
    on={'Season':'Season', 'Team2ID':'TeamID'}
)
modeldata.rename(columns={'WinPct':'WinPct2', 'AvgMOV':'AvgMOV2', 'PPG':'PPG2', 'OppPPG':'OppPPG2'}, inplace=True)

# Insert result from most recent team1 vs team2 matchup in the same season
rsgames['Team1MOV'] = rsgames.Team1Score - rsgames.Team2Score
modeldata = join_insertion(
    into_df=modeldata,
    from_df=rsgames,
    cols='Team1MOV',
    on=['Season', 'Team1ID', 'Team2ID'],
    mult='last'
)
modeldata.rename(columns={'Team1MOV':'Team1RecentMOV'}, inplace=True)
modeldata.fillna({'Team1RecentMOV':0}, inplace=True)

# Insert game stats for games which were actually played
modeldata = join_insertion(
    into_df=modeldata,
    from_df=trnygames,
    cols=['Team1Score', 'Team2Score', 'Team1Won'],
    on=['Season', 'Team1ID', 'Team2ID']
)

# Print snippet
modeldata.head()


# ## Train/Eval/Test Split
# 
# Here we split modeldata into three datasets:
# 
# 1. **train** - modeldata from seasons 1985 - 2014. This is the input we'll feed into our learners (nnet, lightgbm, logistic regression).
# 2. **eval** - model data from seasons 2015 - 2019. As our nnet and lightgbm models go throught the training procedure, we'll evaluate their performance on this dataset to know when to cut-off the training process and prevent overfitting
# 3. **test** - model data from seasons 2015 - 2019. This is the dataset we want to make predicitons for
# 
# **We're cheating a little bit**. Our models will have the benefit of using the actual test data in their training process. But this type of leakage isn't that bad since we're not exposing our models to the individual targets of the test dataset. The benefit of this approach is simplicity and not cutting our training data into a tiny dataset.

# In[ ]:


# train on seasons 1985 - 2014 (30 seasons)
# use seasons 2015 - 2019 to evaluate results and perform early stopping
train = modeldata.loc[(modeldata.Season < 2015) & (modeldata.Team1Won.notna())].copy()
eval = modeldata.loc[(modeldata.Season >= 2015) & (modeldata.Team1Won.notna())].copy()

train['Team1Won'] = train.Team1Won.astype('bool')
eval['Team1Won'] = eval.Team1Won.astype('bool')

# Build test dataset of (Season, Team1ID, Team2ID) for every possible Team1ID vs Team2ID within each tournament
test = modeldata.loc[modeldata.Season >= 2015].copy()


# ## Keras NNet Model
# 
# Here we build a densly connected neural network model with an input layer, two hidden layers, and an output layer, each with sigmoid activation, as well as a couple droupout layers to prevent over-fitting. Note the imports from tensorflow.keras up above ^^. 
# 
# **Why this particular neural network architecture?**  
# No particular reason, other than I tinkered with a few structures and this one seemed to work well.

# In[ ]:


#--- setup --------------------------------------

# Build xtrain, xeval, xtest, ytrain, yeval
feats = [
    'Round', 'Team1RecentMOV',
    'SeedNum1', 'SeedNum2', 'WinPct1', 'WinPct2', 'AvgMOV1', 'AvgMOV2', 'PPG1', 'PPG2', 'OppPPG1', 'OppPPG2'
]

xtrain = train[feats].to_numpy()
xeval = eval[feats].to_numpy()
xtest = test[feats].to_numpy()

ytrain = train.Team1Won.to_numpy().astype('int64')
yeval = eval.Team1Won.to_numpy().astype('int64')

# Scale features to 0-1 (not quite true, but close enough)
xtest = xtest/np.max(xtrain, axis=0)
xeval = xeval/np.max(xtrain, axis=0)
xtrain = xtrain/np.max(xtrain, axis=0)


# In[ ]:


#--- model --------------------------------------

nnet = Sequential()

nnet.add(Dense(64, activation='sigmoid'))
nnet.add(Dropout(0.5))

nnet.add(Dense(16, activation='sigmoid'))
nnet.add(Dropout(0.2))

nnet.add(Dense(1, activation='sigmoid'))

nnet.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

early = EarlyStopping(patience=50, restore_best_weights=True)
nnet.fit(x = xtrain, y = ytrain, batch_size=64, epochs=999, validation_data=(xeval, yeval), callbacks=[early])


# In[ ]:


#--- evaluate --------------------------------------

# Insert preds
eval['Prob_NNet'] = nnet.predict_proba(xeval).astype('float64')
test['Prob_NNet'] = nnet.predict_proba(xtest).astype('float64')

# Plot validation score over time
pd.DataFrame(nnet.history.history).plot(figsize=(12, 8))

# log loss on all seasons combined
log_loss(y_true=eval.Team1Won, y_pred=eval.Prob_NNet)  # 0.5499

# log loss by season
eval.groupby('Season')[['Team1Won','Prob_NNet']].apply(lambda x: log_loss(x.Team1Won, x.Prob_NNet))


# ## LightGBM Model
# 
# Next, we build a gradient boosting model via LightGBM. We'll use the same exact train & eval data as before for convenience. Unlike NNets, we don't need to our input data to be scaled to 0-1 for lightgbm, but doing so shouldn't degrade performance.
# 
# I didn't spend a lot of time tuning hyper-parameters. But the params below seemed to work well.

# In[ ]:


#--- model --------------------------------------

# Create dataset for lightgbm
lgb_train = lgb.Dataset(xtrain, ytrain)
lgb_eval = lgb.Dataset(xeval, yeval, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': ['binary_logloss'],
    'num_leaves': 21,
    'learning_rate': 0.01,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.5,
    'bagging_freq': 5,
    'verbose': 0
}

# train
gbm = lgb.train(
    params=params,
    train_set=lgb_train,
    num_boost_round=999,
    valid_sets=lgb_eval,
    early_stopping_rounds=10
)


# In[ ]:


#--- evaluate --------------------------------------

# Insert preds
eval['Prob_LGB'] = gbm.predict(xeval, num_iteration=gbm.best_iteration)
test['Prob_LGB'] = gbm.predict(xtest, num_iteration=gbm.best_iteration)

# feature importance
print(pd.DataFrame({'Feat':feats, 'Gain':gbm.feature_importance(importance_type='gain')}).sort_values('Gain', ascending=False))

# log loss on all seasons combined
log_loss(y_true=eval.Team1Won, y_pred=eval.Prob_LGB)  # 0.5482

# log loss by season
eval.groupby('Season')[['Team1Won','Prob_LGB']].apply(lambda x: log_loss(x.Team1Won, x.Prob_LGB))


# ## Logistic Regression Model
# 
# Here it turns out that our scaled data does worse than our unscaled data, so we rebuild xtrain and xeval without scaling the features to 0-1.

# In[ ]:


#--- setup --------------------------------------

# Build xtrain, xeval, xtest, ytrain, yeval
xtrain = train[feats].to_numpy()
xeval = eval[feats].to_numpy()
xtest = test[feats].to_numpy()
ytrain = train.Team1Won.to_numpy().astype('int64')
yeval = eval.Team1Won.to_numpy().astype('int64')


# In[ ]:


#--- model --------------------------------------

# Fit model
lr = LogisticRegression(random_state=0, multi_class='ovr', solver='lbfgs', max_iter=1000).fit(xtrain, ytrain)


# In[ ]:


#--- evaluate --------------------------------------

# Insert preds
eval['Prob_LR'] = lr.predict_proba(xeval)[:, 1]
test['Prob_LR'] = lr.predict_proba(xtest)[:, 1]

# log loss on all seasons combined
log_loss(y_true=eval.Team1Won, y_pred=eval.Prob_LR)  # 0.5535

# log loss by season
eval.groupby('Season')[['Team1Won','Prob_LR']].apply(lambda x: log_loss(x.Team1Won, x.Prob_LR))


# ## Ensemble
# 
# It's often the case that ensembling diverse models together can do better than your best single model in isolation. We test that here using a simple averaging of the predicted probabilities from each model.

# In[ ]:


# Combine predictions
eval['ProbAvg'] = (eval.Prob_NNet + eval.Prob_LGB + eval.Prob_LR)/3
test['ProbAvg'] = (test.Prob_NNet + test.Prob_LGB + test.Prob_LR)/3

# Compare
print('NNet: {0}'.format(log_loss(eval.Team1Won, eval.Prob_NNet)))   # 0.5499
print('LGB: {0}'.format(log_loss(eval.Team1Won, eval.Prob_LGB)))     # 0.5482
print('LR: {0}'.format(log_loss(eval.Team1Won, eval.Prob_LR)))       # 0.5535
print('Ensemble: {0}'.format(log_loss(eval.Team1Won, eval.ProbAvg))) # 0.5475


# Indeed, tis the case here!

# # Prepare Submission
# ---
# 
# Alas, we prepare our submission and hold our breath.

# In[ ]:


test['ID'] = test.Season.astype('str') + '_' + test.Team1ID.astype('str') + '_' + test.Team2ID.astype('str')
subm = test[['ID', 'ProbAvg']].rename(columns = {'ProbAvg':'Pred'})
subm.to_csv('professor.csv', index=False)

