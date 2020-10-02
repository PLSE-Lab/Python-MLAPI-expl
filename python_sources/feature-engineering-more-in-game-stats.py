#!/usr/bin/env python
# coding: utf-8

# # Motivation

# The idea in this notebook is to try making sense of multiple data files, and constructing some relevant features that might provide extra information to the models. This is more of a practice in data manipulation, and testing how effective the in-game stats are in providing the results of the games. This approach comes with a lot of challenges- especially the lack of data pertaining to detailed stats. I shall touch more on this further.

# The basic transformation code is taken from some amazing kernels already available. I referred the kernel of @vbmokin https://www.kaggle.com/vbmokin/mm-ncaam-no-leaks-lgb-xgb-logreg

# # Import required packages

# In[ ]:


import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import eli5

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import log_loss
import lightgbm as lgb
import xgboost as xgb

import gc
import os


# # Load the data files

# In[ ]:


tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyDetailedResults.csv')
tourney_seed = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneySeeds.csv')


# In[ ]:


tourney_result = tourney_result.drop(['DayNum','WScore','LScore','WLoc','NumOT'],axis=1)


# # Feature Engineering

# In this part, we shall use the detailed stats available. We shall be grouping them by season and the team ID, so that they reflect the aggregate stats of a team in that particular season. The caveat with feeding your models with this extra piece of information is that detailed stats are available only from season 2003, hence you will be missing out on a huge chunk of training data.
# 
# The motivation behind taking this approach is, there already are kernels with LB score of 0.2, which is extremely high. The LB score of 0.4-0.5 is considered more ideal. Hence, to enable the models to generalize more, I am gonna test this approach of considering in-game stats.

# In[ ]:


season_win_stats = tourney_result[['Season','WTeamID','WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR',
       'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF']]
season_lose_stats = tourney_result[['Season','LTeamID','LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 
        'LDR', 'LAst', 'LTO', 'LStl', 'LBlk', 'LPF']]


# In[ ]:


season_win_stats.rename(columns=lambda x: x[1:], inplace=True)
season_lose_stats.rename(columns=lambda x: x[1:], inplace=True)
season_win_stats.rename(columns={'season':'Season'}, inplace=True)
season_lose_stats.rename(columns={'season':'Season'}, inplace=True)
season_stats = pd.concat((season_win_stats, season_lose_stats)).reset_index(drop=True)
season_stats.rename(columns={'eason':'Season'}, inplace=True)


# In[ ]:


seasonstatsbygroup = season_stats.groupby(['Season','TeamID'])['FGM'].sum().reset_index()


# In[ ]:


for stat in season_stats.columns:
    if stat in ['Season','TeamID','FGM']:
        continue
    else:
        seasonstatsbygroup[stat]=season_stats.groupby(['Season','TeamID'])[stat].sum().reset_index()[stat]


# In[ ]:


tourney_result = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MDataFiles_Stage1/MNCAATourneyCompactResults.csv')


# In[ ]:


tourney_result = tourney_result.drop(['DayNum', 'WScore', 'LScore', 'WLoc', 'NumOT'], axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'WTeamID'], 
                          right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'WSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, tourney_seed, left_on=['Season', 'LTeamID'], 
                          right_on=['Season', 'TeamID'], how='left')
tourney_result.rename(columns={'Seed':'LSeed'}, inplace=True)
tourney_result = tourney_result.drop('TeamID', axis=1)


# In[ ]:


def get_seed(x):
    return int(x[1:3])

tourney_result['WSeed'] = tourney_result['WSeed'].map(lambda x: get_seed(x))
tourney_result['LSeed'] = tourney_result['LSeed'].map(lambda x: get_seed(x))


# In[ ]:


tourney_result = tourney_result[(tourney_result['Season'] >= 2003)]


# In[ ]:


tourney_result = pd.merge(tourney_result, seasonstatsbygroup, left_on=['Season','WTeamID'],
                          right_on = ['Season','TeamID'], how='left')


# In[ ]:


for column in tourney_result.columns:
    if column in ['Season', 'WTeamID', 'LTeamID', 'WSeed', 'LSeed', 'TeamID']:
        continue
    else:
        tourney_result.rename(columns={column:'W'+column}, inplace=True)
        


# In[ ]:


tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_result = pd.merge(tourney_result, seasonstatsbygroup, left_on=['Season', 'LTeamID'], 
                          right_on=['Season', 'TeamID'], how='left')


# In[ ]:


for column in tourney_result.columns:
    if column in ['Season', 'WTeamID', 'LTeamID', 'WSeed', 'LSeed', 'WFGM', 'WFGA',
       'WFGM3', 'WFGA3', 'WFTM', 'WFTA', 'WOR', 'WDR', 'WAst', 'WTO', 'WStl',
       'WBlk', 'WPF', 'TeamID']:
        continue
    else:
        tourney_result.rename(columns={column:'L'+column}, inplace=True)


# In[ ]:


tourney_result = tourney_result.drop('TeamID', axis=1)
tourney_win_result = tourney_result.drop(['Season', 'WTeamID', 'LTeamID'], axis=1)


# In[ ]:


for column in tourney_win_result.columns:
    if column in ['WSeed', 'WFGM', 'WFGA', 'WFGM3', 'WFGA3', 'WFTM', 'WFTA',
       'WOR', 'WDR', 'WAst', 'WTO', 'WStl', 'WBlk', 'WPF' ]:
        tourney_win_result.rename(columns={column:column[1:]+'1'}, inplace=True)
    
    if column in ['LSeed', 'LFGM', 'LFGA', 'LFGM3', 'LFGA3', 'LFTM', 'LFTA', 'LOR', 'LDR', 'LAst', 'LTO', 'LStl',
       'LBlk', 'LPF' ]:
        tourney_win_result.rename(columns={column:column[1:]+'2'}, inplace=True)


# In[ ]:


tourney_lose_result = tourney_win_result.copy()


# In[ ]:


for column in tourney_win_result.columns:
    if column[-1]=='1':
        tourney_lose_result[column] = tourney_win_result[column[:-1]+'2']
    if column[-1]=='2':
        tourney_lose_result[column] = tourney_win_result[column[:-1]+'1']


# I will write a loop for this pafrt later.

# In[ ]:


tourney_win_result['Seed_diff'] = tourney_win_result['Seed1'] - tourney_win_result['Seed2']
tourney_win_result['FGM_diff'] = tourney_win_result['FGM1'] - tourney_win_result['FGM2']
tourney_win_result['FGA_diff'] = tourney_win_result['FGA1'] - tourney_win_result['FGA2']
tourney_win_result['FGM3_diff'] = tourney_win_result['FGM31'] - tourney_win_result['FGM32']
tourney_win_result['FGA3_diff'] = tourney_win_result['FGA31'] - tourney_win_result['FGA32']
tourney_win_result['FTM_diff'] = tourney_win_result['FTM1'] - tourney_win_result['FTM2']
tourney_win_result['FTA_diff'] = tourney_win_result['FTA1'] - tourney_win_result['FTA2']
tourney_win_result['OR_diff'] = tourney_win_result['OR1'] - tourney_win_result['OR2']
tourney_win_result['DR_diff'] = tourney_win_result['DR1'] - tourney_win_result['DR2']
tourney_win_result['Ast_diff'] = tourney_win_result['Ast1'] - tourney_win_result['Ast2']
tourney_win_result['TO_diff'] = tourney_win_result['TO1'] - tourney_win_result['TO2']
tourney_win_result['Stl_diff'] = tourney_win_result['Stl1'] - tourney_win_result['Stl2']
tourney_win_result['Blk_diff'] = tourney_win_result['Blk1'] - tourney_win_result['Blk2']
tourney_win_result['PF_diff'] = tourney_win_result['PF1'] - tourney_win_result['PF2']

tourney_lose_result['Seed_diff'] = tourney_lose_result['Seed1'] - tourney_lose_result['Seed2']
tourney_lose_result['FGM_diff'] = tourney_lose_result['FGM1'] - tourney_lose_result['FGM2']
tourney_lose_result['FGA_diff'] = tourney_lose_result['FGA1'] - tourney_lose_result['FGA2']
tourney_lose_result['FGM3_diff'] = tourney_lose_result['FGM31'] - tourney_lose_result['FGM32']
tourney_lose_result['FGA3_diff'] = tourney_lose_result['FGA31'] - tourney_lose_result['FGA32']
tourney_lose_result['FTM_diff'] = tourney_lose_result['FTM1'] - tourney_lose_result['FTM2']
tourney_lose_result['FTA_diff'] = tourney_lose_result['FTA1'] - tourney_lose_result['FTA2']
tourney_lose_result['OR_diff'] = tourney_lose_result['OR1'] - tourney_lose_result['OR2']
tourney_lose_result['DR_diff'] = tourney_lose_result['DR1'] - tourney_lose_result['DR2']
tourney_lose_result['Ast_diff'] = tourney_lose_result['Ast1'] - tourney_lose_result['Ast2']
tourney_lose_result['TO_diff'] = tourney_lose_result['TO1'] - tourney_lose_result['TO2']
tourney_lose_result['Stl_diff'] = tourney_lose_result['Stl1'] - tourney_lose_result['Stl2']
tourney_lose_result['Blk_diff'] = tourney_lose_result['Blk1'] - tourney_lose_result['Blk2']
tourney_lose_result['PF_diff'] = tourney_lose_result['PF1'] - tourney_lose_result['PF2']


# In[ ]:


tourney_win_result['result'] = 1
tourney_lose_result['result'] = 0
train_df = pd.concat((tourney_win_result, tourney_lose_result)).reset_index(drop=True)


# The training data is ready and this is how it looks like.

# In[ ]:


train_df


# # Creating the test data

# In[ ]:


test_df = pd.read_csv('../input/google-cloud-ncaa-march-madness-2020-division-1-mens-tournament/MSampleSubmissionStage1_2020.csv')


# In[ ]:


test_df['Season'] = test_df['ID'].map(lambda x: int(x[:4]))
test_df['WTeamID'] = test_df['ID'].map(lambda x: int(x[5:9]))
test_df['LTeamID'] = test_df['ID'].map(lambda x: int(x[10:14]))

test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed1'}, inplace=True)

test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, tourney_seed, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')
test_df.rename(columns={'Seed':'Seed2'}, inplace=True)
test_df = test_df.drop('TeamID', axis=1)

test_df = pd.merge(test_df, seasonstatsbygroup, left_on=['Season', 'WTeamID'], right_on=['Season', 'TeamID'], how='left')


# In[ ]:


for column in test_df.columns:
    if column in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast',
       'TO', 'Stl', 'Blk', 'PF']: 
        test_df.rename(columns={column:column+'1'}, inplace=True)
        


# In[ ]:


test_df = test_df.drop('TeamID', axis=1)
test_df = pd.merge(test_df, seasonstatsbygroup, left_on=['Season', 'LTeamID'], right_on=['Season', 'TeamID'], how='left')


# In[ ]:


for column in test_df.columns:
    if column in ['FGM', 'FGA', 'FGM3', 'FGA3', 'FTM', 'FTA', 'OR', 'DR', 'Ast',
       'TO', 'Stl', 'Blk', 'PF']: 
        test_df.rename(columns={column:column+'2'}, inplace=True)


# In[ ]:


test_df = test_df.drop('TeamID', axis=1)


# In[ ]:


test_df['Seed1'] = test_df['Seed1'].map(lambda x: get_seed(x))
test_df['Seed2'] = test_df['Seed2'].map(lambda x: get_seed(x))

test_df['Seed_diff'] = test_df['Seed1'] - test_df['Seed2']
test_df['FGM_diff'] = test_df['FGM1'] - test_df['FGM2']
test_df['FGA_diff'] = test_df['FGA1'] - test_df['FGA2']
test_df['FGM3_diff'] = test_df['FGM31'] - test_df['FGM32']
test_df['FGA3_diff'] = test_df['FGA31'] - test_df['FGA32']
test_df['FTM_diff'] = test_df['FTM1'] - test_df['FTM2']
test_df['FTA_diff'] = test_df['FTA1'] - test_df['FTA2']
test_df['OR_diff'] = test_df['OR1'] - test_df['OR2']
test_df['DR_diff'] = test_df['DR1'] - test_df['DR2']
test_df['Ast_diff'] = test_df['Ast1'] - test_df['Ast2']
test_df['TO_diff'] = test_df['TO1'] - test_df['TO2']
test_df['Stl_diff'] = test_df['Stl1'] - test_df['Stl2']
test_df['Blk_diff'] = test_df['Blk1'] - test_df['Blk2']
test_df['PF_diff'] = test_df['PF1'] - test_df['PF2']


# In[ ]:


test_df = test_df.drop(['ID', 'Pred', 'Season', 'WTeamID', 'LTeamID'], axis=1)


# In[ ]:


test_df

