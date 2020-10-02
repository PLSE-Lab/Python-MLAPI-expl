#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
import os
import re

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# In[ ]:


# container_dir = '/kaggle/input/march-madness-analytics-2020/2020DataFiles/2020DataFiles/2020-Mens-Data/MDataFiles_Stage1'
container_dir = '/kaggle/input/march-madness-analytics-2020/MDataFiles_Stage2/'


# # Purpose of the notebook
# In this notebook we are going to tranform into a __tidy format__ the next dataframes:
#  1. MRegularSeasonCompactResults
#  2. MNCAATourneyCompactResults
#  3. MSecondaryTourneyCompactResults
#  4. MRegularSeasonDetailedResults
#  5. MNCAATourneyDetailedResults
#  6. MSeasons
#  7. cities
#  8. MGameCities
#  
#  
#  
# __Notes:__
# - There are no functions in this notebook for educational purposes. You can play around with each step and feel free to stop and print variables to get a better understanding if needed.
# - This notebook's final result is based on tidy data Hadley Wickham's definition https://en.wikipedia.org/wiki/Tidy_data 

# # Dataframes with game detail level
# It's pretty important in a multi-dataframe modelling process take into account the level of detail in each dataframe (also known as granularity). The main idea is avoid duplicate values and keep everything in one place.

# ### Working with compact data first
# This dataframes share the same structured, so let's append and merge them.

# In[ ]:


#loading data Compact data
regular_compact = pd.read_csv(os.path.join(container_dir, 'MRegularSeasonCompactResults.csv'))
tourney_compact = pd.read_csv(os.path.join(container_dir, 'MNCAATourneyCompactResults.csv'))
secondary_compact = pd.read_csv(os.path.join(container_dir, 'MSecondaryTourneyCompactResults.csv'))


# In[ ]:


#creating identifier fields:;
regular_compact['identifier'] = 'regular'
tourney_compact['identifier'] = 'tourney'
secondary_compact['identifier'] = 'secondary'

#appending compact dataframes:
append_compact = pd.concat([regular_compact, tourney_compact, secondary_compact])


# Same process for detailed data

# In[ ]:


#loading detailed data:
regular_detailed = pd.read_csv(os.path.join(container_dir, 'MRegularSeasonDetailedResults.csv'))
tourney_detailed = pd.read_csv(os.path.join(container_dir, 'MNCAATourneyDetailedResults.csv'))
#Note: secondaryDetailed data wasn't delivered.


# In[ ]:


#creating detailed identifier fields:
regular_detailed['identifier'] = 'regular'
tourney_detailed['identifier'] = 'tourney'

#appending detailed dataframes:
append_detailed = pd.concat([regular_detailed, tourney_detailed])


# Merge dataframes

# In[ ]:


merge_results = append_detailed.merge(append_compact, how = 'outer', on=['Season', 'DayNum', 'WTeamID', 'WScore', 'LTeamID', 'LScore', 'WLoc',
       'NumOT', 'identifier'])
merge_results['game_id'] = np.arange(0, merge_results.shape[0])
merge_results['score_diff'] = merge_results.WScore - merge_results.LScore


# # Unpivoting data
# The main idea is try to integrate columns with the same meaning, for example, WTeamID (winner team id) and LTeamID (loser team id). Doesn't make sense keep them  separated so lets join them.
# Let's use **__melt__** to achieve this.

# In[ ]:


# renaming columns to unpivot
merge_results.rename(columns = {'WTeamID':'Winner','LTeamID': 'Loser'}, inplace = True)

value_vars = ['Winner', 'Loser']
id_vars = [i for i in merge_results.columns if i not in value_vars]
melted_results = merge_results.melt(id_vars = id_vars,
                  value_vars = value_vars,
                  var_name = 'game_status',
                  value_name = 'TeamID')


# In[ ]:


melted_results.head()


# Ok, maybe this looks a little ugly, we have some repeated registers. For example in __season__ 2003, __season day__ 10,  __teams__ 1104 and 1328 were matched. Team 1104 won, so we just need those 'Winner regiters' and drop 'Loser registers' opposite applies to team 1328 

# In[ ]:


mask =  (melted_results.Season== 2003) & (melted_results.DayNum == 10) &(melted_results.TeamID == 1104)
melted_results.loc[mask, :]


# In[ ]:


mask =  (melted_results.Season== 2003) & (melted_results.DayNum == 10) &(melted_results.TeamID == 1328)
melted_results.loc[mask, :]


# In[ ]:


melted_cols = melted_results.columns

# Columns beginning with L
loser_mask_cols = melted_cols.str.contains('^L')
loser_cols = melted_cols[loser_mask_cols]

# Columns beginning with W living Wloc aside
winner_mask_cols = (melted_cols.str.contains('^W')) & (melted_cols != 'WLoc')
winner_cols = melted_cols[winner_mask_cols]

cleaned_cols = winner_cols.str.extract(r'([^W].*)')[0].tolist()

for winner_col, loser_col, new_col in zip(winner_cols, loser_cols, cleaned_cols):
    melted_results[new_col] = np.where(melted_results['game_status'] == 'Winner',
                                      melted_results[winner_col],
                                      melted_results[loser_col])


# In[ ]:


useless_cols = list(winner_cols)+list(loser_cols)
melted_results.drop(useless_cols, axis=1, inplace = True)


# In[ ]:


melted_results.head()


# Much better!

# ### Fixing Wloc behavior
# Wloc columns represents 'Winner's location'. It specifies if the games was played in home ('H'), no-home ('A') and neutral ('N').
# Doesn't make sense to keep filled each 'Loser' register.

# In[ ]:


melted_results['WLoc'] = np.where(melted_results['game_status']== 'Loser' ,
                                 np.nan,
                                 melted_results['WLoc'])


# # working with seasons
# From this table we can estimate the exactly date for each game.

# In[ ]:


seasons = pd.read_csv(os.path.join(container_dir, 'MSeasons.csv'))


# In[ ]:


#Dayzero to datetime
seasons['DayZero'] = pd.to_datetime(seasons.DayZero)
day_zeros = seasons.loc[:, ['Season', 'DayZero']].copy()
results = melted_results.merge(day_zeros, on='Season', how='inner')

#adding game date:
results['date'] = results.DayZero + pd.to_timedelta(results.DayNum, unit='day')


# # working with cities
# From this dataframe we can extract city and state.
# 
# __first__
# we need to merge it with game cities to obtain a __game detail level__

# In[ ]:


# loading cities
cities = pd.read_csv(os.path.join(container_dir, 'Cities.csv'))
cities['city_state'] = cities.City + '-' + cities.State

#loading games cities
game_cities = pd.read_csv(os.path.join(container_dir, 'MGameCities.csv'))


# In[ ]:


game_cities = game_cities.merge(
    cities,
    on = 'CityID',
    how = 'outer'
)
game_cities.drop('CityID', axis=1, inplace=True)


# In[ ]:


game_cities.rename(columns = {'WTeamID':'Winner', 'LTeamID':'Loser'}, inplace=True)
value_vars = ['Winner', 'Loser']
id_vars = [i for i in game_cities.columns if i not in value_vars]

game_cities_melt = game_cities.melt(
                    id_vars = id_vars,
                    value_vars = value_vars,
                    var_name = 'game_status',
                    value_name = 'TeamID'
                    
)


# In[ ]:


game_cities.Season.min() #note we have registers since 2010 so there will be many missing values


# In[ ]:


#merging results with game cities
results = results.merge(game_cities_melt, on=['Season', 'DayNum', 'game_status', 'TeamID'], how='outer')
results.columns = results.columns.str.lower()


# In[ ]:


results.head()


# Next step depends on your preferences so I wrote it as a function. You can get rid of that if you want.

# In[ ]:


def create_derived_cols(games):
    key_days = [
                games['daynum'].between(134, 135), games['daynum'].between(136, 137),
                games['daynum'].between(138, 139),games['daynum'].between(143, 144),
                games['daynum'].between(145, 146),games['daynum'] == 152,
                games['daynum'] == 154

               ]

    key_meanings = [
                'Play In', '64 to 32',
                '32 to 16','Sweet Sixteen',
                'Elite Eight', 'Final Four',
                'Nacional Final'

               ]

    games['season_progress'] = 'regular'
    for days, meaning in zip(key_days, key_meanings):
        games['season_progress'] = np.where(
            days, meaning, games['season_progress']
    )

    games['season_progress'] = np.where(
        games.identifier == 'secondary',
        'secondary',
        games['season_progress']
    )

    champ_condition = (games.daynum == 154) &                        (games.game_status == 'Winner')
    games['champion'] = np.where(champ_condition, 1, 0)

    champions = games.loc[games.champion==1, ['season', 'teamid', 'champion']].set_index(['season', 'teamid'])
    games['champion'] = games.set_index(['season', 'teamid']).index.map(champions.champion).fillna(0)

    games['games_played'] = games.loc[games.identifier == 'regular'].groupby(['season', 'teamid']).cumcount() + 1

    # Cumulative loses Till next victory
    games.sort_values(['season', 'teamid', 'daynum'], inplace=True)
    winners_cumsum = (games.game_status == 'Winner').cumsum()
    games['cum_loses'] = games.groupby(winners_cumsum).cumcount()
    condition = (games.game_status == 'Loser') & (games.cum_loses == 0)
    games['cum_loses'] = np.where(condition, 1, games.cum_loses)

    # Final stage achieved for current team in current season
    games['final_stage'] = games.groupby(['season', 'teamid'])['season_progress'].transform('last')
    games['final_stage'] = pd.Categorical(
                                games['final_stage'],
                                categories=['regular']+key_meanings,
                                ordered=True
    )
    games['final_stage'].fillna('regular', inplace=True)

    games.sort_values(['final_stage','games_played','season'], inplace=True)
    
    # basic measurments
    games['off_ratio'] = games['or'] / (games['or']+games.dr)
    games['def_ratio'] = games['dr'] / (games['or']+games.dr)
    games['fgm_efectuation'] = games['fgm'] / games['fga']
    games['fgm_efectuation3'] = games['fgm3'] / games['fga3']

    games['score_diff'] = np.where(games.game_status == 'Loser',
                                  games.score_diff * (-1),
                                  games.score_diff)
    games['lose_status'] = games.game_status == 'Loser'
    
    return games


# In[ ]:


results = create_derived_cols(results)


# ## tidy dataframe

# In[ ]:


results.sort_values('city_state').head()


# # If you find it useful please don't forget __upvote__ and please give me your __feedback__
# 
