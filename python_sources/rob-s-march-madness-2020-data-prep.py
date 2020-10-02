#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing
# This notebook contains data preparation and preprocessing for my 2020 NCAA Analytics report. It includes:
# 
# - Preparing the Kaggle Events and Player Datasets
# - Merging with existing data from Sports Reference
# - Creating a dataset where each row represents a player/season combination. Each of these rows contains data about the player for that given season.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
from tqdm.notebook import tqdm

import time
sns.set(rc={'figure.figsize':(15, 5)})
palette = sns.color_palette("bright", 10)

pd.set_option('max_columns', 500)
pd.set_option('max_rows', 500)
pd.options.display.float_format = '{:.2f}'.format
# plt.rcParams['figure.figsize'] = (15.0, 5.0)


# ## Player Event Data Prep

# In[ ]:


MENS_PBP_DIR = '../input/march-madness-analytics-2020/MPlayByPlay_Stage2'

MPlayers = pd.read_csv(f'{MENS_PBP_DIR}/MPlayers.csv', error_bad_lines=False)
MTeamSpelling = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MTeamSpellings.csv',
                            engine='python')
mens_events = []
for year in [2015, 2016, 2017, 2018, 2019]:
    mens_events.append(pd.read_csv(f'{MENS_PBP_DIR}/MEvents{year}.csv'))
MEvents = pd.concat(mens_events)

MTeams = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MTeams.csv')
MPlayers = MPlayers.merge(MTeams[['TeamID','TeamName']], on='TeamID', how='left')

area_mapping = {0: np.nan,
                1: 'under basket',
                2: 'in the paint',
                3: 'inside right wing',
                4: 'inside right',
                5: 'inside center',
                6: 'inside left',
                7: 'inside left wing',
                8: 'outside right wing',
                9: 'outside right',
                10: 'outside center',
                11: 'outside left',
                12: 'outside left wing',
                13: 'backcourt'}

MEvents['Area_Name'] = MEvents['Area'].map(area_mapping)

# Normalize X, Y positions for court dimentions
# Court is 50 feet wide and 94 feet end to end.
MEvents['X_'] = (MEvents['X'] * (94/100))
MEvents['Y_'] = (MEvents['Y'] * (50/100))

# Merge Player name onto events
MEvents = MEvents.merge(MPlayers,
              how='left',
              left_on='EventPlayerID',
              right_on='PlayerID')

# Only Look at Events with Player assoicated
MPlayerEvents = MEvents.query('EventPlayerID != 0')

# Create GameId
MPlayerEvents['GameId'] =     MPlayerEvents['Season'].astype('str') + '_' +     MPlayerEvents['DayNum'].astype('str') + '_' +     MPlayerEvents['WTeamID'].astype('str') + '_' +     MPlayerEvents['LTeamID'].astype('str')

# Create Unique Player Season Combo
EventPlayerSeasonCombo = MPlayerEvents[['EventPlayerID','Season']].drop_duplicates().reset_index(drop=True)

# Expand MPlayers to have a row for each player
MPlayerSeason = MPlayers.merge(EventPlayerSeasonCombo,
               left_on=['PlayerID'],
               right_on=['EventPlayerID'],
              validate='1:m') \
    .drop('EventPlayerID', axis=1)


# Create Player/Season Feature
MPlayerSeason['PlayerID_Season'] = MPlayerSeason['PlayerID'].astype('int').astype('str')     + '_' + MPlayerSeason['Season'].astype('str')
MPlayerEvents['PlayerID_Season'] = MPlayerEvents['PlayerID'].astype('int').astype('str')     + '_' + MPlayerEvents['Season'].astype('str')


# ## Create Features for Each Player/Season based on Events Data

# In[ ]:


#################################################
# Add Features to MPlayers
# - Each player/season combo has it's own "stats" as features
#################################################

MPlayerSeason['GameCount'] = MPlayerSeason['PlayerID_Season'].map(
    MPlayerEvents \
    .groupby('PlayerID_Season')['GameId'] \
    .nunique() \
    .to_dict())

event_count = MPlayerEvents     .groupby('PlayerID_Season')['EventID']     .count()     .to_dict()
MPlayerSeason['EventCount'] = MPlayerSeason['PlayerID_Season']     .map(event_count) # Number of Events for this player

# Shot Counts
MPlayerSeason['Miss3Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "miss3"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())
MPlayerSeason['Made3Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "made3"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())
MPlayerSeason['Miss2Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "miss2"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())
MPlayerSeason['Made2Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "made2"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())
MPlayerSeason['Miss1Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "miss1"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())
MPlayerSeason['Made1Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "made1"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())

# Shots Totals
MPlayerSeason['3Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "miss3" or EventType == "miss3"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())
MPlayerSeason['2Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "miss2" or EventType == "made2"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())
MPlayerSeason['1Count'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents.query('EventType == "miss1" or EventType == "made1"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())

MPlayerSeason['ShotCount'] = MPlayerSeason['PlayerID_Season'].map( 
    MPlayerEvents \
    .query('EventType == "miss3" or EventType == "miss3" or EventType == "miss2" or EventType == "made2"') \
    .groupby('PlayerID_Season')['EventID'] \
    .count() \
    .to_dict())

# Games Played by player
MPlayerSeason['GameCount'] = MPlayerSeason['PlayerID_Season'].map(
    MPlayerEvents \
    .groupby('PlayerID_Season')['GameId'] \
    .nunique() \
    .to_dict())

# Areas where shots were taken
area_list = ['inside left', 'backcourt', 'under basket', 'outside center',
       'outside left', 'outside right', 'in the paint', 'inside right',
       'inside center', 'outside left wing', 'outside right wing',
       'inside left wing', 'inside right wing']

# Shot Counts by Area
for a in area_list:
    MPlayerSeason['Miss3Count'+a] = MPlayerSeason['PlayerID_Season'].map( 
        MPlayerEvents.query('EventType == "miss3" and Area_Name == @a') \
        .groupby('PlayerID_Season')['EventID'] \
        .count() \
        .to_dict())
    MPlayerSeason['Made3Count'+a] = MPlayerSeason['PlayerID_Season'].map( 
        MPlayerEvents.query('EventType == "made3" and Area_Name == @a') \
        .groupby('PlayerID_Season')['EventID'] \
        .count() \
        .to_dict())
    MPlayerSeason['Miss2Count'+a] = MPlayerSeason['PlayerID_Season'].map( 
        MPlayerEvents.query('EventType == "miss2" and Area_Name == @a') \
        .groupby('PlayerID_Season')['EventID'] \
        .count() \
        .to_dict())
    MPlayerSeason['Made2Count'+a] = MPlayerSeason['PlayerID_Season'].map( 
        MPlayerEvents.query('EventType == "made2" and Area_Name == @a') \
        .groupby('PlayerID_Season')['EventID'] \
        .count() \
        .to_dict())
    MPlayerSeason['ShotCount'+a] = MPlayerSeason['PlayerID_Season'].map( 
        MPlayerEvents.query('Area_Name == @a') \
        .query('EventType == "made2" or EventType == "miss2" or EventType == "made3" or EventType == "miss3"')
        .groupby('PlayerID_Season')['EventID'] \
        .count() \
        .to_dict())
    
    
# Make Everything with respect to percent of games
stat_cols = ['Miss3Count', 'Made3Count', 'Miss2Count', 'Made2Count', 'Miss1Count',
               'Made1Count', 'Miss3Countinside left',
               'Made3Countinside left', 'Miss2Countinside left',
               'Made2Countinside left', 'Miss3Countbackcourt',
               'Made3Countbackcourt',
               'Miss2Countbackcourt', 'Made2Countbackcourt',
               'Miss3Countunder basket',
               'Made3Countunder basket', 'Miss2Countunder basket',
               'Made2Countunder basket', 'Miss3Countoutside center',
               'Made3Countoutside center', 'Miss2Countoutside center',
               'Made2Countoutside center', 'Miss3Countoutside left',
               'Made3Countoutside left', 'Miss2Countoutside left',
               'Made2Countoutside left', 'Miss3Countoutside right',
               'Made3Countoutside right', 'Miss2Countoutside right',
               'Made2Countoutside right', 'Miss3Countin the paint',
               'Made3Countin the paint', 'Miss2Countin the paint',
               'Made2Countin the paint', 'Miss3Countinside right',
               'Made3Countinside right', 'Miss2Countinside right',
               'Made2Countinside right', 'Miss3Countinside center',
               'Made3Countinside center', 'Miss2Countinside center',
               'Made2Countinside center', 'Miss3Countoutside left wing',
               'Made3Countoutside left wing', 'Miss2Countoutside left wing',
               'Made2Countoutside left wing', 'Miss3Countoutside right wing',
               'Made3Countoutside right wing', 'Miss2Countoutside right wing',
               'Made2Countoutside right wing', 'Miss3Countinside left wing',
               'Made3Countinside left wing', 'Miss2Countinside left wing',
               'Made2Countinside left wing', 'Miss3Countinside right wing',
               'Made3Countinside right wing', 'Miss2Countinside right wing',
               'Made2Countinside right wing', '3Count', '2Count', '1Count',
               'ShotCount', 'ShotCountinside left', 'ShotCountbackcourt',
               'ShotCountunder basket', 'ShotCountoutside center',
               'ShotCountoutside left', 'ShotCountoutside right',
               'ShotCountin the paint', 'ShotCountinside right',
               'ShotCountinside center', 'ShotCountoutside left wing',
               'ShotCountoutside right wing', 'ShotCountinside left wing',
               'ShotCountinside right wing']

for s in tqdm(stat_cols):
    MPlayerSeason[s+'_per_game'] = MPlayerSeason[s] / MPlayerSeason['GameCount']

MPlayerSeason = MPlayerSeason.fillna(0)
    
# Filter Down Player Data to ones with acutal playing time
MPlayers_filtered = MPlayerSeason.query('GameCount > 5 and EventCount >= 0').copy()

# List of features for players by "per game"
per_game_cols = [c for c in MPlayers_filtered.columns if 'per_game' in c]


# # Merge Kaggle Data with Sports Reference Data
# - Created in this notebook: https://www.kaggle.com/robikscube/march-madness-2020-sports-reference-com-scrape
# - Available in this dataset: https://www.kaggle.com/robikscube/ncaa-data-20152019-sportsreference

# In[ ]:


SRRoster = pd.read_csv('../input/ncaa-data-20152019-sportsreference/SRRosters.csv')
SRPerGameStats = pd.read_csv('../input/ncaa-data-20152019-sportsreference/SRPerGameStats.csv')


# In[ ]:


SRPerGameStats = SRPerGameStats.drop([20303, 20563]).reset_index(drop=True)
SRRoster = SRRoster.drop([20303, 20567]).reset_index(drop=True)


# In[ ]:


SRRosterStats = SRRoster.merge(SRPerGameStats, on=['Player','Season','TeamID','TeamName',
                                   'FirstName','LastName','AdditionalName'],
               validate='1:1')

MPlayers_filtered = MPlayerSeason.query('GameCount > 5 and EventCount >= 0').copy()
MPlayers['Player'] = MPlayers['FirstName'] + ' ' + MPlayers['LastName']
MPlayerSeason['Player'] = MPlayerSeason['FirstName'] + ' ' + MPlayerSeason['LastName']
MPlayers_filtered['Player'] = MPlayers_filtered['FirstName'] + ' ' + MPlayers_filtered['LastName']


# ## Merge based on fuzzy matching of names
# - Some manual cleaning was required to account for matching inconsistent spelling and abbreviations of names.

# In[ ]:


MPlayers_filtered['Player_lower'] = MPlayers_filtered['Player']     .str.lower()     .str.replace('.','')     .str.replace('jr','')     .str.replace('sr','')     .str.replace('jr.','')     .str.replace('sr.','')     .str.replace('iii','')     .str.replace('swan-ford','swan')     .str.replace('christopher','chris')     .str.replace('rakiya','rakia')     .str.replace('tracey','tracy')     .str.replace(' ','')     .str.replace('michael','mike')     .str.replace('reginald','reginal')     .str.replace('herbert','herb')     .str.replace('freeman-daniels', 'daniels')     .str.replace('allerik', 'al')     .str.replace('tjmaston', 'terrymaston')     .str.replace('coreybarnes','jrbarnes')     .str.replace('lorencristian','loren')     .str.replace('peewee','darius')     .str.replace("joe'randlecunningham","joe'randletoliver")     .str.replace("natewells","wellsnate")     .str.replace('mrdjangasevic','gasevicmrdjan')     .str.replace('letrellviser','snoopviser')     .str.replace('alexennis','ennisalex')     .str.replace('azariahsykes','sykesazariah')     .str.replace('stanleydavis','davisstanley')     .str.replace('byerstyjhai','tyjhaibyers')     .str.replace('ocheneyoleakuwovo','akuwovoogheneyole')     .str.replace('baebaedaniels','demarcusdaniels')     .str.replace('tarekeyi','tk')     .str.replace('elishaboone','booneelisha')     .str.replace("ja'kwanjones",'slinkyjones')     .str.replace('rodneyhawkins','hawkinsrodney')     .str.replace('patrickkirksey','kenshaykirksey')     .str.replace('maxhoetzel','maxmontana')     .str.replace('jahmelbodrick','bodrickjahmel')     .str.replace('giacomozilli','zilligiacomo')     .str.replace('cameroncurry','cc')

SRRosterStats['Player_lower'] = SRRosterStats['Player']     .str.lower()     .str.replace('.','')     .str.replace('jr','')     .str.replace('sr','')     .str.replace('jr.','')     .str.replace('sr.','')     .str.replace('iii','')     .str.replace('swan-ford','swan')     .str.replace('christopher','chris')     .str.replace('rakiya','rakia')     .str.replace('tracey','tracy')     .str.replace(' ','')     .str.replace('michael','mike')     .str.replace('reginald','reginal')     .str.replace('herbert','herb')     .str.replace('freeman-daniels', 'daniels')     .str.replace('allerik', 'al')     .str.replace('tjmaston', 'terrymaston')     .str.replace('coreybarnes','jrbarnes')     .str.replace('lorencristian','loren')     .str.replace('peewee','darius')     .str.replace("joe'randlecunningham","joe'randletoliver")     .str.replace("natewells","wellsnate")     .str.replace('mrdjangasevic','gasevicmrdjan')     .str.replace('letrellviser','snoopviser')     .str.replace('alexennis','ennisalex')     .str.replace('azariahsykes','sykesazariah')     .str.replace('stanleydavis','davisstanley')     .str.replace('byerstyjhai','tyjhaibyers')     .str.replace('ocheneyoleakuwovo','akuwovoogheneyole')     .str.replace('baebaedaniels','demarcusdaniels')     .str.replace('tarekeyi','tk')     .str.replace('elishaboone','booneelisha')     .str.replace("ja'kwanjones",'slinkyjones')     .str.replace('rodneyhawkins','hawkinsrodney')     .str.replace('patrickkirksey','kenshaykirksey')     .str.replace('maxhoetzel','maxmontana')     .str.replace('jahmelbodrick','bodrickjahmel')     .str.replace('giacomozilli','zilligiacomo')     .str.replace('cameroncurry','cc')

MPlayers_filtered_roster = MPlayers_filtered         .merge(SRRosterStats,
               how='left',
               on=['Player_lower','TeamID','Season'],
               suffixes=('',f'_{s}'),
               validate='m:1')

# Remove Positions that are not common and make more general
MPlayers_filtered_roster['Pos'] = MPlayers_filtered_roster['Pos'].str.replace('PG','G')     .replace('SF','F')     .replace('SG','G')     .replace('PF','F')

count_not_merged = MPlayers_filtered_roster['Pos'].isna().sum()
count_playerseason = len(MPlayers_filtered_roster)
print(f'{count_not_merged} of {count_playerseason} not merged')
print(f'{count_not_merged*100/count_playerseason:0.4f}% missing')


# In[ ]:


# Fuzzy matching of names round two
# These remaining ones that don't match use the second closest match
from difflib import get_close_matches


players_unmatched_df = MPlayers_filtered_roster.loc[MPlayers_filtered_roster['Pos'].isna()][['Player_lower','TeamName','Season']]

fuzzy_match_dict = {}
for i, row in players_unmatched_df.iterrows():
    p = row['Player_lower']
    s = row['Season']
    t = row['TeamName']
    roster_names = SRRosterStats.query('Season == @s and TeamName == @t')['Player_lower'].unique().tolist()
    try:
        closest_match = get_close_matches(p, roster_names)[0]
        # print(p, closest_match)
        fuzzy_match_dict[p] = closest_match
    except:
        pass
#             print(f'Broke for {p}')


# In[ ]:


# Include Fuzzy Matching
MPlayers_filtered['Player_lower'] = MPlayers_filtered['Player']     .str.lower()     .str.replace('.','')     .str.replace('jr','')     .str.replace('sr','')     .str.replace('jr.','')     .str.replace('sr.','')     .str.replace('iii','')     .str.replace('swan-ford','swan')     .str.replace('christopher','chris')     .str.replace('rakiya','rakia')     .str.replace('tracey','tracy')     .str.replace(' ','')     .str.replace('michael','mike')     .str.replace('reginald','reginal')     .str.replace('herbert','herb')     .str.replace('freeman-daniels', 'daniels')     .str.replace('allerik', 'al')     .str.replace('tjmaston', 'terrymaston')     .str.replace('coreybarnes','jrbarnes')     .str.replace('lorencristian','loren')     .str.replace('peewee','darius')     .str.replace("joe'randlecunningham","joe'randletoliver")     .str.replace("natewells","wellsnate")     .str.replace('mrdjangasevic','gasevicmrdjan')     .str.replace('letrellviser','snoopviser')     .str.replace('alexennis','ennisalex')     .str.replace('azariahsykes','sykesazariah')     .str.replace('stanleydavis','davisstanley')     .str.replace('byerstyjhai','tyjhaibyers')     .str.replace('ocheneyoleakuwovo','akuwovoogheneyole')     .str.replace('baebaedaniels','demarcusdaniels')     .str.replace('tarekeyi','tk')     .str.replace('elishaboone','booneelisha')     .str.replace("ja'kwanjones",'slinkyjones')     .str.replace('rodneyhawkins','hawkinsrodney')     .str.replace('patrickkirksey','kenshaykirksey')     .str.replace('maxhoetzel','maxmontana')     .str.replace('jahmelbodrick','bodrickjahmel')     .str.replace('giacomozilli','zilligiacomo')     .str.replace('cameroncurry','cc')

MPlayers_filtered['Player_lower'] = MPlayers_filtered['Player_lower'].replace(fuzzy_match_dict)

SRRosterStats['Player_lower'] = SRRosterStats['Player']     .str.lower()     .str.replace('.','')     .str.replace('jr','')     .str.replace('sr','')     .str.replace('jr.','')     .str.replace('sr.','')     .str.replace('iii','')     .str.replace('swan-ford','swan')     .str.replace('christopher','chris')     .str.replace('rakiya','rakia')     .str.replace('tracey','tracy')     .str.replace(' ','')     .str.replace('michael','mike')     .str.replace('reginald','reginal')     .str.replace('herbert','herb')     .str.replace('freeman-daniels', 'daniels')     .str.replace('allerik', 'al')     .str.replace('tjmaston', 'terrymaston')     .str.replace('coreybarnes','jrbarnes')     .str.replace('lorencristian','loren')     .str.replace('peewee','darius')     .str.replace("joe'randlecunningham","joe'randletoliver")     .str.replace("natewells","wellsnate")     .str.replace('mrdjangasevic','gasevicmrdjan')     .str.replace('letrellviser','snoopviser')     .str.replace('alexennis','ennisalex')     .str.replace('azariahsykes','sykesazariah')     .str.replace('stanleydavis','davisstanley')     .str.replace('byerstyjhai','tyjhaibyers')     .str.replace('ocheneyoleakuwovo','akuwovoogheneyole')     .str.replace('baebaedaniels','demarcusdaniels')     .str.replace('tarekeyi','tk')     .str.replace('elishaboone','booneelisha')     .str.replace("ja'kwanjones",'slinkyjones')     .str.replace('rodneyhawkins','hawkinsrodney')     .str.replace('patrickkirksey','kenshaykirksey')     .str.replace('maxhoetzel','maxmontana')     .str.replace('jahmelbodrick','bodrickjahmel')     .str.replace('giacomozilli','zilligiacomo')     .str.replace('cameroncurry','cc')

SRRosterStats['Player_lower'] = SRRosterStats['Player_lower'].replace(fuzzy_match_dict)

MPlayers_filtered_roster = MPlayers_filtered         .merge(SRRosterStats,
               how='left',
               on=['Player_lower','TeamID','Season'],
               suffixes=('',f'_{s}'),
               validate='m:1')

# Remove Positions that are not common and make more general
MPlayers_filtered_roster['Pos'] = MPlayers_filtered_roster['Pos'].str.replace('PG','G')     .replace('SF','F')     .replace('SG','G')     .replace('PF','F')

count_not_merged = MPlayers_filtered_roster['Pos'].isna().sum()
count_playerseason = len(MPlayers_filtered_roster)
print(f'{count_not_merged} of {count_playerseason} not merged')
print(f'{count_not_merged*100/count_playerseason:0.4f}% missing')


# In[ ]:


def parse_height(df):
    df['HeightInches'] = (pd.to_numeric(df['Height'].str.split('-', expand=True)[0]) * 12) +         (pd.to_numeric(df['Height'].str.split('-', expand=True)[1]))

    df['HeightFeet'] = (pd.to_numeric(df['Height'].str.split('-', expand=True)[0])) +         (pd.to_numeric(df['Height'].str.split('-', expand=True)[1]) / 12)    
    return df


# In[ ]:


MPlayers_filtered_roster = parse_height(MPlayers_filtered_roster)


# # Final Dataset and Save Results

# In[ ]:


MPlayers_filtered_roster.head()


# In[ ]:


MPlayers_filtered_roster.shape


# In[ ]:


MPlayers_filtered_roster.to_csv('MPlayerSeasonStats.csv',
                                 index=False)
MPlayers_filtered_roster.to_parquet('MPlayerSeasonStats.parquet')

