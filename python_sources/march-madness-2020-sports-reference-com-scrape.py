#!/usr/bin/env python
# coding: utf-8

# # Pull NCAA Basketball Player Data
# - Pull data sports-reference.com and format.

# In[ ]:


import pandas as pd
import requests
from tqdm.notebook import tqdm
pd.set_option('max_columns', 500)


# In[ ]:


MENS_PBP_DIR = '../input/march-madness-analytics-2020/MPlayByPlay_Stage2'

MPlayers = pd.read_csv(f'{MENS_PBP_DIR}/MPlayers.csv', error_bad_lines=False)
MTeamSpelling = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MTeamSpellings.csv',
                            engine='python')
MTeams = pd.read_csv('../input/march-madness-analytics-2020/MDataFiles_Stage2/MTeams.csv')
MPlayers = MPlayers.merge(MTeams[['TeamID','TeamName']], on='TeamID', how='left')


# In[ ]:


MTeamSpelling['SpellingLen'] = MTeamSpelling['TeamNameSpelling'].str.len()
# Sort team names by spelling lenghth (longer first)
MTeamSpelling = MTeamSpelling     .sort_values('SpellingLen', ascending=True)     .sort_values('TeamID')


# # Determine the Team URL Names
# - Loop through each proposed team name and check if the url works.
# - Replace spaces with dash '-'
# - Store a DataFrame with the team url name

# In[ ]:


SR_Team_URL_df = pd.DataFrame()
for d in tqdm(MTeamSpelling.itertuples(), total=len(MTeamSpelling)):
    t = d[1]
    t_id = d[2]
    t_url = t.lower().replace(' ','-')
    url = f'https://www.sports-reference.com/cbb/schools/{t_url}/'
    SR_Team_URL_df.loc[t_url, 'TeamID'] = t_id
    request = requests.get(url)
    if request.status_code == 200:
        SR_Team_URL_df.loc[t_url, 'isURL'] = True
    else:
        SR_Team_URL_df.loc[t_url, 'isURL'] = False


# # Determine Teams where no correct name was found
# - Manually find the corrent url name for these missing teams and add them to the dataframe.

# In[ ]:


isMissing = SR_Team_URL_df.groupby('TeamID')['isURL'].sum() == 0
missingTeamIds = SR_Team_URL_df.groupby('TeamID')['isURL'].sum().loc[isMissing].index.tolist()
print(len(missingTeamIds))


# In[ ]:


# Two Teams missing
MTeams.loc[MTeams['TeamID'].isin(missingTeamIds)]


# In[ ]:


SR_Team_URL_df.query('TeamID == 1446')


# In[ ]:


# I can't find this school on the website.
SR_Team_URL_df.query('TeamID == 1445')


# In[ ]:


SR_Team_URL_df.loc['west-texas-am', 'TeamID'] = 1446
SR_Team_URL_df.loc['west-texas-am', 'isURL'] = True


# In[ ]:


SR_Team_URL_df = SR_Team_URL_df.reset_index().rename(columns= {'index':'TeamNameSpelling'})


# In[ ]:


SR_Team_URL_df.to_csv('SRTeamURL.csv', index=False)


# # Loop Through Teams Season 2015-2019
# - Data:
#     - Player Roster (Height, Weight, Etc info)
#     - Player Per-Game Stats (Minutes Played, Shot % etc)
#     - Team Level Stats (PPG, etc)

# In[ ]:


seasons = [2015, 2016, 2017, 2018, 2019]
team_ids = MTeams['TeamID'].unique()
rosters = []
team_stats = []
per_game_stats = []
worked_url_team = []
test_run = False
count = 0
for t_id in tqdm(team_ids, total=len(team_ids)):
    t_name_proper = MTeams.query('TeamID == @t_id')['TeamName'].values[0]
    try:
        t_url = SR_Team_URL_df.query('isURL and TeamID == @t_id')['TeamNameSpelling'].values[0]
    except:
        t_url = '-'
    for s in seasons:
        try:
            data = pd.read_html(f'https://www.sports-reference.com/cbb/schools/{t_url}/{s}.html')
            ros = data[0]
            team_opp = data[1]
            per_game = data[2]
            ros['TeamName'] = t_name_proper
            ros['TeamID'] = t_id
            ros['Season'] = s

            team_opp['TeamName'] = t_name_proper
            team_opp['TeamID'] = t_id
            team_opp['Season'] = s

            per_game['TeamName'] = t_name_proper
            per_game['TeamID'] = t_id
            per_game['Season'] = s

            rosters.append(ros)
            team_stats.append(team_opp)
            per_game_stats.append(per_game)
            worked_url_team.append(t)
            print(f'Worked for {t_name_proper} {s} - using url name: {t_url}')
        except Exception as e:
            print(f'Broke for {t} {s}  - using url name: {t_url} - with exception: {e}')
    count += 1
    if test_run and count == 10:
        break


# # Combine Results and Save

# ## Rosters

# In[ ]:


# Combine and Save
rosters_df = pd.concat(rosters, sort=False)
rosters_df['FirstName'] = rosters_df['Player'].str.split(' ', expand=True)[0]
rosters_df['LastName'] = rosters_df['Player'].str.split(' ', expand=True)[1]
rosters_df['AdditionalName'] = rosters_df['Player'].str.split(' ', expand=True)[2]
# rosters_df = rosters_df.loc[~((rosters_df['Player'] == "Georgi Funtarov") & (rosters_df['Class'].isna()))]
# rosters_df = rosters_df.loc[~((rosters_df['Player'] == "Joseph Battle") & (rosters_df['Class'].isna()))]
# Order Columns
col = rosters_df.pop("AdditionalName")
rosters_df.insert(0, col.name, col)
col = rosters_df.pop("LastName")
rosters_df.insert(0, col.name, col)
col = rosters_df.pop("FirstName")
rosters_df.insert(0, col.name, col)
col = rosters_df.pop("TeamName")
rosters_df.insert(0, col.name, col)
col = rosters_df.pop("TeamID")
rosters_df.insert(0, col.name, col)
col = rosters_df.pop("Season")
rosters_df.insert(0, col.name, col)
col = rosters_df.pop("Player")
rosters_df.insert(0, col.name, col)
rosters_df = rosters_df.reset_index(drop=True)
rosters_df.to_csv('SRRosters.csv', index=False)
rosters_df.to_parquet('SRRosters.parquet')


# In[ ]:


rosters_df.head()


# ## Team Season Stats

# In[ ]:


team_df = pd.concat(team_stats, sort=False)
# Order Columns
col = team_df.pop("Season")
team_df.insert(0, col.name, col)
col = team_df.pop("TeamID")
team_df.insert(0, col.name, col)
col = team_df.pop("TeamName")
team_df.insert(0, col.name, col)
team_df = team_df.rename(columns={'Unnamed: 0': 'Team/Opp'})
team_df.loc[1, 'Team/Opp'] = 'Team_Rank'
team_df.loc[3, 'Team/Opp'] = 'Opponent_Rank'
team_df = team_df.reset_index(drop=True)
team_df.to_csv('SRTeamSeasonStats.csv', index=False)
team_df.to_parquet('SRTeamSeasonStats.parquet')


# In[ ]:


team_df.head()


# ## Per Game Player/Season Stats

# In[ ]:


per_game_df = pd.concat(per_game_stats, sort=False)
per_game_df['FirstName'] = per_game_df['Player'].str.split(' ', expand=True)[0]
per_game_df['LastName'] = per_game_df['Player'].str.split(' ', expand=True)[1]
per_game_df['AdditionalName'] = per_game_df['Player'].str.split(' ', expand=True)[2]
# team_df = team_df.loc[~((team_df['Player'] == "Georgi Funtarov") & (team_df['Class'].isna()))]
# team_df = team_df.loc[~((team_df['Player'] == "Joseph Battle") & (team_df['Class'].isna()))]
col = per_game_df.pop("AdditionalName")
per_game_df.insert(0, col.name, col)
col = per_game_df.pop("LastName")
per_game_df.insert(0, col.name, col)
col = per_game_df.pop("FirstName")
per_game_df.insert(0, col.name, col)
col = per_game_df.pop("TeamName")
per_game_df.insert(0, col.name, col)
col = per_game_df.pop("TeamID")
per_game_df.insert(0, col.name, col)
col = per_game_df.pop("Season")
per_game_df.insert(0, col.name, col)
col = per_game_df.pop("Player")
per_game_df.insert(0, col.name, col)

per_game_df = per_game_df.reset_index(drop=True)
per_game_df.to_csv('SRPerGameStats.csv', index=False)
per_game_df.to_parquet('SRPerGameStats.parquet')


# In[ ]:


per_game_df.head()


# In[ ]:




