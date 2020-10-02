#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Make sure to turn Internet ON!


# In[ ]:


# Kaggle Limits (https://www.kaggle.com/c/halite/discussion/164932)
#    1000 requests per day max;  Rate limits are shared between the ListEpisodes and GetEpisodeReplay endpoints
#    Exceeding limits repeatedly will lead to temporary and then permanent bans


# In[ ]:


NUM_TEAMS = 5
EPISODES = 450

MIN_FINAL_RATING = 1200
MIN_AVG_RATING = 1050


# In[ ]:





# In[ ]:





# In[ ]:



MAX_AVG_RATING = None


# In[ ]:





# ### Initialize

# In[ ]:


# !pip install kaggle-environments --upgrade


# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


import requests
import json
from zipfile import ZipFile


# In[ ]:


import datetime
import time


# In[ ]:


import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10, 4)


# In[ ]:


BUFFER = 1


# In[ ]:


base_url = "https://www.kaggle.com/requests/EpisodeService/"
get_url = base_url + "GetEpisodeReplay"
list_url = base_url + "ListEpisodes"


# In[ ]:





# ### Get Initial Team List

# In[ ]:


r = requests.post(list_url, json = {"teamId":  4820508})

rj = r.json()


# In[ ]:


teams_df = pd.DataFrame(rj['result']['teams'])


# In[ ]:


teams_df.sort_values('publicLeaderboardRank', inplace = True)
teams_df.head()


# In[ ]:


# plt.plot(sorted(teams_df.publicLeaderboardRank.unique()));


# In[ ]:


# teams_df


# In[ ]:


# teams_df[teams_df.teamName == 'Team VQ']


# In[ ]:





# In[ ]:


def getTeamEpisodes(team_id):
    # request
    r = requests.post(list_url, json = {"teamId":  int(team_id)})
    rj = r.json()

    # update teams list
    global teams_df
    teams_df_new = pd.DataFrame(rj['result']['teams'])
    
    if len(teams_df.columns) == len(teams_df_new.columns) and (teams_df.columns == teams_df_new.columns).all():
        teams_df = pd.concat( (teams_df, teams_df_new.loc[[c for c in teams_df_new.index if c not in teams_df.index]] ) )
        teams_df.sort_values('publicLeaderboardRank', inplace = True)
#         print('{} teams on file'.format(len(teams_df)))
    else:
        print('teams dataframe did not match')
    
    # make df
    team_episodes = pd.DataFrame(rj['result']['episodes'])
    team_episodes['avg_score'] = -1;
    
    for i in range(len(team_episodes)):
        agents = team_episodes['agents'].loc[i]
        agent_scores = [a['updatedScore'] for a in agents if a['updatedScore'] is not None]
        team_episodes.loc[i, 'submissionId'] = [a['submissionId'] for a in agents if a['submission']['teamId'] == team_id][0]
        team_episodes.loc[i, 'updatedScore'] = [a['updatedScore'] for a in agents if a['submission']['teamId'] == team_id][0]
        
        if len(agent_scores) > 0:
            team_episodes.loc[i, 'avg_score'] = np.mean(agent_scores)

    for sub_id in team_episodes['submissionId'].unique():
        sub_rows = team_episodes[ team_episodes['submissionId'] == sub_id ]
        max_time = max( [r['seconds'] for r in sub_rows['endTime']] )
        final_score = max( [r['updatedScore'] for r_idx, (r_index, r) in enumerate(sub_rows.iterrows())
                                if r['endTime']['seconds'] == max_time] )

        team_episodes.loc[sub_rows.index, 'final_score'] = final_score
        
    team_episodes.sort_values('avg_score', ascending = False, inplace=True)
    return rj, team_episodes


# In[ ]:





# In[ ]:


# %load_ext line_profiler


# In[ ]:


# %lprun -f getTeamEpisodes getTeamEpisodes(5118779)


# In[ ]:





# In[ ]:


# te = getTeamEpisodes(4714287)


# In[ ]:


# te[1].head()


# In[ ]:





# In[ ]:





# In[ ]:


def saveEpisode(epid, rj):
    # request
    re = requests.post(get_url, json = {"EpisodeId": int(epid)})
    
    # save replay
    with open('{}.json'.format(epid), 'w') as f:
        f.write(re.json()['result']['replay'])

    # save episode info
    with open('{}_info.json'.format(epid), 'w') as f:
        json.dump([r for r in rj['result']['episodes'] if r['id']==epid][0], f)


# In[ ]:





# ### Library

# In[ ]:


all_files = []
for root, dirs, files in os.walk('../input/', topdown=False):
    all_files.extend(files)


# In[ ]:


seen_episodes = [int(f.split('.')[0]) for f in all_files 
                      if '.' in f and f.split('.')[0].isdigit() and f.split('.')[1] == 'json']


# In[ ]:


print('{} games in existing library'.format(len(seen_episodes)))


# In[ ]:





# ### Scraper

# In[ ]:


pulled_teams = {}
pulled_episodes = []
start_time = datetime.datetime.now()
r = BUFFER;

while len(pulled_episodes) < EPISODES:
    # pull team
    top_teams = [i for i in teams_df.id if i not in pulled_teams]
    if len(top_teams) > 0:
        team_id = top_teams[0]
    else:
        break;
        
    # get team data
    team_json, team_df = getTeamEpisodes(team_id); r+=1;
    print('{} games for {}'.format(len(team_df), teams_df.loc[teams_df.id == team_id].iloc[0].teamName))

    
    team_df = team_df[  (MIN_FINAL_RATING is None or (team_df.final_score > MIN_FINAL_RATING))
        
                         &   (MIN_AVG_RATING is None or (team_df.avg_score > MIN_AVG_RATING))
                        & ( MAX_AVG_RATING is None or (team_df.avg_score < MAX_AVG_RATING) )  ]
    
    print('   {} in score range from {} submissions'.format(len(team_df), len(team_df.submissionId.unique() ) ) )
    
    team_df = team_df[~team_df.id.isin(pulled_episodes + seen_episodes)]        
    print('      {} remain to be downloaded\n'.format(len(team_df)))
    
    if len(team_df) > 0:
        plt.hist(team_df.avg_score, bins = 150);
    
    
    # pull games
    target_team_games = int(np.ceil(EPISODES / NUM_TEAMS))
    if target_team_games + len(pulled_episodes) > EPISODES:
        target_team_games = EPISODES - len(pulled_episodes)
     
    pulled_teams[team_id] = 0
#     continue;
    
    i = 0
    while i < len(team_df) and pulled_teams[team_id] < target_team_games:
        epid = team_df.id.iloc[i]
        if not (epid in pulled_episodes or epid in seen_episodes):
            try:
                saveEpisode(epid, team_json); r+=1;
            except:
                time.sleep(20)
                i+=1;
                continue;
                
            pulled_episodes.append(epid)
            pulled_teams[team_id] += 1
            try:
                size = os.path.getsize('{}.json'.format(epid)) / 1e6
                print('Saved Episode #{} @ {:.1f}MB'.format(epid, size))
            except:
                print('  file {}.json did not seem to save'.format(epid))    
            if r > (datetime.datetime.now() - start_time).seconds:
                time.sleep( r - (datetime.datetime.now() - start_time).seconds)
                

        i+=1;
    print(); print()


# In[ ]:


print('\n   Post and share your datasets publicly on this thread:')
print('        https://www.kaggle.com/c/halite/discussion/164932\n')


# In[ ]:





# ### Check JSON

# In[ ]:


file = './{}.json'.format(epid)


# In[ ]:


with open(file) as f:
    data = json.load(f)


# In[ ]:


data['rewards']


# In[ ]:




