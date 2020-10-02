#!/usr/bin/env python
# coding: utf-8

# ## Predicting match scores based on teams' attacking and defensive strength
# Fantasy Premier League can be a great fun, even more when you know that **a lot of data about players and teams is available on the official site of FPL**. Part of it, concerning teams' attacking and defensive strength, **can be easily used to try to predict the outcome** of an upcoming match. This is exactly what I'm doing here.

#  ## Find and load the data
# Normally, you should download it from the official site (url set below). However, there was a problem doing it via Kaggle kernel, so for the sake of this excercise I have downloaded the data manually and posted it as an input file here. Obviously, by the time you read this, it will be outdated.
# 
# I recommend running this notebook on your local machine to get results quickly and automatically.

# In[ ]:


import numpy as np
import pandas as pd
import json
import urllib.request


# In[ ]:


def getFPLData():
    url = 'https://fantasy.premierleague.com/drf/bootstrap-static'
    
    # this works locally
    try:
        file = urllib.request.urlopen(url)
        raw_text = file.read().decode('utf-8')
        
    # this is a workaround for Kaggle kernel to run
    except:
        file = open('../input/FPL_data.json','r')
        raw_text = file.read()
        
    json_data = json.loads(raw_text)
    return json_data


# In[ ]:


FPL_json = getFPLData()


# ## Extract the teams and their upcoming fixtures
# Once we have all the data, we look for team stats. We are interested in their strength as measured by these columns:

# In[ ]:


interesting_columns = ['id','name','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away']

teams_list = []

for t in FPL_json['teams']:
    columns_list = []
    for c in interesting_columns:
        columns_list.append(t[c])
    
    teams_list.append(columns_list)

teams_df = pd.DataFrame(teams_list, columns = interesting_columns)
teams_df


# These columns seem to represent the **ELO rating** of attacking and defensive strength of the teams, calculated separately for home and away matches. The higher the rating, the stronger the team. 
# 
# Now we look into upcoming fixtures.

# In[ ]:


interesting_columns = ['id','kickoff_time_formatted','team_h','team_a']

fixtures_list = []

for f in FPL_json['next_event_fixtures']:
    columns_list = []
    for c in interesting_columns:
        columns_list.append(f[c])
    
    fixtures_list.append(columns_list)

fixtures_df = pd.DataFrame(fixtures_list, columns = interesting_columns)
fixtures_df


# We will join these two tables to get the full picture.

# In[ ]:


temp_cols = ['team_h','name','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away']
teams_df.columns = temp_cols
fixtures_df_join1 = fixtures_df.merge(teams_df, on='team_h')

temp_cols = ['team_a','name','strength_attack_home','strength_attack_away','strength_defence_home','strength_defence_away']
teams_df.columns = temp_cols
fixtures_df_join2 = fixtures_df_join1.merge(teams_df, on='team_a')


# ## Calculate strength differences
# Since we would like to estimate how many goals each team will score, we will calculate **differences** between teams' offensive and defensive strength. The logic here is simple: a strong attack of the home team compared with visitor's weak defense should produce a rainfall of home goals.

# In[ ]:


fixtures_df_join2['home_goals_strength_diff'] = fixtures_df_join2['strength_attack_home_x'] - fixtures_df_join2['strength_defence_away_y']
fixtures_df_join2['away_goals_strength_diff'] = fixtures_df_join2['strength_attack_away_y'] - fixtures_df_join2['strength_defence_home_x']

# Let's clear this up
interesting_columns = ['kickoff_time_formatted','name_x','home_goals_strength_diff','away_goals_strength_diff','name_y']
fixtures_final_df = fixtures_df_join2.loc[:,interesting_columns]
final_column_names = ['kickoff','home','home_score','away_score','away']
fixtures_final_df.columns = final_column_names
fixtures_final_df


# ## Interpretation and prediction
# The way I see it, when a team has a score around 200-300, that means their attacking formation is far better than opponent's defense, hence I expect them to produce two or three goals. If, on the other hand, a team has a negative score of -200 or less, I don't expect them to find the net at all. Results around zero mean that one team's offense is comparable to the other team's defense, so it's possible they will hit one goal. 
# 
# Reasoning this way, I made the following predictions:
# * Swa 0:2 Ars
# * WHU 0:1 CP
# * Hud 0:3 Liv
# * Che 2:0 Bou
# * Eve 0:1 Lei
# * New 1:1 Bur
# * Sou 1:1 Bri
# * MCI 2:0 WBA
# * Tot 1:1 MU
# * Sto 1:1 Wat
# 
# Time will tell how wrong I was:) Obviously, you can't expect this simple method to correctly predict the future. There will be mistakes, and there will be huge mistakes, too.
