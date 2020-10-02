#!/usr/bin/env python
# coding: utf-8

# Main idea to repeat the same output with new **betalytics** library unde Kaggle environemnt 
# Original idea took from - https://colab.research.google.com/drive/1_OSGR54nQB96P4cfeQu5a80mZD31BNjd
# 

# In[ ]:


# Install a new library for betting analytics
# it's so raw at that moment, but all cool stuff should be added later

get_ipython().system('pip install git+https://github.com/sashml/betalytics.git')


# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sqlite3

from betalytics.soccer.const import BOOKIE, MATCH_INFO 
from betalytics.soccer.loader.football_data_loader import load_and_normalize_data
from betalytics.soccer.strategies.all import apply_results
from betalytics.soccer.stats.team_ratings import get_standings_table

get_ipython().run_line_magic('matplotlib', 'inline')


# ## Load Data

# In[ ]:


# Total Matches Over Seasons
data = load_and_normalize_data(db_file_name='/kaggle/input/european-football/database.sqlite', bookie='BET365')
#data = load_and_normalize_data(db_file_name='database.db', bookie='BET365')


# In[ ]:


# Review how many matches do we have
season_gr = data[['LEAGUE', 'SEASON']].groupby(by=['LEAGUE', 'SEASON']).size().unstack(fill_value=0)
season_gr


# ## Apply results

# In[ ]:


match_results = apply_results(data)
match_results = match_results.sort_values(by='DATE')
match_results.loc[:, [
    'DATE','RESULT', 'RESULT_ON_HOME', 'RESULT_ON_FAVORITE', 'RESULT_ON_DOG', 
    'ODDS_ON_HOME', 'ODDS_ON_FAVORITE', 'ODDS_ON_DOG',
    'HOME_ODDS', 'DRAW_ODDS', 'AWAY_ODDS']].tail(5)


# In[ ]:


rated_team = {}
seasons = sorted(match_results['SEASON'].unique())
for league in match_results['LEAGUE'].unique():
    for idx in range(1, len(seasons)):
        prev_season_data = match_results[
            (match_results['SEASON'] == seasons[idx-1]) & 
            (match_results['LEAGUE'] == league)
        ]
        if prev_season_data.empty:
            continue
        teams = get_standings_table(prev_season_data, n_teams=5)
        rated_team.setdefault(league, {}).setdefault(seasons[idx], teams)
print('CALCULATED THE MOST RATED TEAMS!')


# ## Review results

# In[ ]:


sorted(rated_team.keys())


# In[ ]:


rated_team['Spain.La Liga Primera Division']['2018/2019']


# ## Summary
# 
# - It was quick introduction to new BETALYTICS library
# - Demonstrated simple approach to detect the best N-rated teams season over season

# In[ ]:




