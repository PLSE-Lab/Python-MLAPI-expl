#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


fixture_df = pd.read_csv("../input/fantasy-premier-league-2019-20/fixture_details.csv")
gw_df = pd.read_csv("../input/fantasy-premier-league-2019-20/gw_history.csv")
players_df = pd.read_csv("../input/fantasy-premier-league-2019-20/player_details.csv").set_index('id')
teams_df = pd.read_csv("../input/fantasy-premier-league-2019-20/team_details.csv")


# In[ ]:


import urllib.request, json
import matplotlib.pyplot as plt


# In[ ]:


players_df.head()


# In[ ]:


element_types = {'GKP':1, 'DEF': 2, 'MID':3, 'FWD': 4}


# In[ ]:


players_df = players_df[players_df['minutes']>60]
players_df['points_per_game'] = players_df['points_per_game'].astype(float)
players_df['games_played'] = round(players_df['total_points']/players_df['points_per_game'])


# In[ ]:


gkps = players_df[players_df['element_type']==1]
defs = players_df[players_df['element_type']==2]
mids = players_df[players_df['element_type']==3]
fwds = players_df[players_df['element_type']==4]


# In[ ]:


gkps['saves_per_game'] = gkps['saves']/gkps['games_played']
mean_saves_per_game = gkps['saves_per_game'].mean()
mean_saves_per_game


# In[ ]:


s = [300*n for n in gkps['saves_per_game']]
colors = ['r','g']
c = [colors[b] for b in (gkps['saves_per_game']>mean_saves_per_game).to_list()]
plt.figure(figsize=[20,10])
plt.scatter(gkps['now_cost'], gkps['points_per_game'],s=s, color=c)
for i in gkps.index:
    plt.annotate(gkps['web_name'][i], (gkps['now_cost'][i], gkps['points_per_game'][i]))
plt.show()


# * The above graph shows how goal keepers of different value have performed according to points per game.
# * Size of the bubble is the amount of saves per game by the goalkeeper.
# * Green color means the GK has more saves per game than the average, Red otherwise
# * Insight - Lower ranked team GKs are in green color since they face more shots and hence more chances to save

# **Real BPS**
# * Subtract bps points for actions that are already accounted in total points.
# * This will give an idea about what players are true 'bps magnets' and don't have just the inflated bps due to scoring goals or saving penalties.

# In[ ]:


bps = {}
#common to all
bps['min_60'] = 3
bps['assists'] = 9
bps['yellow_card'] = -3
bps['red_card'] = -9
bps['own_goal'] = -6

#gkps, defs
bps['gkps_def_score'] = 12
bps['gkps_def_cs'] = 12
bps['pen_save'] = 15

#mids
bps['mids_score'] = 18

#fwds
bps['fwds_score'] = 24


# In[ ]:


gkps['real_bps'] = gkps['bps'] - gkps['goals_scored']*bps['gkps_def_score'] - gkps['penalties_saved']*bps['pen_save'] - gkps['assists']*bps['assists'] - gkps['clean_sheets']*bps['gkps_def_cs'] 


# In[ ]:


gkps['real_bps_per_game'] = gkps['real_bps']/gkps['games_played']
gkps[['web_name','real_bps_per_game', 'bonus']].sort_values(by ='real_bps_per_game', ascending = False).head(15)


# In[ ]:


defs['real_bps'] = defs['bps'] - defs['goals_scored']*bps['gkps_def_score'] - defs['penalties_saved']*bps['pen_save'] - defs['assists']*bps['assists'] - defs['clean_sheets']*bps['gkps_def_cs']  


# In[ ]:


defs['real_bps_per_game'] = defs['real_bps']/defs['games_played']
defs[['web_name','real_bps_per_game','bonus']].sort_values(by ='real_bps_per_game', ascending = False).head(n=15)


# In[ ]:


mids['real_bps'] = mids['bps'] - mids['goals_scored']*bps['mids_score'] - mids['assists']*bps['assists']
mids['real_bps_per_game'] = mids['real_bps']/mids['games_played']
mids[['web_name','real_bps_per_game', 'bonus']].sort_values(by ='real_bps_per_game', ascending = False).head(n=15)


# In[ ]:


fwds['real_bps'] = fwds['bps'] - fwds['goals_scored']*bps['fwds_score'] - fwds['assists']*fwds['assists']
fwds['real_bps_per_game'] = fwds['real_bps']/fwds['games_played']
fwds[['web_name','real_bps_per_game', 'bonus']].sort_values(by ='real_bps_per_game', ascending = False).head(n=15)

