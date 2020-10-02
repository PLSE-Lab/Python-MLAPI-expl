#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib as mp

from subprocess import check_output
# print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


parks = pd.read_csv('../input/park.csv')
players = pd.read_csv('../input/player.csv')
appearances = pd.read_csv('../input/appearances.csv')
teams = pd.read_csv('../input/team.csv')


# In[ ]:


ps = pd.read_csv('../input/postseason.csv')
ws = ps[(ps['round'] == 'WS') & (ps['year'] > 1899)]


# In[ ]:


teams = teams[teams['year'] > 1899]
teams.rename(columns = {'team_id':'team_id_winner'}, inplace = True)


# In[ ]:


print(ws.columns)
print(teams.columns)


# In[ ]:


team_ids = set(teams['team_id_winner'])
lahman_ids = set(teams['team_id_lahman45'])
retro_ids = set(teams['team_id_retro'])
ws_winner_ids = set(ws['team_id_winner'])


# In[ ]:


for item in [team_ids, lahman_ids, retro_ids]:
    print(len(ws_winner_ids -item))
    print(ws_winner_ids - item)
# use team_id_winner column for merge


# In[ ]:


season_info = teams[['year', 'team_id_winner', 'w', 'l', 'name', 'attendance']]


# In[ ]:


type(ws)


# In[ ]:


df = ws.merge(right=season_info, how='left', on=['year', 'team_id_winner'])


# In[ ]:


df.head()


# In[ ]:


def win_pct(row):
    return row['w']/(row['w'] + row['l'])

df['win_pct'] = df.apply(lambda row : win_pct(row), axis=1)


# In[ ]:


worst_5 = df.sort_values(by='win_pct', ascending=True)


# In[ ]:


attendance = df.sort_values(by='attendance', ascending=True)


# In[ ]:


attendance[attendance['year'] > 1950].head(10)


# In[ ]:


attend = season_info.sort_values(by='attendance', ascending=False)


# In[ ]:


attend[attend['team_id_winner'] == 'BOS']


# In[ ]:




