#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
plt.style.use('ggplot')
print(os.listdir("../input"))


# In[ ]:


team_df = pd.read_csv('../input/team_info.csv')


# In[ ]:


team_df.head()


# In[ ]:


game_df.head()


# In[ ]:


"""
Add home and away team names.
"""
game_df = pd.read_csv('../input/game.csv')
game_df = game_df.merge(team_df[['team_id', 'teamName']],
              left_on='home_team_id', right_on='team_id') \
    .merge(team_df[['team_id', 'teamName']], left_on='away_team_id',
           right_on='team_id', suffixes=('home','away'))


# In[ ]:


game_df.head()


# # Distribution of Goals
# The distribution of home goals had a higher mean than away goals. "Home ice advantage?"

# In[ ]:


game_df[['away_goals','home_goals']].plot(kind='hist', figsize=(15,5), bins=10, alpha=0.5, title='Distribution of Home vs. Away Goals')


# # Average Goals per Team
# 

# In[ ]:


game_df.groupby('teamNamehome').mean()['home_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Scored in Home Games')
plt.show()
game_df.groupby('teamNameaway').mean()['away_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Scored in Away Games')
plt.show()


# # Average Goals Allowed per Team

# In[ ]:


game_df.groupby('teamNamehome').mean()['away_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Allowed in Home Games')
plt.show()
game_df.groupby('teamNameaway').mean()['home_goals']     .sort_values()     .plot(kind='barh', figsize=(15, 8), title='Average Goals Allowed in Away Games')
plt.show()


# # Distribution of Point Differential
# 
# How much do teams usually win/lose by? Point differential is computed as:
# `point_diff` = `home team goals` - `away team goals`

# In[ ]:


game_df['point_diff'] = game_df['home_goals'] - game_df['away_goals']


# In[ ]:


game_df['point_diff'].plot(kind='hist',
                           bins=18,
                           title='NHL Point Differential (Negative Home team Loses, Positive Home team Wins)',
                           xlim=(-10,10))


# # Biggest Blowout

# In[ ]:


#Biggest Blowout was by 10 points
game_df['point_diff'].abs().max()


# In[ ]:


# Blowout game:
game_df.loc[game_df['point_diff'] == 10]


# Ouch that was a blowout. 
# Here are the video highlights in case you were wondering how it happened: https://www.nhl.com/bluejackets/video/recap-mtl-0-cbj-10/t-283041746/c-46026003

# Lets define game types as:
# - Blowout (abs point diff >= 3 points)
# - Normal Game (win > 2 points)
# - Tight Game (win by 1 point)

# In[ ]:


game_df['point_diff_type'] = game_df['point_diff'].abs().apply(lambda x: 'Blowout' if x>=3 else ('Normal' if x>=2 else 'Tight'))


# In[ ]:


# Create one dataframe with the point 
point_diff_team = pd.concat([game_df[['teamNamehome','point_diff_type','point_diff','date_time']].rename(columns={'teamNamehome':'team'}),
    game_df[['teamNameaway','point_diff_type','point_diff','date_time']].rename(columns={'teamNameaway':'team'})])


# In[ ]:


point_diff_team['date_time'] = pd.to_datetime(point_diff_team['date_time'])


# In[ ]:


for team, data in point_diff_team.groupby('team'):
    data.groupby(data['date_time'].dt.year).mean()['point_diff'].plot(kind='line', title='{} Average Point Diff By Year'.format(team), figsize=(15,2))
    plt.show()


# # TODO
# - Better metrics
# - Figure out how to make plot colors = team colors
# - More fun stuff
# 
# GO CAPS!

# In[ ]:




