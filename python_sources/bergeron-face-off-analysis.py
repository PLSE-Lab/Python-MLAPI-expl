#!/usr/bin/env python
# coding: utf-8

# # Bruins/Bergeron Faceoff Analysis
# Taking a look at different faceoff stats when sliced by a variety of dimensions.
# 
# Bergeron has a reputation for being a face-off powerhouse, and while the stats lean in that direction, it's worth noting that when lined up against top talent, that reputation doesn't hold up as much.
# 
# Of note, when looking at Jack Eichel, Sidney Crosby, Leon Draisaitl, Bergeron's production drops below 50%.
# 
# This demostrates that opposing teams have to get good matchups against Bergeron to blunt his wins over other lines. It also hints that Bergeron's reputation is built on matchups against lesser opponents.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import pandas as pd
faceoff_data = pd.read_csv("../input/faceoff_data.csv")

player = 'Patrice Bergeron'


# ## Bruins Games Faceoff Overall

# In[ ]:


temp_df = faceoff_data.groupby('player', sort=False, as_index=False)     .agg({ 'game_id': 'count', 'win': 'sum' })     .rename(columns={'game_id': 'count'})     .set_index('player')

temp_df['percent'] = ( temp_df['win'] / temp_df['count']) * 100

temp_df     .round({'percent': 2})     .sort_values('count', ascending=False)     .head(30)


# ## Bergeron Faceoff Stats by Opponent

# In[ ]:


pb_df = faceoff_data.loc[faceoff_data['player'] == player]

temp_df = pb_df     .groupby('opponent', sort=False, as_index=False)     .agg({ 'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('opponent')

temp_df['percent'] = (temp_df['win'] / temp_df['count']) * 100

temp_df.round({'percent': 2}).sort_values('count', ascending=False).head(20)


# In[ ]:


temp_df = faceoff_data.loc[faceoff_data['player'] == player]     .groupby('opponent', sort=False, as_index=False)     .agg({ 'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('opponent')

temp_df['percent'] = (temp_df['win'] / temp_df['count']) * 100

#temp_df.loc[temp_df['count'] > 10].round({'percent': 2}).sort_values('count', ascending=False).head(30)

fig, ax = plt.subplots()

temp_df.loc[temp_df['count'] > 10].round({'percent': 2}).head(30).plot(     'count',     'percent',     kind='scatter',     ax=ax,     linewidth=0,     figsize=(8,6)
)

plt.title('Bergeron Faceoff Opponents')

for k, v in temp_df.loc[temp_df['count'] > 10].round({'percent': 2}).head(30).iterrows():
    xy = [ v['count'], v['percent']]
    ax.annotate(k, xy, xytext=(10, -5), textcoords='offset points')

ax.axhline(y=50)
ax.axvline(x=25)
ax.text(34, 90, 'Dangerous', bbox=dict(facecolor='red', alpha=0.5))
ax.text(12, 85, 'Annoying', bbox=dict(facecolor='blue', alpha=0.5))
ax.text(12, 40, 'Ignorable', bbox=dict(facecolor='green', alpha=0.5))


# ## Bergeron Faceoff Stats by Zone

# In[ ]:


temp_df = faceoff_data.loc[faceoff_data['player'] == player]     .groupby('zone', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('zone')

temp_df['percent'] = ( temp_df['win'] / temp_df['count']) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron Faceoff Stats by Period

# In[ ]:


temp_df = faceoff_data.loc[faceoff_data['player'] == player]     .groupby('period', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('period')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron Faceoff Stats by Timezone

# In[ ]:


temp_df = faceoff_data.loc[faceoff_data['player'] == player]     .groupby('game_tz', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('game_tz')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron Faceoff Stats by Opposing Team

# In[ ]:


temp_df = faceoff_data.loc[faceoff_data['player'] == player]     .groupby('opposing_team', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('opposing_team')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron Powerplay Faceoff Stats

# In[ ]:


temp_df = faceoff_data.loc[(faceoff_data['player'] == player) & ( faceoff_data['power_play'] == True ) ]     .groupby('power_play', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('power_play')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron Penalty Kill Faceoff Stats

# In[ ]:


temp_df = faceoff_data.loc[(faceoff_data['player'] == player) & ( faceoff_data['penelty_kill'] == True ) ]     .groupby('penelty_kill', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('penelty_kill')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron 5-on-5 Faceoff Stats

# In[ ]:


temp_df = faceoff_data.loc[(faceoff_data['player'] == player) & ( faceoff_data['penelty_kill'] == False ) & ( faceoff_data['power_play'] == False ) ]     .groupby('season', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('season')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron Home Ice Faceoff Stats

# In[ ]:


temp_df = faceoff_data.loc[(faceoff_data['player'] == player) & ( faceoff_data['home_ice'] == True ) ]     .groupby('home_ice', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('home_ice')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('percent', ascending=False)


# ## Bergeron Score Differential Stats

# In[ ]:


temp_df = faceoff_data.loc[faceoff_data['player'] == player]     .groupby('score_diff', sort=False, as_index=False)     .agg({'game_id': 'count', 'win': 'sum'})     .rename(columns={'game_id': 'count'})     .set_index('score_diff')

temp_df['percent'] = ( temp_df['win'] / temp_df['count'] ) * 100

temp_df.round({'percent': 2}).sort_values('score_diff', ascending=False)

