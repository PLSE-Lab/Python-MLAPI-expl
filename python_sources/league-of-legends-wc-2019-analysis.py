#!/usr/bin/env python
# coding: utf-8

# * Introduction
# * Preparations
#     * Matches
#     * Players
#     * Champions
# * Quick Look: Files and content
# * Visual Overview
#     * Matches
#     * Players
#     * Champions
# * Analysis
#     * Matches
#     * Players
# * Conclusion
# * Note

# ## Introduction

# The 2019 League of Legends World Championship was the ninth world championship for League of Legends, an esports tournament for the video game developed by Riot Games. It was held from October 2, 2019, to November 10, 2019, in Berlin, Madrid and Paris. Twenty four teams from 13 regions qualified for the tournament based on their placement in regional circuits such as those in China, Europe, North America, South Korea and Taiwan/Hong Kong/Macau with twelve of those teams having to reach the main event via a play-in stage.

# Let's check why **FunPlus** Phoenix won this competition and we can conclude something about selection of casters on matches.

# ## Preparations

# In[ ]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:


sns.set_style('ticks')


# In[ ]:


data_matches = pd.read_csv('../input/league-of-legends-world-championship-2019/wc_matches.csv')
data_players = pd.read_csv('../input/league-of-legends-world-championship-2019/wc_players.csv')
data_champions = pd.read_csv('../input/league-of-legends-world-championship-2019/wc_champions.csv')


# ### Matches

# In[ ]:


data_matches.info()


# In[ ]:


data_matches.head()


# Let's remove unnecessary column **Unnamed: 0**

# In[ ]:


del data_matches['Unnamed: 0']


# Let's check this table for NA.

# In[ ]:


data_matches.isna().sum()


# No NA values, great !

# Also, **date** column has *object* type, let's convert it to a *datetime*.

# In[ ]:


data_matches['date'] = pd.to_datetime(data_matches['date'])


# Now, problems with this table have been resolved.

# ### Players

# In[ ]:


data_players.info()


# In[ ]:


data_players.head()


# Another unnecessary column **Unnamed: 0**

# In[ ]:


del data_players['Unnamed: 0']


# In[ ]:


data_players.isna().sum()


# In[ ]:


data_players['heraldtime'].isna().sum()


# In[ ]:


del data_players['heraldtime']


# ### Champions

# In[ ]:


data_champions.info()


# In[ ]:


data_champions.head()


# In[ ]:


del data_champions['Unnamed: 0']


# In[ ]:


data_champions.isna().sum()


# Great, no problems with this table.

# ## Quick Look

# In[ ]:


data_matches.describe()


# 23 unique winners and 24 unique teams ? There is team without wins ? Let's find out this team.

# In[ ]:


winners = set(data_matches['winner'].explode().unique())
winners


# In[ ]:


teams = set(data_matches['team1'].explode().unique())
teams


# In[ ]:


teams.difference(winners)


# **ahq eSports Club** the only team that didn't win a single game, interesting.

# In[ ]:


data_players.describe()


# Nothing interesting at first sight. 

# In[ ]:


data_champions.describe()


# Nothing interesting too.

# ## Visual Overview

# ### Matches

# In[ ]:


data_matches.head()


# In[ ]:


mvp_data = data_matches['mvp'].value_counts().to_frame().reset_index()
mvp_data.columns = ['name', 'mvp_count']
mvp_data.head()


# In[ ]:


fig, ax = plt.subplots()
fig.set_size_inches(50.7, 25.27)
sns.barplot(data=mvp_data, x='name', y='mvp_count')


# In[ ]:


sides = ['Blue', 'Red']
def count_win_on_side(row):
    if (row['winner'] == row['blue']):
        return pd.Series([1, 0], sides)
    else:
        return pd.Series([0, 1], sides)

data_sides = data_matches.apply(lambda row: count_win_on_side(row), axis=1).mean()


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))

colors = ['#51acc6', '#ea96a3']
plt.pie(data_sides, colors=colors, labels=sides)

ax.set_title('Distribution of the winning percentage by side', pad=20)
plt.axis('equal')
plt.show()


# In[ ]:


pbp_caster_data = data_matches['pbp_caster'].value_counts().to_frame().reset_index()
pbp_caster_data.columns = ['name', 'count']
fig, ax = plt.subplots()
fig.set_size_inches(7, 5)
sns.barplot(data=pbp_caster_data, x='name', y='count')


# In[ ]:


color_caster_data = data_matches['color_caster'].value_counts().to_frame().reset_index()
color_caster_data.columns = ['name', 'count']
fig, ax = plt.subplots()
fig.set_size_inches(33, 10)
sns.barplot(data=color_caster_data, x='name', y='count')


# In[ ]:


data_winner = data_matches['winner'].value_counts().to_frame().reset_index()
data_winner.columns = ['name', 'count']
fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect='equal'))

wedges, texts = plt.pie(data_winner['count'], labels=data_winner['name'])

ax.set_title('Distribution of the winning percentage by team', pad=20)
ax.legend(wedges, data_winner['name'],
          title='Teams',
          loc='center left',
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.axis('equal')
plt.show()


# ### Players

# In[ ]:


data_players.head()


# In[ ]:


columns_ban = ['ban1', 'ban2', 'ban3', 'ban4', 'ban5']
data_bans = None
for column_name in columns_ban:
    # This is player's data, so we need to divide ban count by members count in a team
    bans = data_players[column_name].value_counts()/5
    bans = bans.astype('int').to_frame().reset_index()
    if (data_bans is None):
        data_bans = bans
    else:
        data_bans = data_bans.merge(bans, on='index')
data_bans['ban_count'] = data_bans['ban1'] + data_bans['ban2'] + data_bans['ban3'] + data_bans['ban4'] + data_bans['ban5']
del data_bans['ban1'], data_bans['ban2'], data_bans['ban3'], data_bans['ban4'], data_bans['ban5']
data_bans.columns = ['champion', 'ban_count']
print(data_bans)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect='equal'))

wedges, texts = plt.pie(data_bans['ban_count'], labels=data_bans['champion'])

ax.set_title('Distribution of the ban percentage by champion', pad=20)
ax.legend(wedges, data_bans['champion'],
          title='Champions',
          loc='center left',
          bbox_to_anchor=(1, 0, 0.5, 1))
plt.axis('equal')
plt.show()


# In[ ]:


data_players.groupby('player')['csat15'].mean().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['xpat10'].mean().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['teamkills'].sum().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['ft'].sum().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['totalgold'].mean().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['pentas'].sum().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['fb'].sum().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['minionkills'].mean().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['wards'].mean().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['visionwards'].mean().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['dmgtochamps'].mean().sort_values(ascending=False)


# In[ ]:


data_players.groupby('player')['fbaron'].sum().sort_values(ascending=False)


# ### Champions

# In[ ]:


data_champions.head()


# In[ ]:


data_champions.sort_values(by='win_total', ascending=False)


# In[ ]:


data_champions.sort_values(by='winrate_total', ascending=False)


# ## Analysis

# ### Matches

# In[ ]:


data_matches.head()


# In[ ]:


data_phoenix_win_matches = data_matches.loc[((data_matches['team1'] == 'FunPlus Phoenix') | (data_matches['team2'] == 'FunPlus Phoenix') & (data_matches['winner'] == 'FunPlus Phoenix'))]


# Let's check sides first.

# In[ ]:


data_phoenix_win_matches


# In[ ]:


data_phoenix_win_sides = data_phoenix_win_matches.apply(lambda row: count_win_on_side(row), axis=1).mean()


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))

colors = ['#51acc6', '#ea96a3']
plt.pie(data_phoenix_win_sides, colors=colors, labels=sides)

ax.set_title('Distribution of the FunPlus Phoenix winning percentage by side', pad=20)
plt.axis('equal')
plt.show()


# On blue side you can more easily contest dragons, so maybe FunPlux Phoenix team played around botside and contest dragons ?

# Let's check loses of this team.

# In[ ]:


data_phoenix_lose_matches = data_matches.loc[(((data_matches['team1'] == 'FunPlus Phoenix') | (data_matches['team2'] == 'FunPlus Phoenix')) & (data_matches['winner'] != 'FunPlus Phoenix'))]


# In[ ]:


data_phoenix_lose_sides = data_phoenix_lose_matches.apply(lambda row: count_win_on_side(row), axis=1).mean()


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 7), subplot_kw=dict(aspect="equal"))

colors = ['#51acc6', '#ea96a3']
plt.pie(data_phoenix_lose_sides, colors=colors, labels=sides)

ax.set_title('Distribution of the FunPlus Phoenix lose percentage by side', pad=20)
plt.axis('equal')
plt.show()


# In[ ]:


data_phoenix_lose_matches


# FunPlus Phoenix lost 2 matches on a red side !

# ### Players

# In[ ]:


phoenix_players = ['GimGoon', 'Lwx', 'Crisp', 'Doinb', 'Tian']


# In[ ]:


data_players.head()


# In[ ]:


phoenix_player_data = data_players.loc[data_players['player'].isin(phoenix_players)]


# In[ ]:


phoenix_player_data


# As for previous player visual analysis, we can say that:

# In[ ]:


data_players.groupby('player')['teamkills'].sum().sort_values(ascending=False)


# FunPlus Phoenix team are top 5 in teamkills, also

# In[ ]:


data_players.groupby('player')['fbaron'].sum().sort_values(ascending=False)


# Top 5 in a first baron time.

# In[ ]:


data_players.groupby('player')['dmgtochamps'].mean().sort_values(ascending=False).head(15)


# As we can see, FPX are not in top15 in damage to champions stat and also,

# In[ ]:


data_players.groupby('player')['visionwards'].mean().sort_values(ascending=False).head(15)


# And only support in top 15 of vision wards.

# In[ ]:


data_players.groupby('player')['wards'].mean().sort_values(ascending=False).head(15)


# Same situation with simple wards.

# In[ ]:


data_players.groupby('player')['minionkills'].mean().sort_values(ascending=False).head(15)


# In[ ]:


data_players.groupby('player')['fb'].sum().sort_values(ascending=False).head(15)


# As we can see in other stat, we can't find players from FPX except GimGoon.

# I can conclude that FPX won early game and stomp their opponents with gold diff. We can't see them in top cs or damage stat, only in baron and teamkills.

# ## Conclusion

# As we see, we can identify several reasons for FPX victory in WC 2019:
# 1. Stomp Early
# 2. Early First Baron 
# 3. A lot of fights, which leads to a large number of teamkills and gold.
# 
# We can only congratulate this team on the victory with such statistics, bypassing all the "rules" of the professional League of legends, they were able to win with their ability to play as a team.
# 
# As for casters Riot Games prefers **Jatt**, **Medic**, **Drakos** as for their main casters for WC 2019.

# ## Note

# If you like this notebook, please upvote it.  
# You can also view my *League of Legends* datasets.
# * [WC 2019](https://www.kaggle.com/ilyadziamidovich/league-of-legends-world-championship-2019)
# * [LCL 2019](https://www.kaggle.com/ilyadziamidovich/league-of-legends-lcl-2019)
