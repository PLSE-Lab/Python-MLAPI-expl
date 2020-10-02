#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
INPUT_DIR = "../input/"
# Any results you write to the current directory are saved as output.


# **groupId** - Integer ID to identify a group within a match. If the same group of players plays in different matches, they will have a different groupId each time.
# 
# **matchId** - Integer ID to identify match. There are no matches that are in both the training and testing set.
# 
# **assists** - Number of enemy players this player damaged that were killed by teammates.
# 
# **boosts** - Number of boost items used.
# 
# **damageDealt** - Total damage dealt. Note: Self inflicted damage is subtracted.
# 
# **DBNOs** - Number of enemy players knocked.
# 
# **headshotKills** - Number of enemy players killed with headshots.
# 
# **heals** - Number of healing items used.
# 
# **killPlace** - Ranking in match of number of enemy players killed.
# 
# **killPoints** - Kills-based external ranking of player. (Think of this as an Elo ranking where only kills matter.)
# 
# **kills** - Number of enemy players killed.
# 
# **killStreaks** - Max number of enemy players killed in a short amount of time.
# 
# **longestKill** - Longest distance between player and player killed at time of death. This may be misleading, as downing a - player and driving away may lead to a large longestKill stat.
# 
# **maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# 
# **numGroups** - Number of groups we have data for in the match.
# 
# **revives** - Number of times this player revived teammates.
# 
# **rideDistance** - Total distance traveled in vehicles measured in meters.
# 
# **roadKills** - Number of kills while in a vehicle.
# 
# **swimDistance** - Total distance traveled by swimming measured in meters.
# 
# **teamKills** - Number of times this player killed a teammate.
# 
# **vehicleDestroys** - Number of vehicles destroyed.
# 
# **walkDistance** - Total distance traveled on foot measured in meters.
# 
# **weaponsAcquired** - Number of weapons picked up.
# 
# **winPoints** - Win-based external ranking of player. (Think of this as an Elo ranking where only winning matters.)
# 
# **winPlacePerc** - The target of prediction. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# # Load data

# In[ ]:


data = pd.read_csv(INPUT_DIR + "train_V2.csv", nrows=10000)
data.info()


# # Visualize data w.r.t. one parameter at a time.
# Get influence of parameter on the winPlacePerc

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

# create co relation matrix using seaborn heatmap
columns_to_ignore = ["Id", "groupId", "matchId"]
columns_to_show = [column for column in data.columns if column not in columns_to_ignore]
co_relation_matrix = data[columns_to_show].corr()
plt.figure(figsize=(11,9))
sns.heatmap(co_relation_matrix,
             xticklabels=co_relation_matrix.columns.values,
             yticklabels=co_relation_matrix.columns.values,
             linecolor="white",
             linewidth=0.1,
             cmap="RdBu"
           )


# **killPoints, matchDuration, maxPlace, numGroups, rankPoints, roadKills, swimDistance, teamKills, vehicleDestroys, winPoints Don't have a lot of significance on the winPlacePerc**

# # Aggrigate data wrt group ID

# In[ ]:


agg = data.groupby("groupId").size().to_frame('players_in_team')
data = data.merge(agg, how="left", on="groupId")

data['headshotKillsOverKills'] = data['headshotKills'] / data['kills']
data['headshotKillsOverKills'].fillna(0, inplace=True)

data['killPlaceOverMaxPlace'] = data['killPlace'] / data['maxPlace']
data['killPlaceOverMaxPlace'].fillna(0, inplace=True)
data['killPlaceOverMaxPlace'].replace(np.inf, 0, inplace=True)

corr = data[['killPlace', 'walkDistance', 'headshotKillsOverKills', 'players_in_team',
             'killPlaceOverMaxPlace', 'winPlacePerc']].corr()
sns.heatmap(corr,
    xticklabels=corr.columns.values,
    yticklabels=corr.columns.values,
    annot=True,
    linecolor='white',
    linewidth=0.1,
    cmap="RdBu"
)


# In[ ]:




