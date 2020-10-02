#!/usr/bin/env python
# coding: utf-8

# # 1. Introduction
# This is a simple estimatation of the season total projected fantasy points for all players based on ESPN's standard fantasy scoring format. Season total fantasy points per player is the most important statistic to look at when drafting players in fantasy.

# # 2. Data Transformation
# First we read in the data

# In[ ]:


import pandas as pd

bball_projects = pd.read_csv("/kaggle/input/nba-stat-projections-2019/FantasyPros_Fantasy_Basketball_Overall_2019_Projections.csv", index_col =0)
df = bball_projects


# Next we are going to create a new column called FPTS that signifies the fantasy point total for the season. Again this value is calculated by using the standard scoring metrics that ESPN uses for their fantasy leagues.

# In[ ]:


df = (df.assign(FPTS = lambda x : x.PTS + x.REB * 1.1 + x.AST * 1.5 + (x.BLK + x.STL)* 2 - (((x.PTS - (3 * x['3PM']))/2)/x['FG%']) - x.TO * 2))


# Now we are going to sort the players based off of the new statistic that we created.

# In[ ]:


df = df.sort_values(by = 'FPTS', ascending = False)
print(df)


# Notes of inaccuracy: Estimated point per REB is given 1.1, becasue DREB = 1
# and OREB = 1.5 so I estimated about 1.1 based on frequencies. Also, points
# subtracted for missed field goals do not take into account FTs and only take
# into account two-point FG%. Therefore, poor FT shooters like Andre Drummond,
# and good 3FG% shooters like Stephen Curry are overrated in this estimate.
# 
# With more statistics, such as FTA/FTM, FGA/FGM, and 3FGA/3FGM a more accurate
# estimate can be made. Additionally, info about their projected draft position
# could be used to find players that are being under/over valued by others.

# 
