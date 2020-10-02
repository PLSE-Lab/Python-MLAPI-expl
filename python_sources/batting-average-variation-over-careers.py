#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pylab
import seaborn as sns
import pandas as pd
import numpy as np


# In[ ]:


df = pd.read_csv("../input/batting.csv")


# To look at the career variation of batting average, we'll look at players who are finished with their careers who have some number of seasons of complete stats.
# 
# For convenience, we'll look at players who played from 1970 on.
# 
# We'll also not compare careers of different lengths directly to avoid trying to normalize career lengths and have the plots still be meaningful.

# In[ ]:


players = df[(df.year >= 1970)].player_id.unique()
season_counts = df[df.player_id.isin(players)].player_id.value_counts()
ax = sns.distplot(season_counts, bins=range(0,35), );
ax.set_xlabel("Number of Career Seasons");
ax.set_title("Distribution of MLB Player Career Lengths\nfor Players in 1970 and on");


# This function gets all players with `n_seasons` of play and gets their batting average for each season. It then normalizes the season averages to the total average so we can see how players change over their careers.

# In[ ]:


def stat_variation(n_seasons, season_counts, df):
    player_list = season_counts[season_counts == n_seasons].index.tolist()
    player_groups = df.groupby("player_id")
    player_list = [x for x in player_list if player_groups.get_group(x).year.max() < 2016]
    P = df[df.player_id.isin(player_list)][["player_id", "g", "ab", "h", "year"]].copy()
    P['avg'] = P.h/P.ab
    player_data = []
    player_data2 = []
    for name, group in P.groupby("player_id"):
        g = group.dropna()
        mn = g.avg.mean()
        vals = g.avg.as_matrix().ravel() - mn
        # ignore players with 0 or NaN (from the dropna) seasons
        if len(vals) != n_seasons or all(vals == 0):
            continue
        player_data.append(vals)
        mn = g.g.mean()
        player_data2.append(g.g.as_matrix().ravel() - mn)
    D = pd.DataFrame(np.array(player_data).ravel(), columns=["Avg"])
    E = pd.DataFrame(np.array(player_data2).ravel(), columns=["Avg"])
    D["Season"] = np.tile(np.arange(1, n_seasons+1), len(player_data))
    E["Season"] = np.tile(np.arange(1, n_seasons+1), len(player_data2))
    return D, E


# In[ ]:


# Let's look at 8 seasons of play
n_seasons = 8
D, E = stat_variation(n_seasons, season_counts, df)
fig, ax = pylab.subplots(figsize=(10, 5))
sns.boxplot("Season", "Avg", data=D, ax=ax, showfliers=False);
ax.plot([-1, n_seasons+1], [0,0], '-k'); ax.set_title("Batting Average Variation from Career Mean");


# In[ ]:


fig, ax = pylab.subplots(figsize=(10, 5))
sns.boxplot("Season", "Avg", data=E, ax=ax, showfliers=False);
ax.plot([-1, n_seasons+1], [0,0], '-k'); ax.set_title("Games Per Season Variation from Career Mean");


# In[ ]:


# Let's look at 10 seasons of play
n_seasons = 10
D, E = stat_variation(n_seasons, season_counts, df)
fig, ax = pylab.subplots(figsize=(10, 5))
sns.boxplot("Season", "Avg", data=D, ax=ax, showfliers=False);
ax.plot([-1, n_seasons+1], [0,0], '-k'); ax.set_title("Batting Average Variation from Career Mean");


# In[ ]:


fig, ax = pylab.subplots(figsize=(10, 5))
sns.boxplot("Season", "Avg", data=E, ax=ax, showfliers=False);
ax.plot([-1, n_seasons+1], [0,0], '-k'); ax.set_title("Games Per Season Variation from Career Mean");


# In[ ]:


# Let's look at 5 seasons of play
n_seasons = 5
D, E = stat_variation(n_seasons, season_counts, df)
fig, ax = pylab.subplots(figsize=(10, 5))
sns.boxplot("Season", "Avg", data=D, ax=ax, showfliers=False);
ax.plot([-1, n_seasons+1], [0,0], '-k'); ax.set_title("Batting Average Variation from Career Mean");


# In[ ]:


fig, ax = pylab.subplots(figsize=(10, 5))
sns.boxplot("Season", "Avg", data=E, ax=ax, showfliers=False);
ax.plot([-1, n_seasons+1], [0,0], '-k'); ax.set_title("Games Per Season Variation from Career Mean");


# Further improvements to make are:
# 
# 1. Filter out pitchers
# 2. Define a function for any kind of stat
# 3. Normalize to different lengths of careers to compare stat motion between career lengths
# 
# For (3), perhaps a low-parameter regression with meaningful coefficients could be built and the coefficients compared. 

# In[ ]:





# In[ ]:




