#!/usr/bin/env python
# coding: utf-8

# It is possible to almost perfectly deduce Yards of the previous play by looking at the change in YardLine, for over a quarter of the plays in the train set. I have not applied this to the test set (yet), but I would like to know:
# 
# 1. is it allowed to do this?
# 2. is the playid logic guaranteed to be the same in the private test set?
# 
# In my opinion, because we were not given Yards for the test set, the organizers do not want us to train a model using the test set. So, to make sure that it is impossibe:
# 
# 3. will the playid logic be changed in the private test set, to create a level playing field that is not influenced by online training within the private set?
# 
# I would prefer the latter.

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)
df_play = df[df.NflId==df.NflIdRusher].copy()

df_play['YardsFromOwnGoal'] = np.where(df_play.FieldPosition == df_play.PossessionTeam,
                                       df_play.YardLine, 50 + (50-df_play.YardLine))
df_play[['prev_game', 'prev_play', 'prev_team', 'prev_yfog', 'prev_yards']] = df_play[
        ['GameId', 'PlayId', 'Team', 'YardsFromOwnGoal', 'Yards']].shift(1)

filt = (df_play.GameId==df_play.prev_game) & (df_play.Team==df_play.prev_team) & (df_play.PlayId-df_play.prev_play<30)
df_play.loc[filt,'est_prev_yards'] = df_play[filt]['YardsFromOwnGoal'] - df_play[filt]['prev_yfog']

plt.figure(figsize=(8,8))
plt.title('deduced yards for %d of %d plays' % (sum(filt), len(filt)))
plt.scatter(*zip(*df_play[['est_prev_yards', 'prev_yards']].dropna().values), alpha=0.1)
plt.xlabel('deduced yards')
plt.ylabel('actual yards')
plt.show()

