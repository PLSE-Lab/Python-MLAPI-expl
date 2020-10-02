#!/usr/bin/env python
# coding: utf-8

# # The issue
# The purpose of this notebook is to adjust player orientation between the two seasons for which we have data. The orientation issue is described in [this post](https://www.kaggle.com/c/nfl-big-data-bowl-2020/discussion/113384) from Peter Hurford as part of the NFL DataBowl challenge.
# 
# ASSUMPTION: We are given data for 2017 and 2018, or at least for two seasons with the same apparent difference in measuring orientation.
# 
# CAUTION: Use this data adjustment at your own risk. It may or may not be valid. But it probably is directionally correct.

# # Evidence
# 
# Here you see the distribution of quarterback directions at the time of ball snap. It makes sense that the peaks are at 90 and 270 depending on whether the offense is moving left or right. 

# In[ ]:


import numpy as np
import pandas as pd
import dask.dataframe as dd
import matplotlib.pyplot as plt

playlist = dd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')
tracks = dd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv', 
                     usecols=['PlayKey', 'event', 'dir', 'o'])

plays = playlist.loc[playlist.Position == 'QB', 'PlayKey'].unique().compute().tolist()
snaps = tracks[(tracks.PlayKey.isin(plays)) & (tracks.event == 'ball_snap')]
snaps_df = snaps.compute()
snaps_df.dir.plot.hist(bins=90, xticks = np.arange(0, 361, 45))


# Compare that plot with the distribution of orientations. Peaks are at 0, 90, 180, and 270. The difference indicates that we may have a 90 degree difference between the two seasons.

# In[ ]:


snaps_df.o.plot.hist(bins=90, xticks = np.arange(0, 361, 45))


# # Insight
# 
# To make the adjustment, we need to separate the data by seasons. We can use PlayerDays as a clue along with the knowledge that the regular season runs Sep-Dec each year. Here are the top values of the last player day for each player.

# In[ ]:


playlist_df = playlist.compute()
print(playlist_df.groupby('PlayerKey')['PlayerDay'].max().value_counts().head())


# The top two values are 113 and 477. 113 is the approximate length of one season. 477 is the approximate length of two seasons spearated by a 8-month gap between seasons. This chart of plays shows the timing for players making it through 350 days or more. Notice the dead space between seasons.

# In[ ]:


tough_guys = playlist_df.loc[playlist_df.PlayerDay >= 350, 'PlayerKey'].unique()
playlist_tough = playlist_df[playlist_df.PlayerKey.isin(tough_guys)].copy()
days = playlist_tough.groupby('PlayerDay')['PlayerGamePlay'].mean()
days.reset_index().plot.scatter('PlayerDay', 'PlayerGamePlay')


# We can separate plays by season based on this knowledge!

# # The fix
# The first thing to do is separate games by season. Here I take an oversimplified approach to show proof of concept. Only players with over 350 days are included, and any player day over 350 is considered as part of the 2nd season. Next we rotate all orientations from the 1st season by 90 degrees. The NFL DataBowl competition has more examples of how to adjust orientations. Here we see the adjusted distribution overlayed onto the directions.

# In[ ]:


playlist_tough['Season'] = np.where(playlist_tough.PlayerDay<350, 1, 2)
games = playlist_tough.drop_duplicates('GameID')[['GameID', 'Season']]

snaps_tough = snaps_df.merge(playlist_tough[['GameID', 'PlayKey']], on='PlayKey')
snaps_tough = snaps_tough.merge(games, on='GameID')

snaps_tough['o'] = np.where(snaps_tough.Season == 1,
                            np.mod(snaps_tough.o+90, 360),
                            snaps_tough.o
                            )

display(snaps_tough.dir.plot.hist(bins=90, alpha=0.8),
        snaps_tough.o.plot.hist(bins=90, alpha=0.8, xticks = np.arange(0, 361, 45))
       )


# This output looks more like what we expect to see.

# # Edit: Alternate Method
# Not all players played for two full seasons. Another method you can try is to compare the difference between direction traveled and orientation. They should be generally the same during the play. In the code below I change the angles so that 0 degrees is along the x axis toward the goal line. You can see below that it's consistent with angles based on x-y coordinates.

# In[ ]:


tracks = dd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv')
one_play = tracks[tracks.PlayKey == '47820-1-2'].compute()


# In[ ]:


start_time = one_play.loc[one_play.event == 'ball_snap', 'time'].to_numpy()[0]
end_time = one_play.loc[one_play.event == 'out_of_bounds', 'time'].to_numpy()[0]
play = one_play[one_play.time.between(start_time, end_time, inclusive=True)].copy()

play['dir_calcd'] =  360 - np.arctan2((play.y-play.y.shift()), 
                                      (play.x-play.x.shift())) *180/np.pi
play['dir_fixd'] = np.mod(play.dir+270, 360)
play['o_fixd'] = np.mod(play.o+270, 360)
display(play.head(10))


# In[ ]:


(play.dir_fixd - play.o_fixd).hist()


# My calcs may not be quite right, but it appears that orientation should be shifted for this play.

# # A lingering question
# 
# One might ask why the seasons aren't explicitly given in the data. More generally, it's not clear why games, players, locations, etc. aren't clearly identified. I'm sure there's a good reason why our NFL hosts have masked the data. Please share any thoughts or opinions. And, good luck with your analysis!
