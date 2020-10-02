#!/usr/bin/env python
# coding: utf-8

# # Watch the Videos Showing Injury
# 
# For manual viewing (with eyes) it's easiest to see collisions causing injury by looking for the jersey numbers in the videos.
# 
# This notebook's output generates a table catered to simple display with minimal distraction data for manual viewing.

# In[ ]:


import pandas as pd
import numpy as np

player_info = pd.read_csv('../input/player_punt_data.csv')
player_info.head()


# Unfortunately, this data set has 637 players that have multiple numbers or positions.

# In[ ]:


grouped = player_info.groupby('GSISID').agg('count')
grouped[(grouped.Number > 1) | (grouped.Position > 1)].count()


# The data provided doesn't specify the play ID or game ID for each GSISID, so group the numbers and positions by GSISID (player ID) so we can look for all the jersey numbers each player might be wearing in each video.  The letters 'o' and 'd' are included in jersey numbers as well, so remove letters since a real jersey number cannot actually include a letter.

# In[ ]:


grouped_player_info = player_info.groupby('GSISID').agg({
    'Number': lambda x: ','.join(x.replace(to_replace='[^0-9]', value='', regex=True).unique()), 
    'Position': lambda x: ','.join(x.unique())})

grouped_player_info.head(50)


# Now, produce a combined table with the jersey number and position information that might be useful later with the play and NGS information.

# In[ ]:


video_review = pd.read_csv('../input/video_review.csv', na_values='Unclear', dtype={'Primary_Partner_GSISID': np.float64}) 
video_footage_injury = pd.read_csv('../input/video_footage-injury.csv')

videos = pd.merge(video_review, video_footage_injury, left_on=['PlayID','GameKey'], right_on=['playid','gamekey'])
videos = pd.merge(videos, grouped_player_info, how='left', left_on='GSISID', right_on='GSISID')
videos.rename({'Number': 'Injured Player Number(s)', 'Position': 'Injured Player Position'}, axis=1, inplace=True)
videos = pd.merge(videos, grouped_player_info, how='left', left_on='Primary_Partner_GSISID', right_on='GSISID')
videos.rename({'Number': 'Other Player Number(s)', 'Position': 'Other Player Position'}, axis=1, inplace=True)
# Remove duplicate columns
videos.drop(['gamekey', 'playid', 'season'], axis=1, inplace=True)
pd.set_option("display.max_columns", 50)
pd.set_option('display.max_colwidth', 30)
videos.head()


# # Watch the video

# In[ ]:


from IPython.display import HTML
# Remove some columns that distract in manual viewing
watch = videos.drop(['GameKey', 'PlayID', 'GSISID', 'Season_Year', 'Primary_Partner_GSISID', 'Type', 'Week', 'Qtr', 'PlayDescription', 'Home_team', 'Visit_Team'], axis=1)
watch.rename({'Player_Activity_Derived': 'Injured Player Action', 'Primary_Partner_Activity_Derived': 'Other Player Action', 'Turnover_Related': 'From Turnover', 'Friendly_Fire': 'Friendly Fire'}, axis=1, inplace=True)
watch['PREVIEW LINK (5000K)'] = watch['PREVIEW LINK (5000K)'].apply(lambda x: '<a href="{0}" target="__blank">video link</a>'.format(x))
pd.set_option('display.max_colwidth', -1)
HTML(watch.to_html(escape=False))


# In[ ]:




