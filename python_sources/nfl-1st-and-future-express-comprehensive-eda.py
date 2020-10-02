#!/usr/bin/env python
# coding: utf-8

# # NFL 1st and Future 2019
# ### Can you investigate the relationship between the playing surface and the injury and performance of NFL athletes?

# **To understand the precise goal of the competittion and the data, please have a look at https://www.kaggle.com/c/nfl-playing-surface-analytics/data, which describes each variable of each datafile**
# 
# ![](https://1ycbx02rgnsa1i87hd1i7v1r-wpengine.netdna-ssl.com/wp-content/uploads/2019/01/nfl.png)
# 

# In[ ]:


import pandas as pd
import matplotlib.pylab as plt
import seaborn as sns
import matplotlib.patches as patches
import pandas_profiling
import warnings
warnings.filterwarnings('ignore')

from time import time


# In[ ]:


# Load the data files
playlist = pd.read_csv('../input/nfl-playing-surface-analytics/PlayList.csv')
injuries = pd.read_csv('../input/nfl-playing-surface-analytics/InjuryRecord.csv')
tracking = pd.read_csv('../input/nfl-playing-surface-analytics/PlayerTrackData.csv', nrows=int(1e6)) # load only a fraction of the data


# ## Playlist Analysis
# Let's first analyse the PlayList. It contains information on the player and on the game. Nothing crazy but it's the main source of info on the player and on the play conditions (weather, stadium, etc.). Note that some features have missing values (but I think it's no big deal).

# In[ ]:


# Make report for playlist
pandas_profiling.ProfileReport(playlist)


# ## Injuries Analysis
# This dataset provides information about each injury. Most information here is categorical and/or one-hot-encoded!

# In[ ]:


# Make report for injuries
pandas_profiling.ProfileReport(injuries)


# ## Tracking Analysis
# This dataset is I think the most informative: it allows to track each trajectory of a player. This continuous information will allow you to make rich features based on trajectories, speed, acceleration, etc.

# In[ ]:


# Make report for tracking data
pandas_profiling.ProfileReport(tracking)


# In[ ]:




