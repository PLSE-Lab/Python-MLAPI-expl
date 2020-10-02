#!/usr/bin/env python
# coding: utf-8

# <h1> Minutes played per player </h1>
# In this kernel, I am aiming to get the number of minutes played per player per match. This we can then use in our later analysis on player level statistics.
# For me, basketball is completely new, so I am going to need some help with the specific knowledge.

# In[ ]:


import pandas as pd
import os
import numpy as np
from IPython.display import display, HTML
import multiprocessing as mp
import gc


# <h2> Load the data </h2>
# First lets load and examine the data provided.
# For now, we will focus on the datasets with the lowest level of granularity i.e. events and players.

# In[ ]:


def load_dataframe(year):
    dir = f'../input/playbyplay_{year}'
    players_df = pd.read_csv(f'{dir}/Players_{year}.csv', encoding = "ISO-8859-1")
    events_df = pd.read_csv(f'{dir}/Events_{year}.csv', encoding = "ISO-8859-1")
    return players_df, events_df


# In[ ]:


years = np.arange(2010, 2019)

with mp.Pool(4) as pool: 
    dfs = pool.map(load_dataframe, years)

dfs = list(zip(*dfs))
players = pd.concat(dfs[0])
events = pd.concat(dfs[1])

del dfs
gc.collect()

display(HTML(f'<h3>Players</h3>'))
display(players.sample(5))
display(players.describe(include="all").T)
display(HTML(f'<h3>Events</h3>'))
display(events.sample(5))
display(events.describe(include="all").T)


# <h2> Calculate minutes played </h2>
# Now that we have the data ready, we are going to calculate the number of minutes played per match for each player. For this, we will first assume that all players leaving/entering the pitch are recorded by sub_in or sub_out.

# In[ ]:


def minutes_played(group, disp=False):
    group = group.sort_values('ElapsedSeconds')
    last_event = group.tail(1)['EventType'].values[0]
    if last_event == 'sub_in':
        group.loc[0, ['ElapsedSeconds', 'EventType']] = (48*60, 'sub_out')
    group['Duration'] = group['ElapsedSeconds'].diff(1).fillna(group['ElapsedSeconds'])
    if disp:
        display(group)
    duration = group.loc[group['EventType'] == 'sub_out', 'Duration'].sum()
    return duration / 60


# In[ ]:


groups = events.loc[events['EventType'].isin(['sub_in', 'sub_out'])].groupby(['Season', 'DayNum', 'EventTeamID', 'EventPlayerID'])
with mp.Pool(4) as pool:
    min_played = pool.map(minutes_played, [group for _, group in groups])


# In[ ]:


mins_played = groups['EventID'].count().to_frame().reset_index()
mins_played['MinutesPlayed'] = min_played
display(mins_played.head(5))


# So far so good, but there are still some problems with the calculation as you can see in the below example. 

# In[ ]:


ev = events.loc[(events['EventType'].isin(['sub_in', 'sub_out'])) & (events['EventPlayerID']==602324)]
get_ipython().run_line_magic('prun', 'minutes_played(ev, disp=True)')


# This player apparently has two sub_in events after eachother. To me, this seems to be really strange. Let's check all the events for this player in the match.

# In[ ]:


events.loc[(events['EventPlayerID']==602324)].sort_values('ElapsedSeconds')


# There was a personal foul prior to the second sub_in. Could this have caused the second sub_in event?

# <h2> What next? </h2>
# The example shows that I either made a mistake in my logic, or there is a discrepancy in the data.
# 
# Any ideas are more then welcome!
