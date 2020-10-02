#!/usr/bin/env python
# coding: utf-8

# # Is there enough data to create an accurate model by runner?

# We can hypothesize that the number of yards completed in a given play is a result of:
# 1. The **play dynamics**: players positions, velocities, accelerations, field spaces, where players are in the field, etc;
# 2. The **teams' dynamics**: offense and defense strategies, who is winning/losing, etc;
# 3. The **match dynamics**: where the game is happening, stadium type, turf, weather, temperature, etc;
# 4. The **teams' and players' SKILLS**: some players are just more trained/more experienced than others.
# 
# So far, I've seen several notebooks with great EDAs and basic models, but they all have one thing in commom: they focus on the first 3 points. Most work I've seen so far here are just dropping the specific data about **who are making the plays**.
# 
# > The **purpose** of this notebook is to start the discussion about whether or not it is a viable (and good) idea to **create an accurate model** to predict the probability of yards completed **by runner/team**, thus **taking into consideration players' skills into the modeling**.

# In[ ]:


import numpy as np
import pandas as pd

df = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)


# In[ ]:


unique_runners = df['NflIdRusher'].nunique()
unique_plays = df['PlayId'].nunique()
print(f'There are {unique_runners} runners, responsible for {unique_plays} plays.')


# Let's see who are these 371 runners, and how their results are different:

# In[ ]:


agg_dic = {'PlayId': 'count', 'Yards': ['mean', 'std', 'min', 'max']}
dfp = df[['NflIdRusher', 'PlayId', 'Yards']].groupby('NflIdRusher').agg(agg_dic).sort_values(('PlayId', 'count'), ascending=False)
dfp


# Looking at this table, it looks like **the number of plays are concentrated**, done by few runners. Let's check it out.

# In[ ]:


df_cum = dfp[[('PlayId', 'count')]].sort_values(('PlayId', 'count'), ascending=False)
df_cum.columns = ['count']
df_cum['cum'] = df_cum['count'].cumsum()
df_cum['perc'] = df_cum['cum'] / df_cum['cum'].iloc[-1]
df_cum.head(100)


# ## Over 90% of the plays are done by the top 100 runners
# For these players, we have well over 1200 datapoints for each, which indicates that the idea of training a specific model for those frequent runners might work.

# In[ ]:




