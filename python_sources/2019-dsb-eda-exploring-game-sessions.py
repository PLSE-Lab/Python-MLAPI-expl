#!/usr/bin/env python
# coding: utf-8

# # Exploring Game Sessions and Installations (2019 DSB EDA)

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


PATH = '/kaggle/input/data-science-bowl-2019/'


# In[ ]:


df_train = pd.read_csv(f'{PATH}train.csv')


# In[ ]:


all_installation_ids = df_train['installation_id'].drop_duplicates()
all_installation_ids.count()


# In[ ]:


ids_to_keep = df_train[df_train.type == 'Assessment']['installation_id'].drop_duplicates()


# In[ ]:


df_train = pd.merge(df_train, ids_to_keep, on='installation_id', how='inner')


# ### How to Sample Consistently?
# I should probably sample from installation ids to keep all observations from one installation ID together. How can I do that?
# 
# Oh, its actually relatively simple, I take all the installation ids and then I sample from them.

# In[ ]:


ids_to_keep.count()


# let's keep 800 installation IDs to make our analysis run faster (~20% of them)

# In[ ]:


sample_installation_ids = ids_to_keep.sample(800)
df_sample = pd.merge(df_train, sample_installation_ids, on='installation_id', how='inner')


# In[ ]:


df_sample.describe()


# In[ ]:


df_sample.head()


# In[ ]:


df_sample['timestamp'] = pd.to_datetime(df_sample['timestamp'])


# In[ ]:


df_sample.groupby('installation_id')     .count()['event_id']     .plot(kind='hist',
          bins=100,
          figsize=(15, 5),
         title='Count of Observations by installation_id')
plt.show()


# ### What Kinds of Events Do We Have?

# In[ ]:


df_sample.groupby('type')     .count()['event_id']     .plot(kind='barh',
          figsize=(15, 5),
         title='Count of Event Types')
plt.show()


# ### How many cases do we have where there is a single assessment without prior events?

# In[ ]:


for act_type in ['Assessment', 'Clip', 'Game', 'Activity']:
    df_sample[df_sample['type'] == act_type].groupby('installation_id')         .count()['event_id']        .plot(kind='hist',
            bins=100,
            figsize=(12, 4),
            title=f'{act_type} by installation_id')
    plt.show()


# ### For how many days do kids play the game?

# In[ ]:


df_sample.groupby('installation_id')['timestamp']    .transform(lambda x: (x.max() - x.min()).days)    .plot(kind='hist',
          bins=100,
          figsize=(12, 4),
          title=f'Time played by installation_id')
plt.show()


# ### How man game sessions are there per installation_id?

# In[ ]:


df_sample.groupby('installation_id')['game_session']    .transform(lambda x: x.nunique())    .plot(
        kind='hist',
        bins=100,
        figsize=(12, 4),
        title='Game sessions per installation id')
plt.show()


# ### How is the relationship between number of game sessions and days plaid?

# In[ ]:


game_sessions = df_sample.groupby('installation_id')['game_session'].nunique()


# In[ ]:


dates_plaid = df_sample.groupby('installation_id')['timestamp'].apply(lambda x: (x.max() - x.min()).days)


# In[ ]:


plt.scatter(
    dates_plaid, 
    game_sessions)
plt.xlabel('days played')
plt.ylabel('game sessions')
plt.title('Game sessions vs days played')


# ### For how long is the game played at a time?
# Since we have the ID of an individual game session, we can analyze how long the game is played at a time

# In[ ]:


sample_session_length = df_sample.groupby('game_session')['timestamp']    .transform(lambda x: (x.max() - x.min()).delta / 60_000_000_000)
sample_session_length = sample_session_length[sample_session_length <= 40]

sample_session_length.plot(kind='hist',
    bins=100,
    figsize=(12, 4),
    title=f'Session length in minutes (sessions under 40 minutes)')
plt.show()


# ### How is the Correlation Between Play Time and Event Count?

# In[ ]:


session_length = df_sample.groupby('game_session')['timestamp']    .apply(lambda x: (x.max() - x.min()).delta / 60_000_000_000)
session_count = df_sample.groupby('game_session')['event_id']    .apply(lambda x: x.count())


# In[ ]:


scatter_index = (session_length <= 120)
plt.figure(figsize=(15, 9))
plt.scatter(
    session_length[scatter_index], 
    session_count[scatter_index], 
    alpha=0.05)
plt.title('Game session length vs game event count')
plt.xlabel('Game session length in minutes')
plt.ylabel('Game event count')
plt.show()

