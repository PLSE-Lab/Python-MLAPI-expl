#!/usr/bin/env python
# coding: utf-8

# ### ** Answering Top-Level Questions with Pandas Dataframes **

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('../input/baseball_reference_2016_clean.csv',index_col=0)


# changing data types

# In[3]:


df['attendance'] = df['attendance'].astype(float)
df['date'] = pd.to_datetime(df['date'])
df['temperature'] = df['temperature'].astype(float)
df['wind_speed'] = df['wind_speed'].astype(float)


# What game had the highest attendance in 2016?

# In[4]:


df[df['attendance']==df['attendance'].max()]


# What was the hottest game of the year?

# In[5]:


df[df['temperature']==df['temperature'].max()]


# What was the coldest game of the year?

# In[6]:


df[df['temperature']==df['temperature'].min()]


# What was the longest game of the year?

# In[7]:


df[df['game_hours_dec']==df['game_hours_dec'].max()]


# What was the shortest game of the year?

# In[8]:


df[df['game_hours_dec']==df['game_hours_dec'].min()]


# How many games ended in a tie in the 2016 season?

# In[9]:


df[df['home_team_runs']==df['away_team_runs']].count()[1]


# What was the last game played of the season?

# In[10]:


df[df['date']==df['date'].dt.date.max()]


# What game had the highest attendance?

# In[11]:


df[df['attendance']==df['attendance'].max()]


# What game had the lowest attendance?

# In[12]:


df[df['attendance']==df['attendance'].min()]


# What was the windiest game of the year?

# In[13]:


df[df['wind_speed']==df['wind_speed'].max()]


# What was the highest scoring game of the year?

# In[14]:


df[df['away_team_runs'] + df['home_team_runs']==29]


# What game had the most errors?

# In[15]:


df['total_errors'] = df['away_team_errors'] + df['home_team_errors']
df[df['total_errors']==df['total_errors'].max()]


# What game had the most runs?

# In[16]:


df[df['total_runs']==df['total_runs'].max()]


# How many games played by each team are in this dataset?

# In[17]:


df['away_team'].value_counts() + df['home_team'].value_counts()[1]


# Which team won the most games in 2016?
# 
# *create a dataframe for regular season games*

# In[18]:


reg_season = df[df['season']=='regular season']


# *create a dataframe for regular season wins*

# In[19]:


reg_wins = pd.DataFrame(reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['home_team'].value_counts() + reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['away_team'].value_counts())
reg_wins.set_axis(['wins'],axis='columns',inplace=True)
reg_wins.index.name = 'team'
reg_wins.sort_values(by='wins',ascending=False).head(1)


# Which team won the most home games in 2016?
# 
# *create a dataframe for regular season home wins*

# In[20]:


reg_home_wins = pd.DataFrame(reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['home_team'].value_counts())
reg_home_wins.set_axis(['home_wins'],axis='columns',inplace=True)
reg_home_wins.index.name = 'team'
reg_home_wins.sort_values(by='home_wins',ascending=False).head(1)


# Which team lost the most games in 2016?
# 
# *create a dataframe for regular season losses*

# In[21]:


reg_losses = pd.DataFrame(reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['home_team'].value_counts() + reg_season[reg_season['home_team_runs'] > reg_season['away_team_runs']]['away_team'].value_counts())
reg_losses.set_axis(['losses'],axis='columns',inplace=True)
reg_losses.index.name = 'team'
reg_losses.sort_values(by="losses",ascending=False).head(1)


# Which team lost the most home games in 2016?
# 
# *create a dataframe for regular season home losses*

# In[22]:


reg_home_losses = pd.DataFrame(reg_season[reg_season['home_team_runs'] < reg_season['away_team_runs']]['home_team'].value_counts())
reg_home_losses.set_axis(['home_losses'],axis='columns',inplace=True)
reg_home_losses.index.name = 'team'
reg_home_losses.sort_values(by='home_losses',ascending=False).head(1)

