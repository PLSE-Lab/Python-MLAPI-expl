#!/usr/bin/env python
# coding: utf-8

# ### ** Cleaning Data Scraped from Baseball-Reference.com **

# In[1]:


import numpy as np
import pandas as pd


# In[2]:


df = pd.read_csv('../input/baseball_reference_2016_scrape.csv')


# stripping extraneous characters from columns in the dataframe

# In[3]:


df['attendance'] = df['attendance'].apply(lambda x: x.strip("]'"))
df['game_duration'] = df['game_duration'].apply(lambda x: x.strip(": "))
df['venue'] = df['venue'].apply(lambda x: x.strip(" :"))
df['start_time'] = df['start_time'].apply(lambda x: x.strip("Start Time: "))


# splitting columns from the dataframe

# In[4]:


df['attendance'] = df['attendance'].str.replace(' ', '')
df['attendance'] = df['attendance'].str.replace(',', '')
df['day_of_week'] = df['date'].str.split(',', 3, expand=True)[0]
df['game_type_remove'] = df['game_type']
df['game_type'] = df['game_type_remove'].str.split(',', 2, expand=True)[0]
df['field_type'] = df['game_type_remove'].str.split(',', 2, expand=True)[1]
df['field_type'] = df['field_type'].str.replace(' on', 'on')


# cleaning and adjusting the weather column, currently titled as 'other_info_string'
# 

# In[5]:


df['start_time_weather'] = df['other_info_string'].str.split('</strong> ', 5, expand=True)[5]

n = 0
for weather in df['start_time_weather']:
    if df.loc[n, 'start_time_weather']==None:
        df.loc[n, 'start_time_weather'] = df['other_info_string'].str.split('</strong> ', 5, expand=True)[4][n]
        n += 1
    else:
        n+= 1
        
df['temperature'] = df['start_time_weather'].str.split('&', 2, expand=True)[0]
df['start_time_weather1'] = df['start_time_weather'].str.split(', ', 3, expand=True)[1]
df['start_time_weather2'] = df['start_time_weather'].str.split('Wind ', 3, expand=True)[1]
df['start_time_weather3'] = df['start_time_weather2'].str.split('.', 2, expand=True)[0]
df['wind_speed'] = df['start_time_weather3'].str.split(', ', 2, expand=True)[0]
df['wind_speed'] = df['start_time_weather3'].str.split('mph', 2, expand=True)[0]
df['wind_speed'] = df['start_time_weather3'].str.split('mph', 2, expand=True)[0]
df['start_time_weather3'] = df['start_time_weather3'].str.split('mph', 2, expand=True)[1]
df['wind_direction'] = df['start_time_weather3'].str.split(', ', 2, expand=True)[0]
df['sky'] = df['start_time_weather3'].str.split(', ', 2, expand=True)[1]


# create a column for total runs

# In[6]:


df['total_runs'] = df['away_team_runs'] + df['home_team_runs']


# adjusting for missing data that caused misalignment in initial scrape

# In[7]:


df.loc[220, 'attendance'] = None
df.loc[220, 'game_duration'] = '3:18'
df.loc[220, 'game_type'] = 'Day Game'
df.loc[220, 'field_type'] = 'on grass'
df.loc[220, 'venue'] = 'Citi Field'

df.loc[1724, 'attendance'] = None
df.loc[1724, 'game_duration'] = '2:40'
df.loc[1724, 'game_type'] = 'Day Game'
df.loc[1724, 'field_type'] = 'on grass'
df.loc[1724, 'venue'] = 'PNC Park'

df.loc[1912, 'attendance'] = None
df.loc[1912, 'game_duration'] = '3:10'
df.loc[1912, 'game_type'] = 'Day Game'
df.loc[1912, 'field_type'] = 'on grass'
df.loc[1912, 'venue'] = 'U.S. Cellular Field'


# changing data types

# In[8]:


df['attendance'] = df['attendance'].astype(float)
df['date'] = pd.to_datetime(df['date'])
df['temperature'] = df['temperature'].astype(float)
df['wind_speed'] = df['wind_speed'].astype(float)
df['game_hours_dec'] = df['game_duration'].str.split(':', 2, expand=True)[1].astype(float)/60 + df['game_duration'].str.split(':', 2, expand=True)[0].astype(float)


# filling in missing data

# In[9]:


df['sky'] = df['sky'].astype(object).fillna('Unknown')

n = 0
for wind_direction in df['wind_direction']:
    if df.loc[n, 'wind_direction']=='':
        df.loc[n, 'wind_direction'] = ' in unknown direction'
        n += 1
    elif df.loc[n, 'wind_direction']==' ':
        df.loc[n, 'wind_direction'] = ' in unknown direction'
        n += 1
    else:
        n+= 1


# dropping columns from the dataframe that will not be needed

# In[10]:


df.drop(['boxscore_url','game_duration','game_type_remove','other_info_string','start_time_weather','start_time_weather1','start_time_weather2','start_time_weather3'],axis=1,inplace=True)


# creating a new field to differentiate between reular season and post season games

# In[11]:


df['season'] = 0
n = 0
for date in df['date']:
    if df.loc[n, 'date'].month==10 and df['date'][n].day > 2:
        df.loc[n, 'season'] = 'post season'
        n += 1
    elif df.loc[n, 'date'].month==11:
        df.loc[n, 'season'] = 'post season'
        n += 1
    else:
        df.loc[n, 'season'] = 'regular season'
        n += 1


# create a column to count the number of home team wins

# In[12]:


df['home_team_win'] = 0
n = 0
for win in df['home_team_win']:
    if df.loc[n, 'home_team_runs'] > df['away_team_runs'][n]:
        df.loc[n, 'home_team_win'] = 1
        n += 1
    else:
        df.loc[n, 'home_team_win'] = 0
        n += 1


# create a column to count the number of home team losses

# In[13]:


df['home_team_loss'] = 0
n = 0
for win in df['home_team_loss']:
    if df.loc[n, 'home_team_runs'] < df['away_team_runs'][n]:
        df.loc[n, 'home_team_loss'] = 1
        n += 1
    else:
        df.loc[n, 'home_team_loss'] = 0
        n += 1


# create a column to state a home team win or loss

# In[14]:


df['home_team_outcome'] = 0
n = 0
for win in df['home_team_outcome']:
    if df.loc[n, 'home_team_runs'] > df['away_team_runs'][n]:
        df.loc[n, 'home_team_outcome'] = 'Win'
        n += 1
    else:
        df.loc[n, 'home_team_outcome'] = 'Loss'
        n += 1


# In[15]:


df.info()


# export cleaned data to csv

# In[16]:


df.to_csv('baseball_reference_2016_clean.csv')

