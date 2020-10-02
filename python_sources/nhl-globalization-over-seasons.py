#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


# get files into dataframes that would be used in the process
df_player_info = pd.read_csv('../input/player_info.csv')
df_game_shifts = pd.read_csv('../input/game_shifts.csv')
df_game = pd.read_csv('../input/game.csv')


#  ### Checking out overall globalization sats

# In[ ]:


# checking out how many countries sent players to the NHL - overall
nations_count =df_player_info['nationality'].value_counts().count()
print('NHL players throughout the years came from {} nations'.format(nations_count))


# ### Showing leading countries by players from season 2012

# In[ ]:


plt.figure(figsize=(15,10))
nations_count_for_bar = df_player_info['nationality'].value_counts()
sns.barplot(nations_count_for_bar.index,nations_count_for_bar.values,palette="BuGn_r")
plt.title('Top countries by overall NHL players')


# ### Prepare data, in order to show NHL globalization per season by time playing

# In[ ]:


# Prepare time played
df_game_shifts['seconds_played'] = df_game_shifts['shift_end'] - df_game_shifts['shift_start']
df_time_played = df_game_shifts.groupby(['game_id','player_id'])['seconds_played'].sum().reset_index()
df_time_played['minutes_played'] = df_time_played['seconds_played'] / 60
df_time_played['hours_played'] = df_time_played['minutes_played'] / 60 
df_time_played.head(2)


# In[ ]:


# Prepare relevent data of games per season
df_season_games = df_game[['game_id','season']]
df_season_games.drop_duplicates() # in case of unexpected duplicate row
df_season_games.head(2)


# In[ ]:


# Prepare relevent data of player's nationality
df_player_info_nationality = df_player_info[['player_id','nationality']]
df_player_info_nationality['international'] =  df_player_info_nationality['nationality'].apply(
    lambda x: 'USA' if (x == 'USA') else 'CAN' if (x == 'CAN') else 'Rest')
df_player_info_nationality.head(2)


# In[ ]:


# Merge three tables relevent data into one table
df_nationality_by_season = pd.merge(df_time_played, df_season_games, how='left', on='game_id')
df_nationality_by_season = pd.merge(df_nationality_by_season, df_player_info_nationality, how='left', on='player_id')
df_nationality_by_season.head(2)


# ## Showing hours played by players from USA vs. CAN vs. Rest of the world by season

# In[ ]:


# remain only USA & CAN data
df_season_hours_by_inter = df_nationality_by_season.groupby(['season','international'])['hours_played'].sum().reset_index()
df_season_hours_by_inter_piv = df_season_hours_by_inter.pivot_table(values='hours_played', index=df_season_hours_by_inter.season,columns='international', aggfunc='first')

df_season_hours_by_inter_piv['season total'] = df_season_hours_by_inter_piv['Rest'] + df_season_hours_by_inter_piv['USA'] ++ df_season_hours_by_inter_piv['CAN']
df_season_hours_by_inter_piv['% Rest'] = df_season_hours_by_inter_piv['Rest'] / df_season_hours_by_inter_piv['season total'] * 100
df_season_hours_by_inter_piv['% USA'] = df_season_hours_by_inter_piv['USA'] / df_season_hours_by_inter_piv['season total'] * 100
df_season_hours_by_inter_piv['% CAN'] = df_season_hours_by_inter_piv['CAN'] / df_season_hours_by_inter_piv['season total'] * 100

df_season_hours_by_inter_piv= df_season_hours_by_inter_piv[['% Rest','% USA','% CAN']]


# In[ ]:


# let's see it also on bar chart
sns.set_style("whitegrid")
df_season_hours_sns = df_nationality_by_season.groupby(['season'])['hours_played'].sum().reset_index()
df_season_hours_by_inter_sns = pd.merge(df_season_hours_by_inter, df_season_hours_sns, how='left', on='season')
df_season_hours_by_inter_sns['% Hours played'] = df_season_hours_by_inter_sns['hours_played_x'] / df_season_hours_by_inter_sns['hours_played_y'] *100
g = sns.catplot(x="season", y="% Hours played", hue="international", data=df_season_hours_by_inter_sns,
                height=6, kind="bar", palette="muted")


# In[ ]:


# and by table view
df_season_hours_by_inter_piv


# ### Candians as expected are playing more then other nationalities in the league,  their dominance is declining and they played ~6% less in 2017-8 vs. 2012-3 season.
# ### USA players playing in each year more and more, and on 2017-8 they played 5% more then 2012-3 season 
# ### However, we can see that Rest of the world players playing around 30% of the time, with no signification changing over the years [~1% over 6 season]
# ### I can see that  the  globalization process hasn't arrived to the NHL at the past 6 years, and the league stayed in USA / CAN control.
