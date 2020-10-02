#!/usr/bin/env python
# coding: utf-8

# # Euro Championship Analysis
# 
# ## Table of Contents
# <ul>
# <li><a href="#intro">Introduction</a></li>
# <li><a href="#wrangling">Data Wrangling</a></li>
# <li><a href="#eda">Exploratory Data Analysis</a></li>
# </ul>

# <a id='intro'></a>
# ## Introduction
# 
# > Uefa Euro Championship Dataset contains data about all matches , all players participated , participated teams and general statistics for all teams participated in the famous football competition Euro which started in 1960 and played every 4 years since then . 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import plotly.graph_objects as go
import plotly.express as px
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id='wrangling'></a>
# ## Data Wrangling

# In[ ]:


matches = pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup All Matches.csv')
players = pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup All Players.csv')
teams_stats = pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup Participated Teams General Statistics.csv')
cups_stats=pd.read_csv('../input/uefa-euro-championship/Uefa Euro Cup General Statistics.csv')


# In[ ]:


matches.head()


# In[ ]:


matches.info()


# In[ ]:


players.head()


# In[ ]:


players.info()


# In[ ]:


teams_stats.head()


# In[ ]:


teams_stats.info()


# In[ ]:


cups_stats.head()


# In[ ]:


cups_stats.info()


# Clean HomeTeamName and AwayTeamName from any latin space 

# In[ ]:


matches['HomeTeamName'] = matches['HomeTeamName'].apply(lambda x : x.replace(u'\xa0', u'')).apply(lambda x : x.strip())
matches['AwayTeamName'] = matches['AwayTeamName'].apply(lambda x : x.replace(u'\xa0', u'')).apply(lambda x : x.strip())


# Replace old names of countries with the new ones 

# In[ ]:


matches.replace('Soviet Union','Russia',inplace=True)
matches.replace('West Germany','Germany',inplace=True)
cups_stats.replace('Soviet Union','Russia',inplace=True)
cups_stats.replace('West Germany','Germany',inplace=True)


# # Notes 
# 
# > Special win condition column consists mainly of missing values , since most of matches are in group stages or ended with a winner .
# 
# > most of players have null goals or caps , as these statistics wasn't recorded that time .
# 
# > Player of the Tournament has 6 values only ( Since it is a new prize ) .
# 
# > Some countries are referenced with there old names like soviet union 

# <a id='eda'></a>
# ## Exploratory Data Analysis

# Let's begin with number of matches played in each city 

# In[ ]:


games_by_city=matches.groupby(['City']).size()
games_by_city = games_by_city.reset_index()


# In[ ]:


games_by_city.head()
games_by_city.columns = ['City','Number of games']
games_by_city


# In[ ]:


top_cities = games_by_city.nlargest(10, ['Number of games']) 


# In[ ]:


plt.figure(figsize=(20,10))
fig = px.bar(top_cities, x='City', y='Number of games',color='Number of games')
fig.update_layout(title='Number of games played in each city in all tournaments',
                   xaxis_title='City',
                   yaxis_title='Number of Games')
fig.show()


# Next let's see city with maximum attendance in each tournament 

# In[ ]:


top_attendance=matches[['City','Attendance','Year']].groupby(['Year']).max()


# In[ ]:


top_attendance=top_attendance.reset_index()


# In[ ]:


plt.figure(figsize=(20,10))
fig = go.Figure(data=[go.Bar(
            x=top_attendance['Year'], y=top_attendance['Attendance'],
            text=top_attendance['City'],
            textposition='outside',
        )])
fig.update_layout(title='Most Attendance in the tournament',
                   xaxis_title='Year',
                   yaxis_title='Attendance')
fig.show()


# Now let's check the trend

# In[ ]:


time_plot_1=go.Figure(go.Scatter(x=top_attendance['Year'], y=top_attendance['Attendance'],
                                 mode='lines+markers', line={'color': 'red'}))
time_plot_1.update_layout(title='Most Attendance in the tournament',
                   xaxis_title='Year',
                   yaxis_title='Attendance')
#showing the figure
time_plot_1.show()


# Let's see win ratio for the host team in each tournament

# In[ ]:


matches.head()


# I will extract the year from Date column to be able to merge matches and cups_stats tables so i can get the host column . 

# In[ ]:


matches['Year']=matches['Date'].apply(lambda x : x.split('(')[0]).apply(lambda x : x.split()[-1]).astype(int)


# In[ ]:


matches.head()


# In[ ]:


merged_data = pd.merge(matches,cups_stats, on=['Year', 'Year'])


# In[ ]:


merged_data.head()


# get the necessary columns 

# In[ ]:


merged_data = merged_data[['Year','HomeTeamName','AwayTeamName','Host','HomeTeamGoals','AwayTeamGoals']]


# In[ ]:


merged_data['Host'].tolist()


# Split Multiple Hosts 

# In[ ]:


merged_data['Host']=merged_data['Host'].apply(lambda x : x.split())


# In[ ]:


splitted_data=pd.DataFrame([
    [year,Hometeam,awayteam, host, hgoals,agoals] for year,Hometeam,awayteam,Hosts, hgoals,agoals in merged_data.values
    for host in Hosts
], columns=merged_data.columns)


# Check for splitted hosts , i will check on 2012 tournament that hosted by poland and ukraine 

# In[ ]:


splitted_data[splitted_data['Year']==2012]


# successfully splitted ! now move on 

# In[ ]:


merged_data = splitted_data


# In[ ]:


merged_data = merged_data[(merged_data['HomeTeamName']==merged_data['Host'])|(merged_data['AwayTeamName']==merged_data['Host'])]


# In[ ]:


home_win = merged_data['HomeTeamGoals']>merged_data['AwayTeamGoals']
home_name = merged_data['HomeTeamName']==merged_data['Host']
away_win = merged_data['AwayTeamGoals']>merged_data['HomeTeamGoals']
away_name = merged_data['AwayTeamName']==merged_data['Host']


# In[ ]:


merged_data['Wins']=((home_win & home_name)|(away_win & away_name))


# In[ ]:


merged_data.head()


# Now that we have all games for hosts let's get percentage of winning 

# In[ ]:


final_data = merged_data.groupby(['Year','Host']).sum()['Wins']/merged_data.groupby(['Year']).count()['Wins']


# In[ ]:


final_data = final_data.reset_index()


# In[ ]:


final_data['Wins']=final_data['Wins'] * 100 


# In[ ]:


plt.figure(figsize=(20,10))
fig = px.bar(final_data, x='Year', y='Wins',color='Wins',hover_data=['Host'])
fig.update_layout(title='Win Percentage for the Hosts',
                   xaxis_title='Year',
                   yaxis_title='Win Percentage')
fig.show()

