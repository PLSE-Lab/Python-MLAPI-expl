#!/usr/bin/env python
# coding: utf-8

# # The ultimate Soccer database for data analysis and machine learning
# 
# What is Given:
# 
#    - 25,000 matches
#    -  +10,000 players
#    -  11 European Countries with their lead championship
#    -  Seasons 2008 to 2016
#    -  Players and Teams' attributes* sourced from EA Sports' FIFA video game series, including the weekly updates
#    -  Team line up with squad formation (X, Y coordinates)
#    -  Betting odds from up to 10 providers
#    -  Detailed match events (goal types, possession, corner, cross, fouls, cards etc...) for +10,000 matches
#   
# 
# 
# ****
# # My goal
# 
# - Finding Top Players from 2008 to 2016
# - Finding Top Teams from 2008 to 2016
# - Comparing Leagues the Leagues with Each Other and Itself.
# 

# # Loading packages and data
# ****

# In[ ]:


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sqlite3
import datetime as dt


# Importing the dataset
conn = sqlite3.connect('../input/database.sqlite')


# In[ ]:


#Making a Connection for execution of SQL Commands
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
#list of all the tables in the schema 
print(cursor.fetchall())


# ## Creating all the DataFrames from tables present in the DataBase

# In[ ]:


#making DataFrame from DataBase
df_league = pd.read_sql_query("select * from League;", conn)
df_league.head()
df_player_attributes = pd.read_sql_query("select * from Player_Attributes;", conn)
df_player = pd.read_sql_query("select * from Player;", conn)
df_match = pd.read_sql_query("select * from Match;", conn)
df_country = pd.read_sql_query("select * from Country;", conn)
df_team_attributes = pd.read_sql_query("select * from Team_Attributes;", conn)
df_team = pd.read_sql_query("select * from Team;", conn)


# In[ ]:


#inspecting the Player_Attributes table
df_player_attributes.head(1)


# In[ ]:


#inspecting the Player table
df_player.head(1)


# In[ ]:


#inspecting the Country table
df_country.head(1)


# In[ ]:


#inspecting the Team_Attributes table
df_team_attributes.head(1)


# In[ ]:


#inspecting the Team table
df_team.head(1)


# In[ ]:


#inspecting the Match table
df_match.head(1)


#  # Database Schema
#  
# ![database-diagram.jpeg](https://raw.githubusercontent.com/abhiksark/UD-ND/master/football_analysis/database-diagram.jpeg)
# 
# This is a complex DataBase Having Many Columns. To crearly Understand the database
# I created the above DataBase Schema E-R Diagram using [Schemacrawler](https://www.schemacrawler.com/). It's a free open source tool for making E-R Diagrams from the Database. You can visit their github page for more Information.
# 
# E-R Diagram gives a indepth look into the Database and It's Structure.

# # Cleaning Data
# ## Checking NULL Data and taking filtering the fields that are being used 
# 
# 

# In[ ]:


df_match.isna().sum()
df_match = df_match[['country_id', 'league_id', 'season', 'stage', 'date', 'match_api_id', 'home_team_api_id', 'away_team_api_id', 'home_player_1',
       'home_player_2', 'home_player_3', 'home_player_4', 'home_player_5','home_player_6', 'home_player_7', 'home_player_8', 'home_player_9',
       'home_player_10', 'home_player_11', 'away_player_1', 'away_player_2','away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
       'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10','away_player_11']]
df_match.isna().sum()


# From the E-R Diagram above we can see that **player_api_id** is not directly realted with the **team_api_id**.
# To relate them together we will use the **Match** Table .
# As, there are a total of 22 mapping from **Player->Team** so I will drop them later. 

# In[ ]:


df_team_attributes = df_team_attributes[['id', 'team_api_id', 'date']]
df_team_attributes.isna().sum()


# In[ ]:


df_team.isna().sum()
df_team = df_team[['id', 'team_api_id', 'team_long_name']]
df_team.isna().sum()


# In[ ]:


df_player.isna().sum()
df_player = df_player[['player_api_id', 'player_name' ]]


# In[ ]:


df_player_attributes = df_player_attributes[[ 'player_api_id', 'date', 'overall_rating','potential']]
df_player_attributes.isna().sum()


# As we will rating the player on the basis of overall_rating. We will be dropping all the NULLs present in the overall_rating column from the DataFrame.

# In[ ]:


df_player_attributes = df_player_attributes.dropna()
df_player_attributes.isna().sum()


# In[ ]:


df_player.head(1)


# In[ ]:


df_player_attributes.head(1) 


# ## Combining DataFrames to Have Player Name and Player attributes in a Single Data Frame

# In[ ]:


#Merging Two Tables
df_comb_player =  pd.merge(df_player, df_player_attributes, on="player_api_id")
df_comb_player.head(1)


# In[ ]:


#converting Object to DateTime so we can run the Date Queries using DateTime
df_comb_player['date'] = pd.to_datetime(df_comb_player['date'])


# # Finding Top Players
# 
# The First Task was to find the top 10(n) players at the end of the year. 
# In the given dataset we are given mutiple attributes column for the same player with a different Dates. To keep things simple we will use only the Last Updated Attribute of the player for that year.
# 
# ### Steps for finding the Top 10 Player
#     1) Filtering the players by year 
#     2) Taking the last updated intance of the player attributes of that year
#     3) Sorting the Players by overall_rating column, if overall_rating for any two player is same we will consider the potential column.
#     4) Returning the top 10 values from the sorted DataFrame.

# In[ ]:


def top_N_players(df,year,n=10):
    """Function which return N Number of Top Player at end of Year"""
    df_top = df[df['date'].dt.year == year]
    df_top = df_top.sort_values('date').groupby('player_api_id').last() #Taking only last Instance
    df_top = df_top.sort_values(['overall_rating','potential']).tail(n)
    df_top = df_top.sort_values(['overall_rating','potential'],ascending=False)
    return df_top


# ### Top 10 Players from 2008-2016 on the Basis of given Data

# In[ ]:


df = top_N_players(df_comb_player,2016)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank1"},inplace=True)
df_rank =df[:5]
df


# In[ ]:


df = top_N_players(df_comb_player,2015)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank2"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2014)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank3"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2013)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank4"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2012)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank5"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2011)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank6"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2010)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank7"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2009)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank8"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2008)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank9"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# In[ ]:



df = top_N_players(df_comb_player,2007)[['player_name']]
df.reset_index(level=0, inplace=True)
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank10"},inplace=True)
df_rank = pd.merge(df_rank,df[:5],on=["player_api_id","player_name"],how='outer')
df


# After getting all the ranks of the players. Our Next task is to visualize the players year by year performance in a line plot. 
# 
# To make a line plot we will need to change the DataFrame according to the format accepted by df.plot() function.
# We will only make Line Plot of top 5 Players as the graph was getting a lot of players, which was making plot very crowded.

# In[ ]:


#Making a Line Plot of top 15 Players
df_rank.index = df_rank.player_name
#Taking only the rank in consideration
df_rank = df_rank[[ 'rank1', 'rank2','rank3', 'rank4', 'rank5', 'rank6', 'rank7', 'rank8', 'rank9','rank10']]


# In[ ]:


df_rank = df_rank.replace(np.NaN,6) # Replacing the NaNs with 11 to better represent the graph
df_rank['sum_rank'] = df_rank[[ 'rank1', 'rank2','rank3', 'rank4', 'rank5', 'rank6', 'rank7', 'rank8', 'rank9','rank10']].sum(axis=1)


# In[ ]:


df_rank =  df_rank.sort_values('sum_rank').head(10) #sorting and taking only top 15 values 
thickness = df_rank.sum_rank
df_rank  = df_rank.drop('sum_rank',axis=1)
df1_transposed = df_rank.T


# In[ ]:


df1_transposed


# In[ ]:


df1_transposed.plot(kind='line',figsize=(15,15), marker='o')
plt.gca().invert_yaxis() #inverting y axis
plt.gca().invert_xaxis() #inverting y axis
plt.yticks(range(1,6))
ind = np.arange(10) 
plt.xticks(ind, ("2016","2015","2014","2013","2012","2011","2010","2009","2008","2007"))
plt.show();


# # Finding Top Teams 
# 
# The Second Task was to find the top 5(n) teams at the end of the year. 
# As Team is not having a direct rating column. We will take the top 15 players of the team and sum all the overall rating of the players to have a total rating of the team.
# 
# ### Steps for finding the Top 5 Team Per Year
#     1) Filtering the players by year 
#     2) Taking the last updated intance of the player attributes of that year
#     3) Making a Table to map team_api_id to player_api_id
#     4) Merging the player_api_id with team_api_id
#     5) Grouping by team_api_id and summing the top 15 players of that team.
#     6) Sorting the grouped values and returning the top 5 teams by total rating.

# Combining Teams with Team Attributes to get the name of the team

# In[ ]:


df_comb_team = pd.merge(df_team, df_team_attributes, on="team_api_id")
df_comb_team.head(1)


# In[ ]:


df_comb_team['date'] = pd.to_datetime(df_comb_team['date'])


# In[ ]:


df_comb_team_2015 = df_comb_team[df_comb_team['date'].dt.year == 2015]
df_comb_team_2015 = df_comb_team_2015.sort_values('date').groupby('team_api_id').last()


# In[ ]:


df_comb_team_2015.head()


# In[ ]:


# making the fuctions for usage

def end_of_year_player(df_comb_player,year):
    df_comb_player['date'] = pd.to_datetime(df_comb_player['date'])
    df_comb_player = df_comb_player[df_comb_player['date'].dt.year == year]
    df_comb_player = df_comb_player.sort_values('date').groupby('player_api_id').last()    
    df_comb_player.reset_index(level=0, inplace=True)
    return df_comb_player[['player_api_id','player_name', 'date', 'overall_rating', 'potential']]


def end_of_year_team(df_comb_team):
    df_comb_team = df_comb_team.sort_values('date').groupby('team_api_id').last()
    df_comb_team.reset_index(level=0, inplace=True)
    return df_comb_team[['team_api_id','team_long_name','date']]


def team_to_player_home(df_match,year):
    players_list_home = ['date','home_team_api_id','home_player_1', 'home_player_2', 'home_player_3',
   'home_player_4', 'home_player_5', 'home_player_6', 'home_player_7',
   'home_player_8', 'home_player_9', 'home_player_10', 'home_player_11']
    df_match = df_match.loc[:,players_list_home]
    df_match['date'] = pd.to_datetime(df_match['date'])
    df_match = df_match[df_match['date'].dt.year == year]
    df_match = df_match.drop(['date'],axis=1)
    df_team_to_player=df_match.melt(['home_team_api_id']).sort_values('home_team_api_id')
    df_team_to_player = df_team_to_player[["home_team_api_id","value"]]
    df_team_to_player.rename( columns={"value":"player_api_id", "home_team_api_id":"team_api_id" },inplace=True)
    df_team_to_player = df_team_to_player.drop_duplicates()
    df_team_to_player = df_team_to_player.dropna()
    return df_team_to_player

def team_to_player_away(df_match,year):
    players_list_away = [ 'date','away_team_api_id','away_player_1', 'away_player_2','away_player_3', 'away_player_4', 'away_player_5', 'away_player_6',
       'away_player_7', 'away_player_8', 'away_player_9', 'away_player_10','away_player_11']
    df_match = df_match.loc[:,players_list_away]
    df_match['date'] = pd.to_datetime(df_match['date'])
    df_match = df_match[df_match['date'].dt.year == year]
    df_match = df_match.drop(['date'],axis=1)
    df_team_to_player=df_match.melt(['away_team_api_id']).sort_values('away_team_api_id')
    df_team_to_player = df_team_to_player[["away_team_api_id","value"]]
    df_team_to_player.rename( columns={"value":"player_api_id", "away_team_api_id":"team_api_id" },inplace=True)
    df_team_to_player = df_team_to_player.drop_duplicates()
    df_team_to_player = df_team_to_player.dropna()
    return df_team_to_player

def team_to_player(df_match,year):    
    df_2 = team_to_player_home(df_match,year)
    df_1 = team_to_player_away(df_match,year)
    df_combined = [df_1,df_2]
    result = pd.concat(df_combined)
    result = result.drop_duplicates()
    return result
    
def top_N_team(df_comb_team,df_comb_player,df_match,season="2015/2016",n=5):
    year = int(season.split("/")[0])
    df_end_of_year_team = end_of_year_team(df_comb_team)
    df_end_of_year_player = end_of_year_player(df_comb_player,year)
    df_team_to_player = team_to_player(df_match,year)
    df_end_of_year_player = pd.merge(df_end_of_year_player, df_team_to_player, on="player_api_id")
    df_comb_player_team_group= df_end_of_year_player.sort_values('overall_rating').groupby('team_api_id').head(16)
    df_comb_player_team_group = df_comb_player_team_group.sort_values('overall_rating').groupby('team_api_id').sum()
    df_top = pd.merge(df_comb_player_team_group,df_end_of_year_team,on="team_api_id")
    df_top = df_top[["team_api_id","overall_rating","team_long_name"]]
    df_top = df_top.sort_values("overall_rating")
    df_top = df_top[-n:]
    df_top = df_top.sort_values("overall_rating",ascending=False)
    return df_top


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2015/2016")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank1"},inplace=True)
df_rank = df
df


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2014/2015")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank2"},inplace=True)
df_rank = pd.merge(df_rank,df,on=["team_api_id","team_long_name"],how='outer')
df


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2013/2014")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank3"},inplace=True)
df_rank = pd.merge(df_rank,df,on=["team_api_id","team_long_name"],how='outer')
df


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2012/2013")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank4"},inplace=True)
df_rank = pd.merge(df_rank,df,on=["team_api_id","team_long_name"],how='outer')
df


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2011/2012")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank5"},inplace=True)
df_rank = pd.merge(df_rank,df,on=["team_api_id","team_long_name"],how='outer')
df


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2010/2011")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank6"},inplace=True)
df_rank = pd.merge(df_rank,df,on=["team_api_id","team_long_name"],how='outer')
df


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2009/2010")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank7"},inplace=True)
df_rank = pd.merge(df_rank,df,on=["team_api_id","team_long_name"],how='outer')
df


# In[ ]:


df = top_N_team(df_comb_team,df_comb_player,df_match,season="2008/2009")
df.index = range(1,len(df.index)+1)
df.reset_index(level=0, inplace=True)
df.rename( columns={"index":"rank8"},inplace=True)
df_rank = pd.merge(df_rank,df,on=["team_api_id","team_long_name"],how='outer')
df


# In[ ]:


df_rank = df_rank.replace(np.NaN,6)
df_rank.index = df_rank.team_long_name
df_rank = df_rank[[ 'rank1', 'rank2',
       'rank3', 'rank4', 'rank5', 'rank6', 'rank7', 'rank8']]


# In[ ]:


#df.pivot(index = "team_long_name")
df1_transposed = df_rank.T


# In[ ]:


df1_transposed


# After getting all the ranks of the teams. Our Next task is to visualize the players year by year performance in a line plot. 
# 
# To make a line plot we will need to change the DataFrame according to the format accepted by df.plot() function.

# In[ ]:


df1_transposed.plot(kind='line',figsize=(15,15), marker='o')
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()
plt.yticks(range(1,6))
ind = np.arange(8) 
plt.xticks(ind, ("2016","2015","2014","2013","2012","2011","2010","2009"))
plt.show();


# In[ ]:


def league_to_team(df_match,year):
    df_match = df_match.loc[:,["date","league_id","home_team_api_id","away_team_api_id"]]
    df_match['date'] = pd.to_datetime(df_match['date'])
    df_match = df_match[df_match['date'].dt.year == year]
    df_match = df_match.drop('date',axis=1)
    df_match=df_match.melt(['league_id'])
    df_match = df_match.drop('variable',axis=1)
    df_match.rename( columns={"value":"team_api_id" },inplace=True)
    df_match = df_match.drop_duplicates()
    return df_match


# In[ ]:


def top_leagues(df_comb_team,df_comb_player,df_match,season="2015/2016"):
    year = int(season.split("/")[0])
    df_end_of_year_team = end_of_year_team(df_comb_team)
    df_end_of_year_player = end_of_year_player(df_comb_player,year)
    df_team_to_player = team_to_player(df_match,year)
    df_end_of_year_player = pd.merge(df_end_of_year_player, df_team_to_player, on="player_api_id")
    df_comb_player_team_group= df_end_of_year_player.sort_values('overall_rating').groupby('team_api_id').head(16)
    df_comb_player_team_group = df_comb_player_team_group.sort_values('overall_rating').groupby('team_api_id').sum()
    df_top = pd.merge(df_comb_player_team_group,df_end_of_year_team,on="team_api_id")
    df_top = df_top[["team_api_id","overall_rating","team_long_name"]]
    df_top = df_top.sort_values("overall_rating")
    df_top = df_top.sort_values("overall_rating",ascending=False)
    df_league_to_team = league_to_team(df_match,year)
    df_top = pd.merge(df_league_to_team,df_top,on="team_api_id")
    return df_top


# In[ ]:


df = top_leagues(df_comb_team,df_comb_player,df_match,season="2015/2016")
df = df.groupby("league_id").sum()
df.reset_index(level=0, inplace=True)
df_league.rename( columns={"id":"league_id" },inplace=True)
df = pd.merge(df,df_league,on="league_id")
df.sort_values("overall_rating")


# In[ ]:


df = top_leagues(df_comb_team,df_comb_player,df_match,season="2015/2016")
df = df.groupby("league_id").median()
df.reset_index(level=0, inplace=True)
df = pd.merge(df,df_league,on="league_id")
df_league.rename( columns={"id":"league_id" },inplace=True)
df.sort_values("overall_rating")


# In[ ]:


df = top_leagues(df_comb_team,df_comb_player,df_match,season="2015/2016")
df = df.groupby("league_id").std()
df.reset_index(level=0, inplace=True)
df = pd.merge(df,df_league,on="league_id")
df_league.rename( columns={"id":"league_id" },inplace=True)
df.sort_values("overall_rating")


# # Comparing Different Leagues
# 
# Our Last Task is to compare the different leagues with each other. To perform this analysis we used group by league_id and merging them with league table to get the names of the league.
# 
# ### Our First analysis was to Year on Year Comparision of the leagues with each other.

# In[ ]:


import seaborn as sns


# In[ ]:


def box_plot_leagues(df_comb_team,df_comb_player,df_match,season="2014/2015"):
    df = top_leagues(df_comb_team,df_comb_player,df_match,season=season)
    df = pd.merge(df,df_league,on="league_id")
    df_league.rename( columns={"id":"league_id" },inplace=True)
    df.sort_values("overall_rating")
    plt.figure(figsize=(10,10))
    my_palette = sns.color_palette("Paired", 11)
    ax = sns.boxplot(x="overall_rating", y="name", data=df,palette= my_palette)
    #ax = sns.swarmplot(x="overall_rating", y="name", data=df, color ='aqua',size=4)
    #sns.boxplot(data=df,x='overall_rating',y='name')
    plt.title(season)
    plt.show()


# In[ ]:


for i in range(2008,2016):
    season = str(i)+"/"+str(i+1)
    box_plot_leagues(df_comb_team,df_comb_player,df_match,season)


# ### Our Second analysis was to Year on Year Comparision of the leagues with itself. To check how different leagues are perfoming.

# In[ ]:


import random
def box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,country):
    df_plot = pd.DataFrame()
    for i in range(2008,2016):
        season = str(i)+"/"+str(i+1)
        df = top_leagues(df_comb_team,df_comb_player,df_match,season)
        df = pd.merge(df,df_league,on="league_id")
        df = df[df.name.str.contains(country)]
        df["season"] = str(season)
        df_plot = df_plot.append(df, ignore_index=True)
    plt.figure(figsize=(10,10))
    my_palette = [(random.random(),random.random(),random.random())]
    ax = sns.boxplot(x="overall_rating", y="season", data=df_plot,palette= my_palette) 
    plt.title(country)
    plt.show()


# In[ ]:





# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"England")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Germany")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"France")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Italy")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Poland")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Spain")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Belgium")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Scotland")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Portugal")


# In[ ]:


box_plot_leagues_through_years(df_comb_team,df_comb_player,df_match,"Netherlands")


# 
# 
# Please upvote this kernel. This is still on Progress. Your comments on how we can improve this kernel is welcome. Thanks.
# 
