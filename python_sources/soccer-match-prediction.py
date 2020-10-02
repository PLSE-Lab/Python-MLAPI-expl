#!/usr/bin/env python
# coding: utf-8

# Idea:
# Can we predict the outcome of a game based on the players' statistics?
# 
# For each match, we'll have to figure out who was on each team and based on their statistics we can make a prediction.
# 
# TODO:
# Make a derived data set.
# 
# a. Each row will represent a match between team A and team B where we have the following columns:
# 
#  i. choice of stats for player 1 from team A, player 2 from team A, ..., 
#     player 1 from team B, player 2 from team B...
# 
# b. Need to decide which stats are useful, make an educated guess to start, we can perform a more systematic search after.
# 
# c. We can make a matrix to represent a team. Each row of the matrix corresponds to the statistics of a player on the team. Then a match would be a matrix multiply A * B = C, followed by a aggregation fuction that reduces C to probability. So we have to learn:
# 
#  i. weights for multiplying A * B, so really we have A * W * B  = C.
# 
#  ii. and then we need to find the right function f(C).
# 
#  iii. maybe W can be the player positions?  Then the formula would be A * W1 * B * W2 = C

# In[ ]:


import numpy as np
import pandas as pd
import sqlite3
import os
from sklearn import svm
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


class Data:
    """Class to store connection to soccer database.
    
    Contains methods for accessing tables from soccer database.
    
    """
    def __init__(self):
        self._database_connection = sqlite3.connect('/kaggle/input/soccer/database.sqlite')
        
        self.df = {}
        self.df["country_ids"] = self.table("Country")
        self.df["players"] = self.table("Player")
        self.df["players_attr"] = self.table("Player_Attributes")
        self.df["teams"] = self.table("Team")
        self.df["teams_attr"] = self.table("Team_Attributes")
        self.df["match"] = self.table("Match")
        self.df["league"] = self.table("League")
            
    def table(self, table_name):
        query = f"SELECT * FROM {table_name};"
        return pd.read_sql_query(query, self._database_connection)
    
    def player(self, player_id):
        """Returns player's stats."""
        basic = self.df["players"].query(f"player_api_id == '{player_id}'")
        print(basic.player_name)
        print(basic.height)
        print(basic.weight)
        return self.df["players_attr"].query(f"player_api_id == '{player_id}'")
    
soccer_data = Data()


# In[ ]:


soccer_data.player("95327")


# In[ ]:


# Select all matches played during the season 2013/2014
# Select all the matches played in England (English Premier League) during the season 2013/2014

class Matches:
    
    def __init__(self, data, season, league):
        """
        
        """
        self._df = data.table("Match")
        self._league_id = data.table("League").query(f"name == '{league}'")["country_id"]
        self._matches = (self._df.query(f"season == '{season}' and league_id == '{int(self._league_id)}'"))

    def get_matches(self, team):
        team_api_id = int(df["teams"].query(f"team_long_name == '{team}'")["team_api_id"])
        matches = self._matches.query(f"home_team_api_id == '{team_api_id}' or away_team_api_id == '{team_api_id}'")
        return matches
    
    def players_from_match(self, match):
        return "players from match"

    
matches = Matches(data=soccer_data, season="2013/2014", league="England Premier League")

# Select the matches played by Manchester United during this season 2013/2014
manchester_matches = matches.get_matches("Manchester United")

# Select the matches played by Liverpool during the season 2013/2014
liverpool_matches = matches.get_matches("Liverpool")

# Pick up the all players of the Liverpool and Manchester United during the season
players_liverpool = None
players_manchester = None

# Results of matches:
# For each match we have a home_team_goal and away_team_goals


# In[ ]:


def get_players(match):
    home_players_list = []
    away_players_list = []
    home_players = match.loc[:, "home_player_1":"home_player_11"]
    away_players = match.loc[:, "away_player_1":"away_player_11"]
    for (home, away) in zip(home_players, away_players):
        home_players_list.append(soccer_data.player(str(int(home))))
        away_players_list.append(soccer_data.player(str(int(away))))


# In[ ]:


liverpool_matches.apply(get_players, axis=0)


# # X
# 
# X will be a matrix, representing a match. Each row will comprise the following data:
# 
# ```
# X_team1 = pd.DataFrame({
#     player_id: player_stats
#     for (player_id, player_stats) in zip(player_id, team_stats)
# })
# 
# X_team2 = pd.DataFrame({
#     player_id: player_stats
#     for (player_id, player_stats) in zip(player_id, team_stats)
# })
# 
# X = pd.concat([X_team1, X_team2], axis=0, arrange=False)
# ```
# 
# # y
# 
# 

# In[ ]:




