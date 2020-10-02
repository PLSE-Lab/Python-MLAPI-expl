#!/usr/bin/env python
# coding: utf-8

# <h1 style="font-size:35px">Predicted Final Positions</h1>
# In this notebook we look at the current league table as of the suspension of the Premier League due to Coronavirus. We will look at how that compares to a league based upon expected goals (scored and conceded), and expected points. We will then look at the recent form of teams up to the suspension of the league, and use that to predict the final league table.
# 
# Scroll down to the bottom of the notebook if you are interested to see our final predictions for champions, Champions League qualification, Europa League qualification, and relegation.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[ ]:


DATA_DIR = '../input/epl-stats-20192020/'
POSITIONS = np.array(range(1, 21))


# # Import Data

# In[ ]:


game_data = pd.read_csv(DATA_DIR + 'epl2020.csv')
print(game_data.columns)
game_data.head(10)


# # How Does The Current Table Look?
# To get a current look at the table we extract wins, draws, losses, goals scored, and goals conceded from the dataset. Games, points, and goal difference can then be calculated from these values. We use these to show the current league table.

# In[ ]:


# Get the list of teams
teams = game_data['teamId'].unique()

# Get the results for each team
team_results = []
for team in teams:
    # Get the data for that team
    team_data = game_data[game_data['teamId'] == team]
    
    wins = team_data['wins'].sum()
    draws = team_data['draws'].sum()
    losses = team_data['loses'].sum()
    scored = team_data['scored'].sum()
    conceded = team_data['missed'].sum()
    games = wins + draws + losses
    points = (3 * wins) + draws
    goal_difference = scored - conceded
    
    team_results.append([team, games, wins, draws, losses, scored, conceded, goal_difference, points])

league_table = pd.DataFrame(team_results, columns=['Team', 'P', 'W', 'D', 'L', 'F', 'A', 'GD', 'Points'])
league_table.sort_values(by=['Points', 'GD', 'F'], ascending=False, inplace=True, ignore_index=True)
league_table.set_index(POSITIONS, inplace=True)
league_table.head(20)


# # Where Would Teams Be On Expected Goals?
# Expected goals scored and conceded take into account how many goals there should be by each team in a game based upon their performance. We can use these two values to create another table. Comparing goals scored and goals conceded for each game allows us to work out the result, and with that we can calculate the table in a similar manner to before.
# 
# To do this, we must first check who won each game based upon the expected goals. Since expected goals can be a decimal number, we round them to the nearest whole number. Without some method of rounding, we are unlikely to see any draws in this data (which we know are actually common in football). The first cell deals with calculations for each game, and the second cell produces the new table.

# In[ ]:


# Get a rounded expected goals scored and conceded
game_data['xGround'] = game_data['xG'].apply(lambda x: round(x))
game_data['xGAround'] = game_data['xGA'].apply(lambda x: round(x))
game_data['xwin'] = game_data.apply(lambda x: 1 if x['xGround'] > x['xGAround'] else 0, axis=1)
game_data['xdraw'] = game_data.apply(lambda x: 1 if x['xGround'] == x['xGAround'] else 0, axis=1)
game_data['xloss'] = game_data.apply(lambda x: 1 if x['xGround'] < x['xGAround'] else 0, axis=1)


# In[ ]:


# Get the results for each team
x_team_results = []
for team in teams:
    # Get the data for that team
    team_data = game_data[game_data['teamId'] == team]
    
    wins = team_data['xwin'].sum()
    draws = team_data['xdraw'].sum()
    losses = team_data['xloss'].sum()
    scored = team_data['xGround'].sum()
    conceded = team_data['xGAround'].sum()
    games = wins + draws + losses
    points = (3 * wins) + draws
    goal_difference = scored - conceded
    
    x_team_results.append([team, games, wins, draws, losses, scored, conceded, goal_difference, points])

x_league_table = pd.DataFrame(x_team_results, columns=['Team', 'P', 'W', 'D', 'L', 'F', 'A', 'GD', 'Points'])
x_league_table.sort_values(by=['Points', 'GD', 'F'], ascending=False, inplace=True, ignore_index=True)
x_league_table.set_index(POSITIONS, inplace=True)
x_league_table.head(20)


# # What About The Expected Points Value?
# An expected points value is also given for each game. Although it will not give us a league table that is as detailed as the previously produced tables, we can still create a table. This is done in the following cell.

# In[ ]:


# Get the results for each team
xp_team_results = []
for team in teams:
    # Get the data for that team
    team_data = game_data[game_data['teamId'] == team]
    
    xp = team_data['xpts'].sum()
    
    xp_team_results.append([team, xp])

xp_league_table = pd.DataFrame(xp_team_results, columns=['Team', 'Points'])
xp_league_table.sort_values(by=['Points'], ascending=False, inplace=True, ignore_index=True)
xp_league_table.set_index(POSITIONS, inplace=True)
xp_league_table.head(20)


# # So How Are Teams Performing?
# The following cell compares current league position to the expected league tables' positions. Both the expected goals table and expected points table give similar positions for most teams, so we could choose to make comparisons to either. In this notebook we choose to make comparisons to the expected goals table (which was more precise).
# 
# We can determine whether a team is overperforming based upon whether their current position is above their expected position.  
# **Overperforming:** Liverpool, Leicester, Sheffield United, Tottenham, Arsenal, Crystal Palace, Newcastle United, West Ham  
# **Not Overperforming:** Man City, Chelsea, Man Utd, Wolves, Burnley, Everton, Southampton, Brighton, Watford, Bournemouth, Aston Villa, Norwich

# In[ ]:


team_positions = []
for team in teams:
    current_pos = league_table[league_table['Team'] == team].index[0]
    xg_pos = x_league_table[x_league_table['Team'] == team].index[0]
    xp_pos = xp_league_table[xp_league_table['Team'] == team].index[0]
    overperforming = 'Yes' if current_pos < xg_pos else 'No'
    team_positions.append([team, current_pos, xg_pos, xp_pos, overperforming])

position_table = pd.DataFrame(team_positions, columns=['Team', 'Position', 'xG Position', 'xPts Position', 'Overperforming'])
position_table.sort_values(by=['Position'], ascending=True, inplace=True, ignore_index=True)
position_table.head(20)


# # But How Is Recent Performance?
# The league is currently suspended, so when we talk about recent performance we are actually referring to the most recent games (despite them taking place over 2 months ago). We can look at how well teams were performing before the suspension by looking at how many points they were picking up. We will look at the points from the six most recent games for each team.

# In[ ]:


recent_form = []

for team in teams:
    # Get the data for that team
    team_data = game_data[game_data['teamId'] == team].tail(6)
    
    wins = team_data['wins'].sum()
    draws = team_data['draws'].sum()
    points = (3 * wins) + draws
    
    recent_form.append([team, points])

recent_form.sort(key=lambda x: x[1])

plt.figure(figsize = (8, 8))
plt.barh(range(20), [x[1] for x in recent_form])
plt.xlabel('Points')
plt.ylabel('Team')
plt.title('Points In Last 6 Games')
plt.yticks(range(20), [x[0] for x in recent_form])
plt.show()


# # So If They Carry On That Performance?
# With the (at least) two month suspension to the league, it is unlikely that teams are going to continue their form on to the end of the season. But at the end of this notebook we are going to make the assumption that they are. In this final cell we will calculate the final number of points that teams will have based upon the points per game in their last six played games.

# In[ ]:


team_points = []

# Add recent points per game to table
for team in teams:
    points_per_game = [x for x in recent_form if x[0] == team][0][1] / 6
    team_data = league_table[league_table['Team'] == team].iloc[0]
    games_to_play = 38 - team_data['P']
    new_points = int(team_data['Points'] + round(points_per_game * games_to_play))
    team_points.append([team, new_points])

predicted_table = pd.DataFrame(team_points, columns=['Team', 'Points'])
predicted_table.sort_values(by=['Points'], ascending=False, inplace=True, ignore_index=True)
predicted_table.set_index(POSITIONS, inplace=True)
predicted_table.head(20)


# Using this table we have the following predictions (points are in brackets).
# 
# **Champions:** Liverpool (104)  
# **Champions League:** Man City (74), Leicester (65), Chelsea (62)  
# **Europa League:** Sheffield United (61), Man United (61), Arsenal (60)  
# **Relegated:** Norwich (27), Aston Villa (32), West Ham or Watford (33)  
# 
# The main drawback to our prediction is that we cannot calculate goal difference, which means we ended up with two teams that could possibly get relegated - West Ham or Watford. They are currently on 27 points each and had the same recent form, and are separated by just 2 goal difference. Based on our predictions they would both finish with 33 points. It would take some very accurate predictions to guess who would get relegated in this situation.
