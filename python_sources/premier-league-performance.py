import numpy as np
import pandas as pd
import sqlite3

conn = sqlite3.connect('../input/database.sqlite')
c = conn.cursor()

def gather( df, key, value, cols ):
    id_vars = [ col for col in df.columns if col not in cols ]
    id_values = cols
    var_name = key
    value_name = value
    return pandas.melt( df, id_vars, id_values, var_name, value_name )



Time = '2014-07-01'

## select all the matches in the last two seasons of Premeier league
sqlcmd = """SELECT id, country_id, league_id, season, stage, date, match_api_id, home_team_api_id, away_team_api_id, home_team_goal FROM Match WHERE league_id = 1729 AND date >= '2014-07-01' """
match_data = pd.read_sql_query(sqlcmd, conn)


## select teams' and players' id and names
sqlcmd1 = """SELECT team_api_id, team_long_name FROM Team"""
teams = pd.read_sql_query(sqlcmd1,conn)
sqlcmd2 = """SELECT player_api_id, player_name, birthday, height, weight FROM Player"""
players = pd.read_sql_query(sqlcmd2,conn)


## select players' ratings in the last two seasons
sqlcmd3 = """SELECT player_api_id, date, overall_rating, potential FROM Player_Attributes WHERE date >= '2014-07-01' """
player_attr = pd.read_sql_query(sqlcmd3,conn)

sqlcmd4 = """SELECT id, date, home_team_api_id, home_player_1, home_player_2, home_player_3, home_player_4, home_player_5, home_player_6, home_player_7, home_player_8, home_player_9, home_player_10, home_player_11 FROM Match WHERE league_id = 1729 AND date >= '2014-07-01'"""
home_player = pd.read_sql_query(sqlcmd4,conn)
###home_player = gather(home_player, player_num, player_id, home_player_1:home_player_11)
home_player = pd.melt(home_player, id_vars=['id', 'date', 'home_team_api_id'], var_name='player_num', value_name='player_id')
home_player.rename(columns={'home_team_api_id': 'team_api_id'})

sqlcmd5 = """SELECT id, date, away_team_api_id, away_player_1, away_player_2, away_player_3, away_player_4, away_player_5, away_player_6, away_player_7, away_player_8, away_player_9, away_player_10, away_player_11 FROM Match WHERE league_id = 1729 AND date >= '2014-07-01'"""
away_player = pd.read_sql_query(sqlcmd5, conn)
###away_player = gather(away_player, player_num, player_id, away_player_1:away_player_11)
away_player = pd.melt(away_player, id_vars=['id', 'date', 'away_team_api_id'], var_name='player_num', value_name='player_id')
away_player.rename(columns={'away_team_api_id': 'team_api_id'})

player_attr.groupby('player_api_id').agg({'overall_rating': mean, 'potential': mean})
player_attr.rename(columns = {'overall_rating':'avgrating', 'potential':'avgPotential'}, inplace = True)

players_performance = home_player.append(away_player) 
player_performance.groupby('team_api_id', 'player_id') 

s1 = pd.merge(player_performance,teams, how = 'left', on = ['team_api_id']).merge(players, how = 'left',on = ['player_api_id']).merge(players_rating, on = ['player_api_id']).merge(player_attr, on = ['player_api_id'])
print(s1.head(2))




