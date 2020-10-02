# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

#Shows All Dataframe Columns and Rows When Printed
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)

stats = pd.read_csv('../input/1997.csv')
year = 1997

# 2017 teams and abbreviations 
team_abbv = {'BOS': "Boston Celtics", 'CLE': "Cleveland Cavaliers", 'TOR': "Toronto Raptors", 'WAS': "Washington Wizards", 'ATL': "Atlanta Hawks", 'MIL': "Milwaukee Bucks", 'IND': "Indiana Pacers", 'CHI': "Chicago Bulls", 'MIA': "Miami Heat", 'DET': "Detroit Pistons", 'CHO': "Charlotte Hornets", "NYK": "New York Knicks", "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "BRK": "Brooklyn Nets", "GSW" : "Golden State Warriors", "SAS": "San Antonio Spurs", "HOU": "Houston Rockets", "LAC": "Los Angeles Clippers", "UTA": "Utah Jazz", "OKC": "Oklahoma City Thunder", "MEM": "Memphis Grizzlies", "POR": "Portland Trail Blazers", "DEN": "Denver Nuggets", "NOP": "New Orleans", "DAL": "Dallas Mavericks", "SAC": "Sacremento Kings", "MIN": "Minnesota Timberwolves", "LAL": "Los Angeles Lakers", "PHO": "Phoenix Suns"}


# CSV file contains team name as abbreviations. Adjust Team Abbv based on year, This accounts for team moves and changes. 
if "MIA" in team_abbv and year < 1989:
	del team_abbv["MIA"]
if "NOP" in team_abbv and year < 1989:
	del team_abbv["NOP"]
if "ORL" in team_abbv and year < 1990:
	del team_abbv["ORL"]
if "MIN" in team_abbv and year < 1990:
	del team_abbv["MIN"]
if "MEM" in team_abbv and year < 1996:
	del team_abbv["MEM"]
if "TOR" in team_abbv and year < 1996:
	del team_abbv["TOR"]
if "CHO" in team_abbv and year < 2004:
	del team_abbv["CHO"]

for key in list(team_abbv): 
	if key == "LAC" and year < 1985:
		del team_abbv["LAC"]
		team_abbv["SDC"] = "San Diego Clippers" 
	if key == "SAC" and year < 1986:
		del team_abbv["SAC"]
		team_abbv["KCK"] = "Kansas City Kings"
	if key == "WAS" and year < 1998:
		del team_abbv["WAS"]
		team_abbv["WSB"] = "Washington Bullets"
	if key == "MEM" and year < 2002:
		del team_abbv["MEM"]
		team_abbv["VAN"] = "Vancouver Grizzlies"
	if key == "NOP":
		if year < 2003:
			del team_abbv["NOP"]
			team_abbv["CHH"] = "Charlotte Hornets"
		if year == 2006 or year == 2007:
			del team_abbv["NOP"]
			team_abbv["NOK"] = "New Orleans/Oklahoma City Hornets"
		if year != 2006 and year != 2007 and year > 2002 and year < 2014:
			del team_abbv["NOP"]
			team_abbv["NOH"] = "New Orleans Hornets"
	if key == "OKC" and year < 2009:
		del team_abbv["OKC"]
		team_abbv["SEA"] = "Seattle Supersonics"
	if key == "BRK" and year < 2013:
		del team_abbv["BRK"]
		team_abbv["NJN"] = "New Jersey Nets"
		value = "New Jersey Nets"
	if key == "CHH" and year > 2002 and year < 2015:
		del team_abbv["CHH"]
		team_abbv["CHA"] = "Charlotte Bobcats"


# Total Games in an NBA season
games = 82
# Between 23 and 30 teams
total_teams = len(team_abbv) 
# Total games, Counts one game as two games since two teams play one game.
total_games_by_team = total_teams * games  
#Total Games in a Season
total_games = total_games_by_team / 2 

#Create Series for Totals and Rankings
def create_totals():
	totals = []
	for key, value in team_abbv.items():
		team_tot = {'team': value, 'mp': 0, 'fg' : 0, 'fga' : 0, 'fg3' : 0, 'fg3a' : 0, 'ft' : 0, 'fta': 0, 'orb' : 0, 'drb' : 0, 'trb' : 0, 'ast' : 0, 'stl' : 0, 'blk' : 0, 'tov' : 0, 'pf' : 0, 'pts' : 0, 'plus_minus' : 0}
		totals.append(team_tot)
	totals = pd.DataFrame(totals)
	totals.set_index('team', inplace=True)
	return totals

def average_plus_minus(n):
	return (n / games) / 5 # 5 == players on the court at one time

#Team Totals and Opposing Totals
def find_team_totals(stats):
	team_totals = create_totals()
	for i, player in stats.iterrows():
		for index, row in player.iteritems():
			if(index in team_totals):
				team_totals.loc[player['player_team']].loc[index] += row
	team_totals['plus_minus'] = team_totals['plus_minus'].map(average_plus_minus)
	return team_totals

# Example 
# print(find_team_totals(stats))

# All averages by player, separated by which team they played for
def averages(logs):
	del logs['game_location']
	del logs['opp_team']
	del logs['game_result']
	del logs['date_game']
	agg_obj = { 
		'gs': 'sum',
		'game_season': 'count',
   		'mp': 'mean', 
     	'fg': 'mean',  
      	'fga': 'mean', 
       	'fg3': 'mean', 
        'fg3a': 'mean', 
        'ft': 'mean', 
        'fta': 'mean', 
        'orb': 'mean', 
        'drb': 'mean', 
        'trb': 'mean', 
        'ast': 'mean', 
        'stl': 'mean', 
        'blk': 'mean', 
        'tov': 'mean', 
        'pf': 'mean', 
        'pts': 'mean', 
        'game_score': 'mean'
    }
	return logs.groupby(['player', 'player_team']).agg(agg_obj).reset_index()


def find_averages_by_player(player, logs):
	player_averages = averages(logs)
	return player_averages[player_averages['player'] == player]

# Example
# print(find_averages_by_player('Charles Smith', stats))

def find_averages_by_team(team, logs):
	player_averages = averages(logs)
	return player_averages.loc[player_averages['player_team'] == team]

# Example
# print(find_averages_by_team("San Antonio Spurs", stats))	

# This is used to also find W/L records for every team
def all_games(stats):
	total_stats_by_game = []
	test_games_list = []
	for index, row in stats.iterrows():
		p_team, o_team, date = row["player_team"], row["opp_team"], row["date_game"]
		home = [date, p_team, o_team]
		away = [date, o_team, p_team]
		gamelogs = stats[(stats["date_game"] == date)] 
		gamelogs = gamelogs[(gamelogs["player_team"] == p_team) | (gamelogs["player_team"] == o_team)]
		if home not in test_games_list or away not in test_games_list:
			test_games_list.append(home)
			test_games_list.append(away)
			#total_stats_by_game_totals.append(gamelogs) Might want combined gamelogs to have a separate array
			g = gamelogs.groupby('player_team', as_index=True).sum()
			total_stats_by_game.append(g)
	return total_stats_by_game

#Add up wins /****NOTE*****/ Celtics/Pacers game was cancelled in 2011
def team_wins(logs):
	total_stats_by_game = all_games(logs) # Get All Stats Separated By Game
	wins_losses = pd.Series()
	for game in total_stats_by_game:
		only_pts = game["pts"] # Get Points Of Both Teams and Index
		wins = (only_pts >= only_pts.max()).astype(int) # Return Boolean Value (1 or 0) winner true, loser false
		wins_losses = wins_losses.add(wins, fill_value=0) # Add to series
	return wins_losses


# print(team_wins(stats))	


def season_total_averages(total_games_by_team, logs):
    # Takes sum of all team_totals divided total games to season averages for the entire league
	return find_team_totals(logs).sum() / total_games_by_team


# season_total_averages(total_games_by_team, stats)


def rankSeries(column):
	rank = 0
	name = column.name
	if name == 'pf' or name == 'tov':
		column = abs(column - column.max()) # lowest number becomes highest because we need pf and tov rankings in reverse order
	column = column + abs(column.min()) + len(column) # prevents errors for numbers under length of columns
	while rank < len(column):
		max_index = column.argmax()
		column[max_index] = rank + 1
		rank += 1
	return column

def rank(df):
	return df.apply(rankSeries)

def stat_rankings(logs):
	team_totals = find_team_totals(logs)
	return rank(team_totals)


# print(stat_rankings(stats))
