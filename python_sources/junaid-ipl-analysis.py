# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 


import pandas as pd
import numpy as np
import operator
import collections
from matplotlib import pyplot as pp
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv('/kaggle/input/ipldata/deliveries.csv')
data_2 = pd.read_csv('/kaggle/input/ipldata/matches.csv')

data_rcb = data.loc[data['batting_team'] == 'Royal Challengers Bangalore']

new_Data = data_rcb.loc[data_rcb['over'] > 15]

death_over_data = new_Data.groupby(['bowling_team']).sum()['total_runs']

scoring_batsman = new_Data.groupby(['batsman']).sum()['total_runs']

print(scoring_batsman.sort_values(ascending=False).head(10))
print("----------------------------------------------------------------------------------------------------------")
print(death_over_data.head(10).values)

data_competition = data_2.loc[(((data_2['team1'] == 'Royal Challengers Bangalore') | (data_2['team2'] == 'Royal '
                                                                                                         'Challengers '
                                                                                                         'Bangalore')) &
                               (data_2['winner'] == 'Royal Challengers Bangalore'))]

data_competition['count'] = 1

home_games = data_competition.groupby(['team1']).sum()['count']
away_games = data_competition.groupby(['team2']).sum()['count']

home_games = home_games.drop(['Royal Challengers Bangalore'])
away_games = away_games.drop(['Royal Challengers Bangalore'])

total_games = home_games + away_games

total_games.reindex(index=['team'])

print(home_games)
print("-----------------------------------------------------------------------------------------------------")
print(away_games)
print("-----------------------------------------------------------------------------------------------------")
print(total_games)

batsman_score = data.groupby(['batsman']).sum()['batsman_runs']
batsman_score_s = batsman_score.sort_values(ascending=False).head(20)

wickets = data.groupby(['bowler']).count()['player_dismissed']
top_takers = wickets.sort_values(ascending=False).head(20)
ball_count = 1
balls_strike_rate = data.groupby(['batsman']).count()

# strike_rate = balls_strike_rate.sort_values(ascending=False).head(50)
print("------------------------------------------------------------------------------------------------------")
strike_rate_dict = {}
for idx, row in balls_strike_rate.iterrows():
    if row['ball'] > 1500:
        # print(batsman_score[idx], row['ball'])
        strike_rate = (batsman_score[idx] / row['ball']) * 100
        strike_rate_dict[idx] = strike_rate
sorted_dict = collections.OrderedDict(strike_rate_dict)
for i in sorted_dict:
    print(i, sorted_dict[i])
    axx = pp.scatter(sorted_dict[i], i)

pp.show(axx)
as2 = top_takers.plot.barh(x='bowler', y='total_wickets', rot=0, figsize=(10, 10))
pp.show(as2)
as1 = batsman_score_s.plot.bar(x='batsman', y='batsman_runs')
pp.show(as1)
ax1 = scoring_batsman.sort_values(ascending=False).head(10).plot.bar(x='batsman', y='total_runs', rot=0)
pp.show(ax1)
ax = away_games.plot.barh(x='team1', y='count', rot=0, figsize=(4, 4))
pp.show(ax)
print("---------------------------------------------------------------------------------------------------------")
piechart1 = home_games.plot.pie(autopct='%.2f', figsize=(6, 6))
pp.show(piechart1)
print("----------------------------------------------------------------------------------------------------------")
ppp = total_games.plot.bar()
pp.show(ppp)
print("----------------------------------------------------------------------------------------------------------")
bat_bowl_data = data[['batsman', 'bowler', 'total_runs', 'player_dismissed']]
bowl_bat_wicket = bat_bowl_data.copy()
bat_bowl_data = bat_bowl_data.groupby(['batsman', 'bowler']).sum()
bowl_bat_wicket = bowl_bat_wicket.groupby(['batsman', 'bowler']).count()

bat_bowl_dict = {}
bowl_bat_dict={}
for idx, rows in bat_bowl_data.iterrows():
    if rows['total_runs'] > 100:
        bat_bowl_dict[idx] = rows['total_runs']
sorted_bat_bowl_dict = dict(sorted(bat_bowl_dict.items(), key=operator.itemgetter(1), reverse=True))

pp.bar(range(len(sorted_bat_bowl_dict)), list(sorted_bat_bowl_dict.values()), align='center', width=0.8)
pp.xticks(range(len(sorted_bat_bowl_dict)), list(sorted_bat_bowl_dict.keys()), rotation=90)
pp.show()
print("-----------------------------------------------------------------------------------------------------------")
for idx, rows in bowl_bat_wicket.iterrows():
    if rows['player_dismissed'] > 5:
        bowl_bat_dict[idx] = rows['player_dismissed']
sorted_bowl_bat_dict = dict(sorted(bowl_bat_dict.items(), key=operator.itemgetter(1), reverse=True))

pp.bar(range(len(sorted_bowl_bat_dict)), list(sorted_bowl_bat_dict.values()), align='center', width=1)
pp.xticks(range(len(sorted_bowl_bat_dict)), list(sorted_bowl_bat_dict.keys()), rotation=90)
pp.show()

fielder_data = data[['dismissal_kind', 'fielder', 'player_dismissed']]

fielder_data = fielder_data.groupby(['fielder', 'dismissal_kind']).count()

fielding = {}

for idx, rows in fielder_data.iterrows():
    if rows['player_dismissed'] > 40:
        fielding[idx] = rows['player_dismissed']
sorted_fielding = dict(sorted(fielding.items(), key=operator.itemgetter(1), reverse=True))
key_list = list(sorted_fielding.keys())
# key_list_11 = [list(key_list[0])[0] for item in key_list]
# print(key_list_11)
namelist = []
for i, item in key_list:
    namelist.append(i)
pp.bar(range(len(sorted_fielding)), list(sorted_fielding.values()), align='center', width=0.8)
pp.xticks(range(len(sorted_fielding)), namelist, rotation=90)
pp.show()
print(sorted_fielding)
venue_dict={}
venue = data_2[['winner', 'venue', 'city']]
venue_details = venue.groupby(['venue', 'winner']).count()
for idx,rows in venue_details.iterrows():
    if rows['city'] > 10:
        venue_dict[idx] = rows['city']

pp.bar(range(len(venue_dict)), list(venue_dict.values()), align='center', width=0.8)
pp.xticks(range(len(venue_dict)),venue_dict, rotation=90)
pp.show()

# Any results you write to the current directory are saved as output.