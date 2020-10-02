#!/usr/bin/env python
# coding: utf-8

# ##IPL Data
# We are once again working on the IPL data. The focus this time is to look at the overall team performance and the performance of individual players

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter(action = "ignore", category = FutureWarning)
import re
from pandas import Series, DataFrame

matches = pd.read_csv('../input/matches.csv')
matches["type"] = "pre-qualifier"
for year in range(2008, 2017):
   final_match_index = matches[matches['season']==year][-1:].index.values[0]
   matches = matches.set_value(final_match_index, "type", "final")
   matches = matches.set_value(final_match_index-1, "type", "qualifier-2")
   matches = matches.set_value(final_match_index-2, "type", "eliminator")
   matches = matches.set_value(final_match_index-3, "type", "qualifier-1")

matches.groupby(["type"])["id"].count()
deliveries = pd.read_csv('../input/deliveries.csv')


# ##Depth of the batting line up
# ####Mean and standard deviation of the runs scored by each batsman per season
# On the face of it, some teams are supposed to have a much deeper batting line-up with tail-enders come in towards the end and hit big. Lets see if the assumption is correct

# In[ ]:


#Groupby season and team and get the runs scored by batsmen
agg = matches[['id','season', 'winner', 'toss_winner', 'toss_decision', 'team1']].merge(deliveries, left_on = 'id', right_on = 'match_id', how = 'left').drop('id', axis = 1)
batsman_grp = agg.groupby(["season","match_id", "inning", "batting_team", "batsman"])
batsmen = batsman_grp["batsman_runs"].sum().reset_index()
runs_scored = batsmen.groupby(['season','batting_team', 'batsman'])['batsman_runs'].agg(['sum','mean', 'std']).sort(ascending = False).reset_index()
runs_scored.head()


# ##Team scores in first inning vs second inning

# In[ ]:


agg_over = agg.groupby(['season','match_id', 'inning', 'over'])['total_runs'].sum().reset_index()
agg_score = agg_over.groupby(['season', 'inning'])['total_runs'].sum().reset_index()
agg_score = agg_score.drop(agg_score[agg_score.inning > 2 ].index)
agg_score.pivot(index='season', columns='inning', values='total_runs').plot(kind = 'bar')


# ####First Innings score is higher than second inning score but we need to separate the data in to the teams batting first vs batting second to come to any conclusion.

# ###Average runs scored by teams winning batting first vs teams winning batting second

# In[ ]:


agg_battingteam = agg.groupby(['season','match_id', 'inning', 'batting_team', 'bowling_team','winner'])['total_runs'].sum().reset_index()
winner = agg_battingteam[agg_battingteam['batting_team'] == agg_battingteam['winner']]#agg_batting = agg_battingteam.groupby(['season', 'inning', 'team1','winner'])['total_runs'].sum().reset_index()
winner_batting_first = winner[winner['inning'] == 1]
winner_batting_second = winner[winner['inning'] == 2]

winner_runs_batting_first = winner_batting_first.groupby(['season', 'winner'])['total_runs'].mean().reset_index().round()
winner_runs_batting_second = winner_batting_second.groupby(['season', 'winner'])['total_runs'].mean().reset_index().round()

winner_runs = winner_runs_batting_first.merge(winner_runs_batting_second, on = ['season','winner'])
winner_runs.columns = ['season', 'winner', 'Runs batting first', 'Runs batting second']
#winner_runs = winner_runs[(winner_runs['winner'] != 'Kochi Tuskers Kerala')| (winner_runs['winner'] != 'Pune Warriors')| (winner_runs['winner'] != 'Deccan Chargers') ]
winner_runs.head()


# In[ ]:


#Average Team score batting first vs second
fig, axes = plt.subplots(nrows=4, ncols=2)
fig.tight_layout()
ax = winner_runs[winner_runs['winner'] == 'Chennai Super Kings'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by CSK", ax=axes[0,0], figsize=(8, 15))
ax = winner_runs[winner_runs['winner'] == 'Delhi Daredevils'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by DD",ax=axes[0,1], figsize=(8, 15))
ax = winner_runs[winner_runs['winner'] == 'Kings XI Punjab'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by Kings XI",ax=axes[1,0], figsize=(8, 15))
ax = winner_runs[winner_runs['winner'] == 'Kolkata Knight Riders'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by KKR",ax=axes[1,1], figsize=(8, 15))
ax = winner_runs[winner_runs['winner'] == 'Mumbai Indians'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by MI",ax=axes[2,0], figsize=(8, 15))
ax = winner_runs[winner_runs['winner'] == 'Rajasthan Royals'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by RR",ax=axes[2,1], figsize=(8, 15))
ax = winner_runs[winner_runs['winner'] == 'Royal Challengers Bangalore'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by RCB",ax=axes[3,0], figsize=(8, 15))
ax = winner_runs[winner_runs['winner'] == 'Sunrisers Hyderabad'].drop('winner', axis = 1).plot(x = 'season', title = "Average runs by SH",ax=axes[3,1], figsize=(8, 15))
ax.set_xticklabels(winner_runs['season'].unique())
plt.show()


# ##Team scores of winning and losing teams as the inning progresses
# We will break the 20 over innings in to four parts and see the average scores to see how typically the match progresses. Lets keep the first and the second innings separate as the second inning progression is highly dependent on the score in the first

# In[ ]:


agg_winner = agg[agg['winner'] == agg['batting_team']]
agg_loser = agg[agg['winner'] != agg['batting_team']]
agg_winner = agg_winner.groupby(['season','match_id', 'inning', 'over'])['total_runs'].sum().reset_index()
agg_loser = agg_loser.groupby(['season','match_id', 'inning', 'over'])['total_runs'].sum().reset_index()

agg_1_winner = agg_winner[agg_winner['over'] < 6]
agg_1_winner = agg_1_winner.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()
agg_1_loser = agg_loser[agg_loser['over'] < 6]
agg_1_loser = agg_1_loser.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()


agg_2_winner = agg_winner[(agg_winner['over'] > 5) & (agg_winner['over'] < 11)]
agg_2_winner = agg_2_winner.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()
agg_2_loser = agg_loser[(agg_loser['over'] > 5) & (agg_loser['over'] < 11)]
agg_2_loser = agg_2_loser.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()


agg_3_winner = agg_winner[(agg_winner['over'] > 10) & (agg_winner['over'] < 16)]
agg_3_winner = agg_3_winner.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()
agg_3_loser = agg_loser[(agg_loser['over'] > 10) & (agg_loser['over'] < 16)]
agg_3_loser = agg_3_loser.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()

agg_4_winner = agg_winner[(agg_winner['over'] > 15) & (agg_winner['over'] < 21)]
agg_4_winner = agg_4_winner.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()
agg_4_loser = agg_loser[(agg_loser['over'] > 15) & (agg_loser['over'] < 21)]
agg_4_loser = agg_4_loser.groupby(['season','match_id', 'inning'])['total_runs'].sum().reset_index()

score_progress_winner = agg_1_winner.merge(agg_2_winner,on=['season','match_id', 'inning']).merge(agg_3_winner,on=['season','match_id', 'inning']).merge(agg_4_winner,on=['season','match_id', 'inning'])
score_progress_loser = agg_1_loser.merge(agg_2_loser,on=['season','match_id', 'inning']).merge(agg_3_loser,on=['season','match_id', 'inning']).merge(agg_4_loser,on=['season','match_id', 'inning'])

score_progress_winner.columns = ['season','match_id', 'inning', 'score_5overs', 'score_5_10_overs', 'score_10_15_overs', 'score_15_20_overs']
score_progress_winner.drop('match_id', axis = 1, inplace = True)
score_progress_loser.columns = ['season','match_id', 'inning', 'score_5overs', 'score_5_10_overs', 'score_10_15_overs', 'score_15_20_overs']
score_progress_loser.drop('match_id', axis = 1, inplace = True)


# In[ ]:


score_progress_winner_mean = score_progress_winner.groupby(['season', 'inning']).mean().reset_index()
score_progress_loser_mean = score_progress_loser.groupby(['season', 'inning']).mean().reset_index()

score_progress_winner_inning1 = score_progress_winner_mean[score_progress_winner_mean["inning"] == 1].drop('inning', axis = 1)
score_progress_winner_inning2 = score_progress_winner_mean[score_progress_winner_mean["inning"] == 2].drop('inning', axis = 1)

score_progress_loser_inning1 = score_progress_loser_mean[score_progress_loser_mean["inning"] == 1].drop('inning', axis = 1)
score_progress_loser_inning2 = score_progress_loser_mean[score_progress_loser_mean["inning"] == 2].drop('inning', axis = 1)

ax = score_progress_winner_inning1.plot(x = 'season', title = 'Average first inning score of winning and losing (--) teams', color = ['b', 'k', 'r', 'g']); #Total first inning score of winning teams by season
score_progress_loser_inning1.plot(x = 'season', style= ['b--', 'k--', 'r--', 'g--'], ax = ax); #Total first inning score of winning teams by season
ax.set_xticklabels(score_progress_winner_inning1['season'])
plt.show()


# ####The teams batting first score close to 1.5 times runs in the last 5 overs compared to first three 5 over spells.

# In[ ]:


ax = score_progress_winner_inning2.plot(x = 'season', title = 'Average second inning score of winning and losing (--) teams', color = ['b', 'k', 'r', 'g']) #Total first inning score of winning teams by season
score_progress_loser_inning2.plot(x = 'season', style= ['b--', 'k--', 'r--', 'g--'], ax = ax) #Total first inning score of winning teams by season
ax.set_xticklabels(score_progress_winner_inning1['season'])
plt.show()


# When a team is chasing, they try to score even runs through-out the match and there is only a slight increase in the runs scored in the last 5 overs compared to the first three spells. 

# In[ ]:




