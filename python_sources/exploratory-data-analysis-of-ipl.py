#!/usr/bin/env python
# coding: utf-8

# <h1>This notebook is written specifically to perform exploratory data analysis of the IPL dataset. The intention behind writing this notebook is to utilise maximum number of basic to intermediate level plots for the sake of reference. Please upvote and share your feedback if this benefits you in anyway.</h1>

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

matches_df = pd.read_csv('../input/matches.csv');
deliveries_df = pd.read_csv('../input/deliveries.csv');

print('Matches Data Dimensions: ', matches_df.shape)
print (matches_df.columns)
print('Deliveries Data Dimensions: ', deliveries_df.shape)
print (deliveries_df.columns)


# In[ ]:


print ("==================== NULLS in DELIVERIES ====================")
print (deliveries_df.isnull().sum())
print ("==================== NULLS in MATCHES ====================")
print (matches_df.isnull().sum())


# In[ ]:


matches_drop_cols = ['date','umpire3']
deliveries_drop_cols = ['player_dismissed']
matches_df = matches_df.drop(matches_drop_cols, axis=1)
deliveries_df = deliveries_df.drop(deliveries_drop_cols, axis=1)


# In[ ]:


combined_df = pd.merge(matches_df, deliveries_df, left_on='id', right_on='match_id')
print (combined_df.columns)


# In[ ]:


combined_df['team1'].replace({'Rising Pune Supergiant' : 'Rising Pune Supergiants'}, inplace=True)
combined_df['team2'].replace({'Rising Pune Supergiant' : 'Rising Pune Supergiants'}, inplace=True)
combined_df['winner'].replace({'Rising Pune Supergiant' : 'Rising Pune Supergiants'}, inplace=True)
combined_df['toss_winner'].replace({'Rising Pune Supergiant' : 'Rising Pune Supergiants'}, inplace=True)
combined_df['batting_team'].replace({'Rising Pune Supergiant' : 'Rising Pune Supergiants'}, inplace=True)
combined_df['bowling_team'].replace({'Rising Pune Supergiant' : 'Rising Pune Supergiants'}, inplace=True)


combined_df['venue'].replace({'Punjab Cricket Association IS Bindra Stadium, Mohali' : 'Punjab Cricket Association Stadium, Mohali'}, inplace=True)

combined_df['is_winner'] = np.where(combined_df['batting_team'] == combined_df['winner'], 'yes', 'no')

# Delete matches data which are incomplete
ids_of_matches_with_no_decision = matches_df[matches_df['winner'].isnull()]['id'].unique()
combined_df = combined_df[~combined_df['match_id'].isin(ids_of_matches_with_no_decision)]


# <h1>Let's start by plotting the number of matches won in each innings. Check whether batting first or fielding first helps in winning the match or not</h1>

# In[ ]:


match_won_in_each_innings = combined_df.loc[: , ['match_id', 'inning', 'is_winner']].drop_duplicates()
print (match_won_in_each_innings.groupby(['inning', 'is_winner'])['match_id'].count().reset_index().rename(columns={'inning' : 'Innings' , 'is_winner' : 'Winner', 'match_id' : 'Count of Matches'}))

plt.figure(figsize=(20, 8))

ax0 = plt.subplot(121)
g = sns.countplot(x='inning', hue='is_winner', data=match_won_in_each_innings, ax=ax0)
g.set_xlabel('Innings')
g.set_ylabel('Count of Matches')
g.set_title('Count of Matches Won/Lost in each Innings')
plt.legend(['Matches Won', 'Matches Lost'])


# <h1>Result:</h1>
# <p>The difference is there! Most matches are won when batting in second innings. The plot shows that chasing the score actually helps.</p>

# <h1>What is the average score per over in powerplay/non-powerplay overs? Does it help in winning the match?</h1>

# In[ ]:


teams_score_per_power_play = combined_df.loc[:, ['match_id', 'over', 'total_runs', 'is_winner', 'season']]
teams_score_per_power_play['power_play'] = np.where(teams_score_per_power_play['over'] <= 6, 'yes', 'no')
teams_score_per_power_play = teams_score_per_power_play.groupby(['season', 'match_id', 'is_winner', 'power_play'])['total_runs'].sum().reset_index(name ='score')
teams_score_per_power_play["avg_score_per_over"] = np.where(teams_score_per_power_play["power_play"] == 'yes', teams_score_per_power_play["score"].div(6, axis=0), teams_score_per_power_play["score"].div(14, axis=0))


# In[ ]:


plt.figure(figsize=(20 , 8))

ax0 = plt.subplot(221)
sns.boxplot(x="season", y="avg_score_per_over", hue="power_play", data=teams_score_per_power_play, ax=ax0, palette='winter')
plt.xlabel('Season')
plt.ylabel('Per Over Average Score')
plt.title('Average Score Per Over in Each Season By Powerplay')

ax1 = plt.subplot(222)
sns.boxplot(x="season", y="avg_score_per_over", hue="is_winner", data=teams_score_per_power_play, ax=ax1, palette='summer')
plt.xlabel('Season')
plt.ylabel('Over Average Score')
plt.title('Average Score Per Over in Each Season By Winner')

ax2 = plt.subplot(212)
# sns.boxplot(x="power_play", y="avg_score_per_over", hue="is_winner", data=teams_score_per_power_play, ax=ax2, palette='Set1')
sns.distplot(teams_score_per_power_play[teams_score_per_power_play['is_winner'] == 'yes']['avg_score_per_over'], bins=50, ax=ax2, label='Winner')
sns.distplot(teams_score_per_power_play[teams_score_per_power_play['is_winner'] == 'no']['avg_score_per_over'], bins=50, ax=ax2, label='Not Winner')
plt.legend()
plt.xlabel('Power Play')
plt.ylabel('Over Average Score')
plt.title('Average Score Per Over By Winner')

plt.subplots_adjust(hspace=0.5)


# <h1>Result:</h1>
# <p>1. The boxplot which displays average score per over by powerplay is surprising. In every season, teams have performed lesser average score per over in power play than non power play overs. Though the difference is not much.</p>
# <p>2. The median average score per over of winning teams for every season is close to 8</p>
# <p>2. The median average score per over of losing teams for every season is 7.5 or lesser</p>

# <h1>Were the teams able to make the right toss decision? Did toss winning team also won the match?</h1>

# In[ ]:


winner_team_score_each_season = combined_df.loc[:, ['match_id', 'toss_winner', 'toss_decision', 'winner']].drop_duplicates()
winner_team_score_each_season['winning_team_won_toss'] = np.where(winner_team_score_each_season['toss_winner'] == winner_team_score_each_season['winner'], 'yes', 'no')
winner_team_score_each_season = winner_team_score_each_season.groupby(['winning_team_won_toss', 'toss_decision'])['match_id'].count().reset_index().rename(columns={"match_id" : 'Count of Matches', 'toss_decision' : 'Toss Decision'})

plt.figure(figsize=(20, 8))

ax0 = plt.subplot(121)
sns.barplot(x="winning_team_won_toss", y="Count of Matches", hue="Toss Decision", data=winner_team_score_each_season, ax=ax0);
ax0.set_xlabel('Toss Winner')
ax0.set_ylabel('Matches Won (Count)')
ax0.set_title('Did match winning team also won the toss?')

ax1 = plt.subplot(122)
sns.barplot(x = 'winning_team_won_toss', y='Count of Matches', data=winner_team_score_each_season.groupby(['winning_team_won_toss'])['Count of Matches'].sum().reset_index(), ax=ax1)
ax1.set_xlabel('Toss Winner')
ax1.set_ylabel('Matches Won (Count)')
ax1.set_title('Did match winning team also won the toss?')

winner_team_score_each_season


# <h1>Result:</h1>
# <p>It is clear that the most number of matches are won by teams when they decided to field first after winning the toss. However, the teams which lost the toss but won the match, it didn't really whether they played first or chased. </p>

# <h1>How did teams perform when they opted either to bat first or field first? What was the average score per over of each team according to toss decision and the match result?</h1>

# In[ ]:


# Lets check average per over score of winning teams
total_scores_and_balls_winner = combined_df[combined_df['batting_team'] == combined_df['winner']].groupby(['batting_team', 'toss_decision'])['total_runs'].agg(['count', 'sum']).reset_index().rename(columns={'count' : 'balls_count', 'sum' : 'match_score'})
total_scores_and_balls_winner['overs_count'] = np.divide(total_scores_and_balls_winner['balls_count'], 6.0)
total_scores_and_balls_winner['average_score_per_over'] = np.divide(total_scores_and_balls_winner['match_score'], total_scores_and_balls_winner['overs_count'])
total_scores_and_balls_winner[['batting_team', 'toss_decision', 'average_score_per_over']]

plt.figure(figsize=(20,8))

ax0 = plt.subplot(121)
g = sns.pointplot(x='batting_team', y='average_score_per_over', hue='toss_decision', data=total_scores_and_balls_winner, ax=ax0)
plt.ylim(6, 10)
plt.xticks(rotation=90)
plt.title('Winning Teams Average Score Per Over According to Toss Decision')
plt.ylabel('Average Score Per Over')
plt.xlabel('Batting Team')

total_scores_and_balls_winner = combined_df[combined_df['batting_team'] != combined_df['winner']].groupby(['batting_team', 'toss_decision'])['total_runs'].agg(['count', 'sum']).reset_index().rename(columns={'count' : 'balls_count', 'sum' : 'match_score'})
total_scores_and_balls_winner['overs_count'] = np.divide(total_scores_and_balls_winner['balls_count'], 6.0)
total_scores_and_balls_winner['average_score_per_over'] = np.divide(total_scores_and_balls_winner['match_score'], total_scores_and_balls_winner['overs_count'])
total_scores_and_balls_winner[['batting_team', 'toss_decision', 'average_score_per_over']]

ax1 = plt.subplot(122)
g = sns.pointplot(x='batting_team', y='average_score_per_over', hue='toss_decision', data=total_scores_and_balls_winner, ax=ax1, sharey=ax0)
plt.ylim(6, 10)
plt.xticks(rotation=90)
plt.title('Losing Teams Average Score Per Over According to Toss Decision')
plt.ylabel('Average Score Per Over')
plt.xlabel('Batting Team')


# <h1>Result:</h1>
# <p>1. Visualization says that most of the teams performed/scored better when they chose to field first no matter they won the match or not.</p>
# <p>2. The average score per over when teams won matches, irrespective of the toss decision, oscilates around 8 or higher with a few exceptions.</p>
# <p>3. Kochi Tuskers Kerala opted for field first in all the matches they won.</p>
# <p>4. For Gujarat Lions, toss decision does not really matter since the average score per over is more or less the same when they won the match.</p>

# <h1>It will be interesting to see how boundaries played role in winning the match. Since the game is of limited overs, let's see the boundaries stats in each season by each team</h1>

# In[ ]:


total_4s6s_per_team_per_match = combined_df.loc[:, ['season', 'total_runs', 'match_id', 'is_winner']]
boundaries_per_season = total_4s6s_per_team_per_match.groupby(['season', 'match_id', 'is_winner'])['total_runs'].agg(lambda runs: ((runs == 4) | (runs == 6)).sum()).reset_index()
boundaries = boundaries_per_season.rename(columns={'total_runs':'boundaries'})

plt.figure(figsize=(20,8))
ax0 = plt.subplot(211)
g = sns.boxplot(x='season', y= 'boundaries', hue='is_winner' ,data=boundaries, palette='Set1', ax=ax0)
g.set_xlabel('Season')
g.set_ylabel('Boundaries Hit')
g.set_title('Boundaries Hit Per Season')

total_4s6s_per_team_per_match = combined_df.loc[:, ['match_id', 'batting_team', 'total_runs', 'is_winner']]
boundaries_per_season = total_4s6s_per_team_per_match.groupby(['match_id', 'batting_team' , 'is_winner'])['total_runs'].agg(lambda runs: ((runs == 4) | (runs == 6)).sum()).reset_index()
boundaries = boundaries_per_season.rename(columns={'total_runs':'boundaries'})

ax1 = plt.subplot(212)
g = sns.boxplot(x='batting_team', y= 'boundaries', hue='is_winner' ,data=boundaries, palette='Set2', ax=ax1)
g.set_xlabel('Batting Team')
g.set_ylabel('Boundaries Hit')
g.set_title('Boundaries Hit By Teams')
plt.xticks(rotation=90)

plt.subplots_adjust(hspace=0.5)


# <h1>Display scores of each team in every season and see which teams scored the higest and lowest scores till date</h1>

# In[ ]:


teams_score_per_season = combined_df.loc[: , ['season', 'match_id', 'batting_team', 'total_runs']]
teams_score_per_season = teams_score_per_season.groupby(['season', 'match_id', 'batting_team'])['total_runs'].agg(np.sum).reset_index()
#teams_score_per_season

plt.figure(figsize=(20, 8))
ax0 = plt.subplot(111)
g = sns.swarmplot(x='season', y='total_runs', hue='batting_team', data=teams_score_per_season, ax=ax0, palette='Set1')
g.set_title('Teams Score in Each Season')
g.set_xlabel('Season')
g.set_ylabel('Score')
ax0.legend(loc='center left', bbox_to_anchor=(1, 0.5))


# <h1>Let's find out the top batsmen in all the seasons and see how they relate to each other. Compare the strike rates of top scorers</h1>

# In[ ]:


batsman_score = combined_df.loc[:, ['batsman', 'total_runs']]
batsman_score = batsman_score.groupby(['batsman'])['total_runs'].agg(['sum', 'count']).reset_index().rename(columns={'sum' : 'total_runs' , 'count' : 'balls_count'})
batsman_score['strike_rate'] = np.divide(batsman_score['total_runs'], batsman_score['balls_count']) * 100

plt.figure(figsize=(20, 8))
g = sns.relplot(x="batsman", y="strike_rate", size="total_runs",sizes=(40, 400), palette="muted", height=10, data=batsman_score.nlargest(30, ['total_runs']).sort_values(['batsman']))
plt.xlabel('Batsman')
plt.ylabel('Strike Rate')
plt.xticks(rotation=90)

batsman_score.nlargest(30, ['total_runs'])


# <h1>Result:</h1>
# <p>1. SK Raina is the highest scorer</p>
# <p>2. V Sehwag has the highest strike rate</p>

# <h1>How did teams perform when chasing or defending the score?<br/>What is the average 'win_by_runs' for each team?<br/>What is the average 'win_by_wickets' for each team?</h1>

# In[ ]:


win_by = combined_df.loc[:, ['id', 'toss_winner','toss_decision', 'win_by_wickets', 'win_by_runs']]
win_by_wickets = win_by[(win_by['toss_decision'] == 'field') & (win_by['win_by_wickets'] != 0)].loc[:, ['id', 'toss_winner', 'win_by_wickets']].drop_duplicates()
plt.figure(figsize=(20, 8))

ax0 = plt.subplot(121)
g = sns.barplot(data=win_by_wickets, x='toss_winner', y='win_by_wickets', ax=ax0)
g.set_xlabel('Team')
g.set_ylabel('Wickets')
g.set_title('Win by Wickets (Field First)')
plt.xticks(rotation=90)

win_by_runs = win_by[(win_by['toss_decision'] == 'bat') & (win_by['win_by_runs'] != 0)].loc[:, ['id', 'toss_winner', 'win_by_runs']].drop_duplicates()
ax0 = plt.subplot(122)
g = sns.barplot(data=win_by_runs, x='toss_winner', y='win_by_runs', ax=ax0)
g.set_xlabel('Team')
g.set_ylabel('Runs')
g.set_title('Win by Runs (Bat First)')
plt.xticks(rotation=90)


# <h1>Result:</h1>
# <p>1. Sunrisers Hyderabad have the least win by runs average. Hence, they found it relatively hard to defend the score.</p>
# <p>2. Kochi Tuskers Kerala and Sunrisers Hyderabad have the highest average of win by wickets. They comfortable chased the scores in the matches they won saving a lot of wickets which means they have a strong top order.</p>
# <p>3. Every team was able to chase score by saving at least 5 wickets on average.</p>

# <h1>Most Number of Man of the Matches Award</h1>

# In[ ]:


top_man_of_matches = combined_df.loc[:, ['match_id', 'player_of_match']].drop_duplicates()
top_man_of_matches = top_man_of_matches.groupby(['player_of_match'])['match_id'].count().nlargest(5).reset_index().rename(columns={'match_id' : 'total_matches'})
top_man_of_matches


# <h1>Result:</h1>
# <p>Christopher Henry Gayle is the man!</p>
# 

# <h1>How did the different pitches behave? What was the average score for each stadium?</h1>

# In[ ]:


score_per_venue = combined_df.loc[:, ['match_id', 'venue', 'inning', 'total_runs']]
average_score_per_venue = score_per_venue.groupby(['match_id', 'venue', 'inning']).agg({'total_runs' : 'sum'}).reset_index()
average_score_per_venue = average_score_per_venue.groupby(['venue', 'inning'])['total_runs'].mean().reset_index()
average_score_per_venue = average_score_per_venue[(average_score_per_venue['inning'] == 1) | (average_score_per_venue['inning'] == 2)]

plt.figure(figsize=(20, 8))

ax2 = plt.subplot(211)
g = sns.pointplot(x='venue', y='total_runs', hue='inning', data=average_score_per_venue, ax=ax2, palette='Set1')
g.set_ylabel('Score')
g.set_title('Average Score in Each Stadium by Innings')
plt.xticks(rotation=90)


# <h1>Result:</h1>
# <p>1. Looks like Brabourne Stadium is favorite for batsmen. It has the highest average score for both innings</p>
# <p>2. Newlands is the most difficult pitch to bat on in second innings. Well, a sensible toss decision has to be made!</p>
# <p>3. There are few pitches which do not change and hence it does not matter whether the teams bat in first innings or second. The average score is almost the same.</p>
# 

# <h1>Does the toss decision vary according to the stadium/pitch? Does it help?</h1>

# In[ ]:


toss_vs_venue = combined_df.loc[:, ['match_id', 'venue', 'toss_decision', 'toss_winner', 'winner']].drop_duplicates()
toss_vs_venue['is_winner'] = np.where(toss_vs_venue['toss_winner'] == toss_vs_venue['winner'], 'yes', 'no')
toss_vs_venue.drop(['toss_winner', 'winner'], axis=1, inplace=True)
toss_vs_venue = toss_vs_venue.groupby(['venue', 'toss_decision'])['is_winner'].count().reset_index().rename(columns={'is_winner' : 'matches_won'})
toss_vs_venue = pd.DataFrame(toss_vs_venue.pivot_table('matches_won', ['venue'], 'toss_decision').reset_index().to_records())
toss_vs_venue = toss_vs_venue.fillna(0)

plt.figure(figsize=(20, 8))
p1 = plt.bar(toss_vs_venue['index'], toss_vs_venue['bat'], width=0.35)
p2 = plt.bar(toss_vs_venue['index'], toss_vs_venue['field'], width=0.35, bottom=toss_vs_venue['bat'])
plt.xticks(toss_vs_venue['index'], toss_vs_venue['venue'], rotation=90)
plt.legend((p1[0], p2[0]), ('Bat', 'Field'))
plt.xlabel('Matches Won')
plt.title('Matches Won in Each Stadium According to Toss Decision')

toss_vs_venue


# <h1>Result:</h1>
# <p>Clearly some stadiums favor fielding and some batting first!</p>
# <p>1. Most number of matches have been played in M Chinnaswammy Stadium which has clearly favored the toss decision of field</p>

# <h1>How did boundaries contribute in wining the match? What percentage of boundaries were hit by winning and losing teams?</h1>

# In[ ]:


def findPercentScoreOfBoundaries(ser):
    boundaries_score = 0
    total_score = 0
    for runs in ser:
        if ((runs == 4) | (runs == 6)):
            boundaries_score += runs
        total_score += runs
    return (boundaries_score / total_score) * 100
 
winning_team_percent_boundaries = total_4s6s_per_team_per_match.groupby(['match_id', 'is_winner'])['total_runs'].apply(findPercentScoreOfBoundaries).reset_index().rename(columns={'total_runs' : 'percentage_of_boundaries'})

plt.figure(figsize=(20, 8))

ax0 = plt.subplot(121)
g = sns.boxenplot(x='is_winner', y='percentage_of_boundaries', data=winning_team_percent_boundaries, ax=ax0)
g.set_title('Percentage of Boundaries')
g.set_ylabel('Percentage')
g.set_xlabel('Is Winner')
g.set_ylim(0, 100)

ax1 = plt.subplot(122)
g = sns.distplot(winning_team_percent_boundaries[winning_team_percent_boundaries['is_winner'] == 'yes']['percentage_of_boundaries'], hist=False, ax=ax1, color='darkorange')
g = sns.distplot(winning_team_percent_boundaries[winning_team_percent_boundaries['is_winner'] == 'no']['percentage_of_boundaries'], hist=False, ax=ax1, color='blue')
g.set_xlabel('Percentage of Boundaries')
g.set_title('Winning Team (Orange) vs Losing Team (Blue) Percentage of Boundaries per Match')


# <h1>Result:</h1>
# <p>1. Winning teams' boundary percentage is somewhere close to 60 whereas for losing team, it is somewhere close to 52. It does makes sense.</p>

# <h1>Distribution of scores in each over of winning and losing teams</h1>

# In[ ]:


score_per_over_distribution = combined_df.loc[:, ['match_id', 'over', 'total_runs', 'batting_team', 'winner']]
score_per_over_distribution['is_winner'] = np.where(score_per_over_distribution['batting_team'] == score_per_over_distribution['winner'], 'yes', 'no')
score_per_over_distribution.drop(['batting_team', 'winner'], axis=1, inplace=True)
score_per_over_distribution = score_per_over_distribution.groupby(['match_id', 'over', 'is_winner'])['total_runs'].agg(np.sum).reset_index()
plt.figure(figsize=(20, 16))

over = 1
for i in range(5):
    for j in range(4):
        ax = plt.subplot2grid((5,4), (i,j))
        p1 = sns.distplot(score_per_over_distribution[(score_per_over_distribution['over'] == over) & (score_per_over_distribution['is_winner'] == 'yes')]['total_runs'], hist=False, ax=ax)
        p2 = sns.distplot(score_per_over_distribution[(score_per_over_distribution['over'] == over) & (score_per_over_distribution['is_winner'] == 'no')]['total_runs'], hist=False, ax=ax)
        p1.set_title("Over " + str(over))
        p1.set_ylim([0, 0.12])
        p2.set_ylim([0, 0.12])
        over = over + 1

plt.subplots_adjust(hspace=0.8)
print('Winning Team (Blue) vs Losing Team (Orange)')


# In[ ]:


plt.figure(figsize=(20,8))
p1 = sns.kdeplot(score_per_over_distribution[score_per_over_distribution['is_winner'] == 'yes']['total_runs'], shade=True, legend=True)
p1.set_ylim([0, 0.11])
sns.kdeplot(score_per_over_distribution[score_per_over_distribution['is_winner'] == 'no']['total_runs'], shade=True, legend=True)
p2.set_ylim([0, 0.11])
plt.legend(['Winner', 'Not Winner'])
plt.xlabel('Score')
plt.title('Distribution of score per over')


# <h1>In short format matches, every single run is important. Hence, teams try their best to avoid any sort of extra runs (all extras). But does it really help in winning*</h1>

# In[ ]:


extra_runs = combined_df.loc[:, ['match_id', 'batting_team', 'winner', 'extra_runs']]
extra_runs['is_winner'] = np.where(extra_runs['batting_team'] == extra_runs['winner'], 'yes', 'no')
extra_runs.drop(['batting_team', 'winner'], axis=1, inplace=True)
extra_runs = extra_runs.groupby(['match_id', 'is_winner'])['extra_runs'].sum().reset_index()
#extra_runs
plt.figure(figsize=(20, 8))
ax0 = plt.subplot(121)
sns.boxplot(x='is_winner', y='extra_runs', data=extra_runs, ax=ax0)


# <h1>Result:</h1>
# <p>Not Really!</p>

# <h1>What is the breakdown of the runs of top 5 batsmen? Did they solely rely on hitting the ball out of the park?</h1>

# In[ ]:


plt.figure(figsize=(20, 16))
plt.title('Breakdown of Top Batsmen Score')

top_scorer = batsman_score.nlargest(5, ['total_runs'])['batsman'].reset_index()
score_break_down = combined_df[combined_df['batsman'].isin(top_scorer['batsman'])].loc[:, ['id', 'batsman', 'total_runs']]
score_break_down = score_break_down.groupby(['batsman', 'total_runs'])['id'].count().reset_index().rename(columns={'id' : 'count'})

axes = [231, 232, 233, 234, 235]
index = 0
for batsman in top_scorer['batsman'].unique():
    subplt = plt.subplot(axes[index])
    subplt.set_title(batsman)
    index = index + 1
    subplt.pie(score_break_down[score_break_down['batsman'] == batsman]['count'], labels=score_break_down[score_break_down['batsman'] == batsman]['total_runs'], autopct='%1.1f%%', shadow=True)


# <h1>Result:</h1>
# <p>The cricket gurus were right afterall! A good batsman needs to rely on singles to score bulk of runs and this is the special thing about top batsmen.</p>

# In[ ]:


pd.crosstab(combined_df.loc[:, ['match_id', 'winner']].drop_duplicates()['winner'], combined_df['season'], margins=True).style.background_gradient(cmap='summer_r')


# In[ ]:




