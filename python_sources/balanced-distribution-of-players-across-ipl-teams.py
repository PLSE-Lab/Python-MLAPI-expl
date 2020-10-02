#!/usr/bin/env python
# coding: utf-8

# Thanks to Manas for providing this great data set.
# 
# One of the drawbacks of the last couple of IPL seasons has been the disparity between the top placed and the bottom placed teams with the result that many of the games have been one-sided. This notebook does some basic EDA, followed by a simple model to allocate players to teams such that all IPL teams are balanced. This allocation is based on the performance of all players in the last IPL season.
# This notebook is structured as follows:
# * Sanity check of data
# * Basic cleaning of data
# * Basic EDA
# * Feature extraction 
# * Model to generate a balanced distribution of players across IPL teams based on their last season's performance

# **Let's first import the necessary modules:**

# In[ ]:


import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from sklearn.cluster import MiniBatchKMeans
import random
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.simplefilter('ignore')


# **Now, let's read the data sets and perform some basic sanity checks and data cleaning.**

# In[ ]:


matches = pd.read_csv('../input/matches.csv')
deliveries = pd.read_csv('../input/deliveries.csv')


# **Check if each match id in deliveries has an entry in matches:**

# In[ ]:


chk = deliveries.merge(matches, left_on='match_id', right_on='id', how='left')
print(np.sum(np.isnan(chk['win_by_runs'])))


# **Check if there are 20 overs at max. in each inning:**

# In[ ]:


chk = deliveries.groupby(['match_id', 'inning'])['over'].aggregate(np.max)
print(sum(chk>20)) #each inning is at max 20 overs


# **Check if there at max. 6 legitmate deliveries per over:**

# In[ ]:


chk = deliveries.groupby(['match_id', 'inning', 'over'])['ball', 'wide_runs', 'noball_runs'].aggregate([np.sum, np.max])
chk.drop(labels=[('ball', 'sum'), ('wide_runs', 'amax'), ('noball_runs', 'amax')], inplace=True, axis=1)
print(sum(chk['ball']['amax'] - chk['wide_runs']['sum'] - chk['noball_runs']['sum'] > 6))


# **And surprisingly, there are 3 overs with over 6 legitimate deliveries!
# Let's see which overs have this discrepancy:**

# In[ ]:


print(chk[(chk['ball']['amax'] - chk['wide_runs']['sum'] - chk['noball_runs']['sum'] >6)])


# **These 3 anomalies may have been umpiring errors in all likelihood, with an extra delivery bowled. Let's ignore them for now.
# Let's now check if there are any duplicate deliveries ie. does ball i occur more than once within the same over in any inning:**

# In[ ]:


chk = deliveries.groupby(['match_id', 'inning', 'over', 'ball'])['batting_team'].aggregate(np.count_nonzero)
chk[(chk>1)]


# **And there are 8 duplicates are present ie. there are 8 cases where ball 1 is repeated within the same over in the same inning.
# On further analysis, we can see this is actually the 10th delivery due to 4 wides/noballs in that over.
# In the ball column, 10 seems to have been truncated in the deliveries data file to 1, possibly because of a fixed column size defined in the csv for this column when the csv was created.
# We can set these back to 10 to clean up the data, so let's do that now:**

# In[ ]:


chk = pd.DataFrame(chk[(chk>1)]).reset_index()[['match_id', 'inning', 'over', 'ball']]
duplicates = deliveries.merge(chk, how='inner',on=['match_id', 'inning', 'over', 'ball'])
tenthballrecords = duplicates.drop_duplicates(subset=['match_id', 'inning', 'over', 'ball'], keep='last')
tenthballrecords.loc[:, 'ball'] = 10
deliveries = deliveries.drop_duplicates(subset=['match_id', 'inning', 'over', 'ball'], keep='first')
deliveries = deliveries.append(tenthballrecords).sort_values(by=['match_id', 'inning', 'over', 'ball'])


# **All good now - duplicate 1 has been changed to 10 for those 8 cases. We can re-verify now:**

# In[ ]:


chk = deliveries.groupby(['match_id', 'inning', 'over', 'ball'])['batting_team'].aggregate(np.count_nonzero)
sum(chk>1)==0 #True now


# **Are there max 11 players batting in each match per team?**

# In[ ]:


chk = deliveries.groupby(['match_id', 'inning'])['batsman'].nunique()
print(sum(chk>11)==0) #all good. Max 11 players batting per team
chk = deliveries.groupby(['match_id', 'inning'])['bowler'].nunique()
print(sum(chk>11)==0) #all good. Max 11 players bowler per team


# **All good now. Let's perform one last clean-up action and rename all occurrences of "Deccan Chargers" to "Sunrisers Hyderabad" since the team name changed along with ownership a few seasons back, but the team itself remained largely the same.**

# In[ ]:


matches.replace(to_replace='Deccan Chargers', value='Sunrisers Hyderabad', inplace=True)
deliveries.replace(to_replace='Deccan Chargers', value='Sunrisers Hyderabad', inplace=True)


# **Let's do some basic exploratory analysis now.
# First, we will create a dict to map team names to abbreviated team names for ease of display.
# Then, let's see the number of wins by team looks:**

# In[ ]:


teamdict = dict({'Chennai Super Kings': 'CSK', 'Delhi Daredevils': 'DD', 'Gujarat Lions': 'GL', 'Kings XI Punjab': 'KXIP', 'Kochi Tuskers Kerala': 'KTK', 'Kolkata Knight Riders': 'KKR', 'Mumbai Indians': 'MI', 'Pune Warriors': 'PW', 'Rajasthan Royals': 'RR', 'Rising Pune Supergiants': 'RPG', 'Royal Challengers Bangalore': 'RCB', 'Sunrisers Hyderabad': 'SRH'})
winners = matches.groupby('winner')['id'].aggregate(len).sort_values(ascending=False)
teams = [teamdict[t] for t in winners.index]
wins = winners.values
plot = plt.figure()
plt.bar(range(len(teams)), wins)
plt.gca().set_xlabel('Team')
plt.gca().set_ylabel('Wins')
plt.title('Wins by team')
plt.xticks(range(len(teams)), teams)
x = plt.gca().xaxis
# rotate the tick labels for the x axis
for item in x.get_ticklabels():
    item.set_rotation(45)


# **GL and RPG look skewed in the above graph since they are recent additions to IPL. Let's plot win % instead.**
# **Let's also overlay the average runs scored per match by team over the win % graph.**

# In[ ]:


winners = pd.DataFrame(matches.groupby('winner')['id'].aggregate({'wins':len}))
played = pd.DataFrame(matches.groupby('team1')['id'].aggregate({'played':len}) + matches.groupby('team2')['id'].aggregate({'played':len}), columns=['played'])
winners = played.merge(winners, how='inner', left_index=True, right_index=True)
winners['win_pct'] = np.round((winners['wins']/winners['played']) * 100, decimals=0)
winners.sort_values('win_pct', ascending=False, inplace=True)
teams = [teamdict[t] for t in winners.index]

#Let's also overlay the average runs scored per match by team over the win % graph.
runs = deliveries.groupby('batting_team')['total_runs'].aggregate({'runs':np.sum})
runs = runs.merge(played, how='inner', left_index=True, right_index=True)
runs = runs['runs']/runs['played']
runs = [runs[idx] for idx in winners.index] #sort runs in the same order as the winners data

fig, ax1 = plt.subplots()
#plt.rcParams["figure.figsize"] = [9, 4]
ax1.bar(range(len(teams)), winners['win_pct'], label='Win %')
ax1.set_xlabel('Team')
ax1.set_ylabel('Win %')
plt.xticks(range(len(teams)), teams)

ax2 = ax1.twinx()
ax2.plot(range(len(teams)), runs, '-xr')
ax1.plot(np.nan, '-r', label='Avg. Runs') #workaround to get the legend to display both graphs
ax1.legend(loc=0)
ax2.set_ylabel('Avg. runs per match')
ax2.set_ylim(0, 175)

plt.title('Win % by team');


# This looks much better and we can see GL has actually performed quite well.
# We get some interesting insights out of this graph:
# * We can see that teams like RCB, KXIP, DD, RPG get relatively high scores, but still don't win often. This may mean their bowling is not as strong as the top teams
# * CSK has the best batting score average and also have the best win % and clearly are one of the best and most consistent sides through the seasons.
# 
# Let's now see the same graph with win % with wickets taken overlaid on top.

# In[ ]:


wkts = deliveries.groupby('bowling_team')['player_dismissed'].aggregate({'wickets':'count'})
wkts = wkts.merge(played, how='inner', left_index=True, right_index=True)
wkts = wkts['wickets']/wkts['played']
wkts = [wkts[idx] for idx in winners.index] #sort wickets in the same order as the winners data

fig, ax1 = plt.subplots()
#plt.rcParams["figure.figsize"] = [9, 4]
ax1.bar(range(len(teams)), winners['win_pct'], label='Win %')
ax1.set_xlabel('Team')
ax1.set_ylabel('Win %')
plt.xticks(range(len(teams)), teams)

ax2 = ax1.twinx()
ax2.plot(range(len(teams)), wkts, '-xr')
ax1.plot(np.nan, '-r', label='Avg. Wkts') #workaround to get the legend to display both graphs
ax1.legend(loc=0)
ax2.set_ylabel('Avg. wkts per match')
ax2.set_ylim(0, 7)

plt.title('Win % by team');


# **We can clearly see GL has one of the weakest bowling sides, yet have a good win%. This indicates that their batting is much stronger than their bowling.**
# 
# **Given their performance in the last season, we know the above is true, and also indicates they can be a much stronger side if they improve their bowling combination.
# CSK and MI are again the leaders here and shows once more that they are the most consistent and well-balanced sides over the seasons.**
# 
# **Let's now create a distribution of runs per over scored:**

# In[ ]:


avg_runs_per_over = deliveries.groupby('over')['total_runs'].aggregate({'runs':'sum'})
overs_count = deliveries.groupby(['match_id', 'inning', 'over'])['ball'].aggregate({'ball':'max'}).reset_index().groupby('over')['match_id'].aggregate({'count':'count'})
avg_runs_per_over = avg_runs_per_over.merge(overs_count, how='inner', left_index=True, right_index=True)
avg_runs_per_over['rpo'] = np.round(avg_runs_per_over['runs']/avg_runs_per_over['count'], 2)
#Above gives us the rpo for each over across all matches
#Let's get a violin plot for number of runs per over.
runs_per_over = deliveries.groupby(['match_id', 'inning', 'over'])['total_runs'].aggregate({'runs': 'sum'}).reset_index().drop(labels=['match_id', 'inning'], axis=1)
data = [runs_per_over[(runs_per_over['over']==o)]['runs'] for o in np.unique(runs_per_over['over'])]
plt.rcParams["figure.figsize"] = [15, 4];
fig = plt.figure()
plt.violinplot(data, showmeans=True);
plt.xticks(range(len(data)+1));
plt.title('Distribution of runs per over');
plt.xlabel('Over number');
plt.ylabel('Runs scored');


# **We can see from the shapes of the plots that teams typically (and understandably) start slow in the first 1 or 2 overs.
# We can then see a period of acceleration between overs 3-6 which corresponds to the closing of the powerplay,
# then a slowdown in the overs 7-11 with the slowdown particularly marked in overs 7 and 8 (which is right after the powerplay ends), and then a gradual acceleration with a final surge in overs 16-20.**
# 
# **Now, how about running a k-means cluster to group the overs into 3 clusters based on average runs per over and see if it correlates with our observations above:**

# In[ ]:


def circle(x, y, radius=0.5, clr='black'):
    from matplotlib.patches import Circle
    from matplotlib.patheffects import withStroke
    circle = Circle((x, y), radius, clip_on=False, zorder=10, linewidth=1,
                    edgecolor=clr, facecolor=(0, 0, 0, .0125),
                    path_effects=[withStroke(linewidth=5, foreground='w')])
    plt.gca().add_artist(circle)


def text(x, y, text, clr='blue'):
    plt.gca().text(x, y, text, backgroundcolor="white",
            ha='center', va='top', weight='bold', color=clr)

#Now, how about running a k-means cluster to group the overs into 3 clusters based on average runs per over and see if it correlates with our observations above.
from sklearn.cluster import MiniBatchKMeans
kmodel = MiniBatchKMeans(n_clusters=3, random_state=0).fit(avg_runs_per_over['rpo'].values.reshape(-1, 1))
cluster_colours = ['#4EACC5', '#FF9C34', '#4E9A06']
clusters = kmodel.predict(avg_runs_per_over['rpo'].reshape(-1,1))
colours = [cluster_colours[clusters[o]] for o in range(len(clusters))]
fig = plt.figure()
plt.bar(avg_runs_per_over.index, avg_runs_per_over['rpo'], color=colours, label='Avg. runs in each over');
plt.title('k-means cluster with 3 clusters based on average runs in each over');
plt.xlabel('Over number');
plt.ylabel('Avg. runs in over');
plt.xticks(range(len(clusters)+1));
circle(6, -0.4, clr='blue')
text(6, -1, 'End of powerplay', clr='blue')


# **This correlates quite well with our previous observation ie. over 1-2 and 7-10 are relatively slow and belong to 1 cluster, overs 3-6 and 11-16 show runs being scored at a slightly faster rate and belong to a second cluster, and the last few overs show a significant surge and form a separate cluster.**
# 
# **Next, let's do a little more of EDA and check how important are sixes and fours to win %:**

# In[ ]:


fours = deliveries[(deliveries['batsman_runs']==4)]
sixes = deliveries[(deliveries['batsman_runs']==6)]

fours = fours.groupby('batting_team')['batsman_runs'].aggregate({'count':'count'})
sixes = sixes.groupby('batting_team')['batsman_runs'].aggregate({'count':'count'})

fours = fours.merge(played, how='inner', left_index=True, right_index=True)
sixes = sixes.merge(played, how='inner', left_index=True, right_index=True)
fours = fours['count']/fours['played']
sixes = sixes['count']/sixes['played']


fours = [fours[idx] for idx in winners.index] #sort fours in the same order as the winners data
sixes = [sixes[idx] for idx in winners.index] #sort sixes in the same order as the winners data

fig, ax1 = plt.subplots()
#plt.rcParams["figure.figsize"] = [9, 4]
ax1.bar(range(len(teams)), winners['win_pct'], label='Win %')
ax1.set_xlabel('Team')
ax1.set_ylabel('Win %')
plt.xticks(range(len(teams)), teams)

ax2 = ax1.twinx()
ax2.plot(range(len(teams)), fours, '-xr')
ax2.plot(range(len(teams)), sixes, '-xg')
ax2.plot(range(len(teams)), np.array(fours)+np.array(sixes), '-xy')
ax1.plot(np.nan, '-r', label='# of 4s') #workaround to get the legend to display both graphs
ax1.plot(np.nan, '-g', label='# of 6s') #workaround to get the legend to display both graphs
ax1.plot(np.nan, '-y', label='# of 4s+6s') #workaround to get the legend to display both graphs
ax1.legend(loc=0)
ax2.set_ylabel('# of 4s and 6s')

plt.title('Win % by team vs. # of 4s/6s hit');


# **From this, we can see that RCB and KXIP hit a relatively high number of 4s and 6s, yet do not win as much as the top teams.
# This indicates that either their bowling is weak, or they don't take singles/twos/threes effectively.**
# 
# **Let's also look at effect of 4s and 6s in both matches won and in matches lost:**

# In[ ]:


fours = deliveries[(deliveries['batsman_runs']==4)].merge(matches, left_on=['match_id', 'batting_team'], right_on=['id', 'winner'], how='left')
sixes = deliveries[(deliveries['batsman_runs']==6)].merge(matches, left_on=['match_id', 'batting_team'], right_on=['id', 'winner'], how='left')
fours_lost = fours[pd.isnull(fours['winner'])]
sixes_lost = sixes[pd.isnull(sixes['winner'])]
fours_match_count = fours_lost.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})
sixes_match_count = sixes_lost.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})
fours_lost = fours_lost.groupby('batting_team')['batsman_runs'].aggregate({'fours':'count'})
sixes_lost = sixes_lost.groupby('batting_team')['batsman_runs'].aggregate({'sixes':'count'})

fours_lost = fours_lost.merge(pd.DataFrame(fours_match_count), how='inner', left_index=True, right_index=True)
sixes_lost = sixes_lost.merge(pd.DataFrame(sixes_match_count), how='inner', left_index=True, right_index=True)
fours_lost['4s_per_match'] = fours_lost['fours']/fours_lost['played']
sixes_lost['6s_per_match'] = sixes_lost['sixes']/sixes_lost['played']
data_lost = pd.merge(fours_lost, sixes_lost, left_index=True, right_index=True, how='inner')
data_lost.sort_index(inplace=True)

fours_won = fours[~pd.isnull(fours['winner'])]
sixes_won = sixes[~pd.isnull(sixes['winner'])]
fours_match_count = fours_won.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})
sixes_match_count = sixes_won.groupby('batting_team')['match_id'].aggregate({'played':'nunique'})
fours_won = fours_won.groupby('batting_team')['batsman_runs'].aggregate({'fours':'count'})
sixes_won = sixes_won.groupby('batting_team')['batsman_runs'].aggregate({'sixes':'count'})

fours_won = fours_won.merge(pd.DataFrame(fours_match_count), how='inner', left_index=True, right_index=True)
sixes_won = sixes_won.merge(pd.DataFrame(sixes_match_count), how='inner', left_index=True, right_index=True)
fours_won['4s_per_match'] = fours_won['fours']/fours_won['played']
sixes_won['6s_per_match'] = sixes_won['sixes']/sixes_won['played']
data_won = pd.merge(fours_won, sixes_won, left_index=True, right_index=True, how='inner')
data_won.sort_index(inplace=True)

fig = plt.figure()
plt.bar(np.array(range(len(data_lost.index)))-0.2, data_lost['4s_per_match'], color='blue', width=0.2, label='Avg. 4s in losses')
plt.bar(np.array(range(len(data_lost.index)))-0.2, data_lost['6s_per_match'], color='red', width=0.2, label='Avg. 6s in losses', bottom=data_lost['4s_per_match'])
plt.bar(np.array(range(len(data_won.index))), data_won['4s_per_match'], color='grey', width=0.2, label='Avg. 4s in wins')
plt.bar(np.array(range(len(data_won.index))), data_won['6s_per_match'], color='green', width=0.2, label='Avg. 6s in wins', bottom=data_won['4s_per_match'])
plt.xlabel('Team')
plt.ylabel('Avg. # of 4s and 6s in lost and won matches')
plt.xticks(range(len(data_lost.index)), [teamdict[t] for t in data_lost.index])
plt.yticks([3*n for n in range(10)])
plt.legend(frameon=False, loc=9)
plt.title('Avg. # of 4s/6s in lost and won matches');


# **KKR and SRH show very little difference between wins and losses in terms of 4s and 6s hit. This indicates that when they win, it is either because they rotate the strike well, or because their bowling wins them matches.**
# 
# **On the other hand, GL looks to be very reliant on 4s and 6s to win them matches - this is in line with what we saw before ie. GL is reliant on batting and does not have a balanced bowling attack.**

# **In the last few IPL seasons, many of the matches ended up being one-sided, and there were some teams that were clearly significantly better than the others.**
# 
# **Let's do an experimental clustering analysis to see if we can re-distribute all players among the IPL teams so that each team is as good as the other.**
# 
# **For this, we will consider only the players and performance in the last season, 2016. Given that we don't have data sets to identify if a player is a batsman, bowler, keeper or allrounder, we will cluster all players together instead of by type of player.**
# 
# **If we had data around whether a player is a batsman, bowler, all-rounder, keeper, we could cluster them separately. Similarly, if we had data around Indian and foreign players, we could cluster them as well separately.**
# 
# **For the purpose of this exercise, we will just ignore player type and cluster all of them together based on their batting performance and bowling performance.**
# 
# **Let's create some features now for each player.**

# In[ ]:


matches_2016 = matches[(matches['season']==2016)]
deliveries_2016 = deliveries.merge(matches_2016, left_on='match_id', right_on='id', how='inner')
bowler_list = np.unique(deliveries_2016['bowler'])
batsman_list = np.unique(deliveries_2016['batsman'])
player_list = pd.DataFrame(np.union1d(bowler_list, batsman_list), columns=['player'])
#Let's now add features to the player dataframe
#Can the player bowl?
player_list = player_list.merge(pd.DataFrame(sorted(zip(bowler_list, np.ones(len(bowler_list)))), columns=['player', 'can_bowl']), how='left', left_on='player', right_on='player')
player_list.loc[(pd.isnull(player_list['can_bowl'])), 'can_bowl'] = 0
player_list.set_index('player', inplace=True)
#For the rest of the features, we follow the rule that better performance implies higher value for the feature. Poorer performance should result in lower value for the feature.
#This means that batsman scoring runs would be a regular feature, whereas the number of times he got out would need to be made a reciprocal of itself in the feature.
#Similarly, for bowling performance, wickets is a regular feature, whereas wides needs to be made a reciprocal of itself.
#The reason for this is to ensure the clustering works consistently on all features.
#how many runs in total did the player score
player_list = player_list.merge(deliveries_2016.groupby('batsman')['batsman_runs'].aggregate({'runs':'sum'}), how='left', left_index=True, right_index=True)
player_list.loc[(pd.isnull(player_list['runs'])), 'runs'] = 0
#how many times did the batsman get out
player_list = player_list.merge(deliveries_2016.groupby('batsman')['player_dismissed'].aggregate({'outs':'count'}), how='left', left_index=True, right_index=True)
player_list['outs'] = 1/player_list['outs']
player_list.loc[(pd.isnull(player_list['outs'])), 'outs'] = 1
player_list.loc[(np.isinf(player_list['outs'])), 'outs'] = 1
#how many times was the batsman involved in a run-out while at the non-striker end
player_list = player_list.merge(deliveries_2016[(deliveries_2016['dismissal_kind']=='run out')].groupby('non_striker')['player_dismissed'].aggregate({'runouts':'count'}), how='left', left_index=True, right_index=True)
player_list['runouts'] = 1/player_list['runouts']
player_list.loc[(pd.isnull(player_list['runouts'])), 'runouts'] = 1
player_list.loc[(np.isinf(player_list['runouts'])), 'runouts'] = 1
#how many 4s did the batsman hit
player_list = player_list.merge(deliveries_2016[(deliveries_2016['batsman_runs']==4)].groupby('batsman')['batsman_runs'].aggregate({'4s':'count'}), how='left', left_index=True, right_index=True)
player_list.loc[(pd.isnull(player_list['4s'])), '4s'] = 0
#how many 6s did the batsman hit
player_list = player_list.merge(deliveries_2016[(deliveries_2016['batsman_runs']==6)].groupby('batsman')['batsman_runs'].aggregate({'6s':'count'}), how='left', left_index=True, right_index=True)
player_list.loc[(pd.isnull(player_list['6s'])), '6s'] = 0
#how many balls did the batsman face
player_list = player_list.merge(deliveries_2016.groupby('batsman')['ball'].aggregate({'balls_faced':'count'}), how='left', left_index=True, right_index=True)
player_list.loc[(pd.isnull(player_list['balls_faced'])), 'balls_faced'] = 0
#how many wickets did the player take
player_list = player_list.merge(deliveries_2016.groupby('bowler')['player_dismissed'].aggregate({'wickets':'count'}), how='left', left_index=True, right_index=True)
player_list.loc[(pd.isnull(player_list['wickets'])), 'wickets'] = 0
#how many runs did the bowler give away
player_list = player_list.merge(deliveries_2016.groupby('bowler')['batsman_runs'].aggregate({'bowl_runs':'sum'}), how='left', left_index=True, right_index=True)
player_list['bowl_runs'] = 1/player_list['bowl_runs']
player_list.loc[(pd.isnull(player_list['bowl_runs'])), 'bowl_runs'] = 1
player_list.loc[(np.isinf(player_list['bowl_runs'])), 'bowl_runs'] = 1
#how many wides did the player bowl
player_list = player_list.merge(deliveries_2016.groupby('bowler')['wide_runs'].aggregate({'wides':'sum'}), how='left', left_index=True, right_index=True)
player_list['wides'] = 1/player_list['wides']
player_list.loc[(pd.isnull(player_list['wides'])), 'wides'] = 1
player_list.loc[(np.isinf(player_list['wides'])), 'wides'] = 1
#how many no-balls did the player bowl
player_list = player_list.merge(deliveries_2016.groupby('bowler')['noball_runs'].aggregate({'noballs':'sum'}), how='left', left_index=True, right_index=True)
player_list['noballs'] = 1/player_list['noballs']
player_list.loc[(pd.isnull(player_list['noballs'])), 'noballs'] = 1
player_list.loc[(np.isinf(player_list['noballs'])), 'noballs'] = 1
#how many balls did the player bowl
player_list = player_list.merge(deliveries_2016.groupby('bowler')['ball'].aggregate({'deliveries':'count'}), how='left', left_index=True, right_index=True)
player_list.loc[(pd.isnull(player_list['deliveries'])), 'deliveries'] = 0
#how many wickets did the player effect as a fielder
player_list = player_list.merge(deliveries_2016.groupby('fielder')['player_dismissed'].aggregate({'wickets_effected':'count'}), how='left', left_index=True, right_index=True)
player_list.loc[(pd.isnull(player_list['wickets_effected'])), 'wickets_effected'] = 0


# In[ ]:


#get the number of matches each player batted and bowled
player_list['bat_matches'] = deliveries_2016.groupby('batsman')['match_id'].aggregate({'bat_matches':'nunique'})
player_list.loc[(pd.isnull(player_list['bat_matches'])), 'bat_matches'] = 0
player_list['bowl_matches'] = deliveries_2016.groupby('bowler')['match_id'].aggregate({'bowl_matches':'nunique'})
player_list.loc[(pd.isnull(player_list['bowl_matches'])), 'bowl_matches'] = 0


# **Let's add some derived features now:**

# In[ ]:


player_list['strike_rate'] = player_list['runs']/player_list['balls_faced']
player_list.loc[(pd.isnull(player_list['strike_rate'])), 'strike_rate'] = 0
player_list['bat_average'] = player_list['runs']/player_list['bat_matches']
player_list.loc[(pd.isnull(player_list['bat_average'])), 'bat_average'] = 0
player_list['4s_per_balls'] = player_list['4s']/player_list['balls_faced']
player_list.loc[(pd.isnull(player_list['4s_per_balls'])), '4s_per_balls'] = 0
player_list['6s_per_balls'] = player_list['6s']/player_list['balls_faced']
player_list.loc[(pd.isnull(player_list['6s_per_balls'])), '6s_per_balls'] = 0
player_list['4s_per_match'] = player_list['4s']/player_list['bat_matches']
player_list.loc[(pd.isnull(player_list['4s_per_match'])), '4s_per_match'] = 0
player_list['6s_per_match'] = player_list['6s']/player_list['bat_matches']
player_list.loc[(pd.isnull(player_list['6s_per_match'])), '6s_per_match'] = 0
player_list['outs_per_match'] = player_list['outs']/player_list['bat_matches']
player_list['outs_per_match'] = 1/player_list['outs_per_match']
player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'outs_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']
player_list.loc[(np.isinf(player_list['outs_per_match'])), 'outs_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']
player_list['runouts_per_match'] = player_list['runouts']/player_list['bat_matches']
player_list['runouts_per_match'] = 1/player_list['runouts_per_match']
player_list.loc[(pd.isnull(player_list['runouts_per_match'])), 'runouts_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']
player_list.loc[(np.isinf(player_list['runouts_per_match'])), 'runouts_per_match'] = player_list.loc[(pd.isnull(player_list['outs_per_match'])), 'bat_matches']

player_list['bowl_strike_rate'] = player_list['wickets']/player_list['deliveries']
player_list.loc[(pd.isnull(player_list['bowl_strike_rate'])), 'bowl_strike_rate'] = 0
player_list['bowl_average'] = player_list['bowl_runs']*player_list['bowl_matches'] #since bowl_runs was made reciprocal of itself earlier
player_list.loc[(pd.isnull(player_list['bowl_average'])), 'bowl_average'] = 1
player_list['wides_per_match'] = player_list['wides']*player_list['bowl_matches'] #since wides was made reciprocal of itself earlier
player_list.loc[(pd.isnull(player_list['wides_per_match'])), 'wides_per_match'] = 1
player_list['noballs_per_match'] = player_list['noballs']*player_list['bowl_matches'] #since noballs was made reciprocal of itself earlier
player_list.loc[(pd.isnull(player_list['noballs_per_match'])), 'noballs_per_match'] = 1
player_list['wickets_per_match'] = player_list['wickets']/player_list['bowl_matches']
player_list.loc[(pd.isnull(player_list['wickets_per_match'])), 'wickets_per_match'] = 0


# **We don't need the below features as we have derived features based on these and these don't directly relate to player performance. So let's drop the below features:**

# In[ ]:


player_list.drop(labels=['balls_faced', 'deliveries', 'bat_matches', 'bowl_matches'], inplace=True, axis=1)


# **Scale the features now:**

# In[ ]:


scaler = MinMaxScaler()
player_names = pd.DataFrame(player_list.index)
player_list = scaler.fit_transform(player_list)


# **We are now ready to cluster the players into the required number of clusters to distribute across the teams:**

# In[ ]:


num_clusters = len(matches_2016.groupby('team1')['id'].nunique())
kmodel = MiniBatchKMeans(n_clusters=num_clusters, random_state=0).fit(player_list)
player_names['cluster'] = kmodel.predict(player_list)
player_names = pd.DataFrame(player_names.groupby('cluster')['player'].unique())
[print('Cluster ', i+1, ': ', list(player_names.iloc[i,0])) for i in range(num_clusters)]


# **We have now got the players clustered into 8 clusters based on their performance.**
# 
# **Now, to have an equitable distribution of these players among the 8 IPL teams, we need to make sure we allocate players from each cluster to every team ie. if player X is allocated to team 1 from cluster 1, players from cluster 1 need to be allocated to the other 7 teams as well.**
# 
# **Let's do a random such allocation that follows the above rule and generate the 8 teams.**

# In[ ]:


balanced_teams = []
for i in range(num_clusters):
    player_arr = player_names.iloc[i,0]
    num_sel_per_team = int(np.floor(len(player_arr) / num_clusters))
    rand_indexes = random.sample(range(len(player_arr)), num_sel_per_team*num_clusters)
    for j in range(num_sel_per_team):
        balanced_teams.append(player_arr[rand_indexes[j*8:(j+1)*8]])
balanced_teams = pd.DataFrame(balanced_teams, columns=['Team ' + str(i+1) for i in range(num_clusters)])
print(balanced_teams)


# **There! That does't look too bad, and on the face of it, does look like a balanced distribution of players across teams. Let's get rid of the auction process now :-)**
# 
# **Jokes apart, this can do with a considerable amount of tuning:**
# **
# 1. Update the data set to include the type of player for each player ie. batsman, bowler, allrounder or wicket-keeper
# 2. Run the clustering multiple times, once for batsmen, once for bowler etc.
# 3. Use the different sets of clusters created to allocate batsmen, bowlers, allrounders and wicket-keepers separately. This will make the distribution of players to teams much more realistic and balanced
# 4. Include data around whether the player is an Indian or an outstation player. Cluster the outstation players separately so that each team gets a fair representation of outstation players while also balancing them using the performance clusters
# 5. Futher categorise batsmen as openers, middle-order and bowlers as pace and spin bowlers and cluster them separately as well.**
# 
# **If we do all of the above, we will probably get a very balanced and improved distribution of players across all teams.
# **

# **Finally, thanks, once again, to Manas for providing this wonderful data set.**
