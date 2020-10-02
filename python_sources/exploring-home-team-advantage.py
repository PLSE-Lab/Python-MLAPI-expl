#!/usr/bin/env python
# coding: utf-8

# # Comparing goals scored in major European football leagues

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.gridspec as gridspec
from numpy import random

with sqlite3.connect('../input/database.sqlite') as con:
    countries = pd.read_sql_query("SELECT * from Country", con)
    matches = pd.read_sql_query("SELECT * from Match", con)
    leagues = pd.read_sql_query("SELECT * from League", con)
    teams = pd.read_sql_query("SELECT * from Team", con)
    tempmatch = pd.read_sql_query("SELECT * from Match", con)


# In[ ]:


#Subsetting the five countries of interest
main_countries = ['England','France','Germany','Italy','Spain']
countries = countries[countries.name.isin(main_countries)]
leagues = countries.merge(leagues,on='id',suffixes=('', '_y'))
seasons = matches.season.unique()
leagues


# In[ ]:


def res(row):
    if row['home_team_goal'] == row['away_team_goal']:
        val = 0
    elif row['home_team_goal'] > row['away_team_goal']:
        val = 1
    else:
        val = -1
    return val


# In[ ]:


#Merge the leagues with corresponding matches
req_matches = matches[matches.league_id.isin(leagues['id'])]
req_matches = req_matches[['id','league_id','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal','season']]
req_matches["total_goals"] = req_matches['home_team_goal'] + req_matches['away_team_goal']
req_matches["result"] = req_matches.apply(res,axis = 1)
req_matches.dropna(inplace=True)
req_matches.head()


# In[ ]:


#Separating the leagues for plotting and further analysis
new_matches = pd.merge(req_matches,leagues,left_on='league_id', right_on='id')
new_matches = new_matches.drop(['id_x','id_y','country_id'],axis = 1)
english = new_matches[new_matches.name == "England"]
french = new_matches[new_matches.name == "France"]
italian = new_matches[new_matches.name == "Italy"]
spanish = new_matches[new_matches.name == "Spain"]
german = new_matches[new_matches.name == "Germany"]
# sum_goals = new_group_matches.home_team_goal.sum()
e = english.groupby('season')
f = french.groupby('season')
i = italian.groupby('season')
s = spanish.groupby('season')
g = german.groupby('season')
seasons


# In[ ]:


#Plotting total goals scored each season
fig = plt.figure(figsize=(10, 10))
plt.title("Total goals in each season")
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlabel("Season")
plt.ylabel("Total Goals Scored")
num_seasons = range(len(seasons))
plt.plot(num_seasons,e.total_goals.sum().values,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f.total_goals.sum().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g.total_goals.sum().values,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i.total_goals.sum().values,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s.total_goals.sum().values,label = "Spanish La Liga", marker = 'o')
plt.legend()


# The Spanish La Liga teams resort to much more aggressive tactics and formations every game as compared to the other leagues' teams. This, I believe, translates directly to more goals being scored than in other leagues.
# 
# Also, even though it might look like the Bundesliga teams scored fewer goals each season, it is to be remember that they play fewer games compared to teams from other leagues. In Bundesliga, there are only 18 teams playing each season and hence, they play a total of 306 games. In contrast, the other leagues have 20 teams and hence end up playing 380 games.
# 
# Therefore, it would be a good idea to look at the average number of goals scored each game by season.
# 

# In[ ]:


#Plotting average goals scored each season
fig = plt.figure(figsize=(10, 10))
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlabel("Season")
plt.title("Average goals per game each season")
plt.ylabel("Average goals per game")
plt.plot(num_seasons,e.total_goals.mean().values,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f.total_goals.mean().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g.total_goals.mean().values,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i.total_goals.mean().values,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s.total_goals.mean().values,label = "Spanish La Liga", marker = 'o')
#plt.xlim = (-20,20)
plt.legend(loc = 2)


# The Bundesliga teams score way more on average in each game that the teams in other leagues. In the 2013/2014 season, Bundesliga teams scored more than 3 goals each game on an average (967 goals in 306 matches - 3.16 per match). And teams in Ligue 1 have scored the least number of goals on a per game basis. 

# Investigating home team advantage?
# 
# There is always the conspicuous home team advantage in every sport but does it truly exist? Do teams really end up scoring more goals at their home stadium than when playing at an away fixture?
# 
# Let's try plotting number of goals scored at home (vs away) and subsequently the number of home wins.

# In[ ]:


#Plotting home/away scored each season
fig = plt.figure(figsize=(10, 10))
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.title('Home-away goal ratio each season')
plt.xlabel('Season')
plt.ylabel('Home-goals to Away-goals ratio')
plt.plot(num_seasons,e.home_team_goal.mean().values / e.away_team_goal.mean().values,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f.home_team_goal.mean().values / f.away_team_goal.mean().values,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g.home_team_goal.mean().values / g.away_team_goal.mean().values,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i.home_team_goal.mean().values / i.away_team_goal.mean().values,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s.home_team_goal.mean().values / s.away_team_goal.mean().values,label = "Spanish La Liga", marker = 'o')
#plt.xlim = (-20,20),
plt.legend(loc = 1)


# 
# In the last five seasons, La Liga teams have had a strong home advantage. This could be a key factor when they host teams from other leagues for European Leagues like the UEFA Championship or the Europa League. In such championship matches, there are usually two fixtures played (home and away) by each team. And since goals scored in each of these fixtures plays a role in teams qualifying to the next round, there seems to a big advantage to the La Liga teams in recent seasons.
# 
# The ratios for other leagues seem to vary significantly and not show any significant trends except the Bundesliga, again, which keeps oscillating between a strong home game and weak home game season. 
# 
# All the (home-goals/away-goals) ratios are above 1 and we can consistently see the ratio being above 1.3 for most leagues. This only shows that, on an average, teams score more at home fixtures than away. But how many of these matches do they win?
# 
# Again, since Bundesliga teams play 74 fewer games than the other league teams, let's plot the home-away win ratio for each league every season.

# In[ ]:


#Subsetting homewins vs homeloss from each of the leagues - ignoring draws.
e_hw = np.true_divide(english[english.result == 1].groupby('season').result.sum().values,english[english.result == -1].groupby('season').result.sum().values * -1)
f_hw = np.true_divide(french[french.result == 1].groupby('season').result.sum().values,french[french.result == -1].groupby('season').result.sum().values *-1)
g_hw = np.true_divide(german[german.result == 1].groupby('season').result.sum().values,german[german.result == -1].groupby('season').result.sum().values*-1)
i_hw = np.true_divide(italian[italian.result == 1].groupby('season').result.sum().values,italian[italian.result == -1].groupby('season').result.sum().values*-1)
s_hw = np.true_divide(spanish[spanish.result == 1].groupby('season').result.sum().values,spanish[spanish.result == -1].groupby('season').result.sum().values*-1)


# In[ ]:


#Plotting number of home wins vs home losses each season
fig = plt.figure(figsize=(10, 10))
plt.xticks(range(len(seasons)),seasons)
plt.style.use('ggplot')
plt.xlim = (-20,20)
plt.ylim = (0,120)
plt.title("Number of home wins each vs home loss each season")
plt.xlabel("Season")
plt.ylabel("Home Wins vs loss")
plt.plot(num_seasons,e_hw,label = "English Premier League", marker = 'o')
plt.plot(num_seasons,f_hw,label = "France Ligue 1", marker = 'o')
plt.plot(num_seasons,g_hw,label = "German Bundesliga", marker = 'o')
plt.plot(num_seasons,i_hw,label = "Italian Serie A", marker = 'o')
plt.plot(num_seasons,s_hw,label = "Spanish La Liga", marker = 'o')
plt.legend(loc = 1)


# We can see that a few leagues are winning fewer matches at home recently. In particular, there seems to be a steady decreasing trend in the French Ligue and the English Premier League. However, we are plotting only the home wins against the home losses and not taking draws into consideration. 
# 
# We could analyse the home-goals analysis for each team in the lague and also compare the ratio of number of home wins to home losses. In doing so, we could choose one of two things :
# 1. Include draw as a loss for the home team
# 2. Include draw as a win for the home team

# In[ ]:


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1*height,
                '%d' % int(height),
                ha='center', va='bottom')


# In[ ]:


#New dataframe merging home_team names and matches.
matches_w_teams = pd.merge(new_matches,teams,left_on='home_team_api_id', right_on='team_api_id')
matches_w_teams = matches_w_teams.drop(['id','team_api_id','team_fifa_api_id'],axis = 1)
matches_w_teams = matches_w_teams.rename(columns={'team_long_name':'home_team_long_name','name_y':'league_name','name':'country_name'})
matches_w_teams.head(1)


# In[ ]:


#Color scheme for each country - from colourbrewer2.org
import matplotlib.patches as mpatches
colors = {'England':'#e41a1c', 'Spain':'#377eb8', 'Italy':'#4daf4a', 'France':'#984ea3', 'Germany':'#ff7f00'}
color = []

e = mpatches.Patch(color='#e41a1c', label='England')
s = mpatches.Patch(color='#377eb8', label='Spain')
it = mpatches.Patch(color='#4daf4a', label='Italy') #Facepalm note to self : never use i as it'll be used as an iterable for a for loop
f = mpatches.Patch(color='#984ea3', label='France')
g = mpatches.Patch(color='#ff7f00', label='Germany')


# In[ ]:


#Analysing teams in each league
top_n = 15
top_goal_scorers = matches_w_teams.groupby('home_team_long_name').total_goals.sum().sort_values(ascending = False)

for i in range(top_n):
    color.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == top_goal_scorers.head(top_n).index[i]].country_name.values][0])

fig = plt.figure(figsize = (10,8))
ax = fig.add_subplot(1,1,1)
rects = ax.bar(range(top_n), top_goal_scorers.head(top_n).values,align = "center",color = color)

ax.set_xticks(range(top_n))
ax.set_xticklabels(top_goal_scorers.head(top_n).index,rotation = "vertical")
ax.set_title("Top goal scorers at home")
ax.set_ylabel("Number of goals")
ax.set_xlabel("Team name")

ax.legend([it,g,e,s,f], colors.keys())
autolabel(rects)


# Again, we see the dominance by the two big Spanish teams (Real Madrid and Barcelona) in scoring goals at home. The next competitor, Manchester City(490) is nearly a 100 goals behind FC Barcelona(596). And Real Madrid(652) leads the race ahead of its nearest competitor (and fierce rival), FC Barcelona, by 56 goals. 
# 
# Apart from this, we also notice that 5 of the top 15 teams are from Spain. England, Italy, Germany have three teams each and France has just the one big-spending club, Paris Saint-Germain in the top 15. 
# 
# It would be interesting to also look at the ratio of home wins to home losses for teams and compare it with this graph.

# In[ ]:


#We get teams' home win vs home loss ratio
team_home_win = matches_w_teams[matches_w_teams.result == 1].groupby('home_team_long_name').result.sum().sort_index()
team_home_loss = matches_w_teams[matches_w_teams.result ==  -1].groupby('home_team_long_name').count().result.sort_index()
team_home_draw_loss = matches_w_teams[matches_w_teams.result !=  1].groupby('home_team_long_name').count().result.sort_index()
team_home_draw_win =  matches_w_teams[matches_w_teams.result !=  -1].groupby('home_team_long_name').count().result.sort_index()
np.setdiff1d(team_home_loss.index,team_home_win.index)

#We notice that a team "SpVgg Greuther Furth" never won a home game in the one season it played in Germany. Remove it.
team_home_loss = team_home_loss[team_home_loss.index.str.contains("SpV") == False]
team_home_draw_win = team_home_draw_win[team_home_draw_win.index.str.contains("SpV") == False]

team_home_wl_ratio = team_home_win /team_home_loss
team_home_wl_ratio = team_home_wl_ratio.sort_values(ascending = False)
#print team_home_wl_ratio.head()

team_home_wld_ratio = team_home_win / team_home_draw_loss
team_home_wld_ratio = team_home_wld_ratio.sort_values(ascending = False)
#team_home_wld_ratio.head()

team_home_wdl_ratio = team_home_draw_win / team_home_loss
team_home_wdl_ratio = team_home_wdl_ratio.sort_values(ascending = False)


# In[ ]:


#Plotting top_n ratios
fig = plt.figure(figsize = (20,20))
plt.style.use('ggplot')

colorwl = []
colorwld = []
colorwdl = []
for i in range(top_n):
    colorwl.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == team_home_wl_ratio.head(top_n).index[i]].country_name.values][0])
    colorwld.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == team_home_wld_ratio.head(top_n).index[i]].country_name.values][0])
    colorwdl.append([colors[t] for t in matches_w_teams[matches_w_teams.home_team_long_name == team_home_wdl_ratio.head(top_n).index[i]].country_name.values][0])

gs = gridspec.GridSpec(2, 4)
ax1 = fig.add_subplot(gs[0, :2],)
rects1 = ax1.bar(range(top_n), team_home_wl_ratio.head(top_n).values,align = "center", color = colorwl)
ax1.set_xticks(range(top_n))
ax1.set_xticklabels(team_home_wl_ratio.head(top_n).index,rotation = "vertical")
ax1.set_title("Team home to loss ratio (without draws)")
ax1.set_ylabel("Win to Loss ratio")
ax1.set_xlabel("Team names")
ax1.legend([it,g,e,s,f], ["Italy","Germany","England","Spain","France"])

ax2 = fig.add_subplot(gs[0, 2:])
rects2 = ax2.bar(range(top_n), team_home_wld_ratio.head(top_n).values,align = "center", color = colorwld)
ax2.set_xticks(range(top_n))
ax2.set_xticklabels(team_home_wld_ratio.head(top_n).index,rotation = "vertical")
ax2.set_title("Team home to loss ratio (with draws considered as a loss)")
ax2.set_ylabel("Win to (Loss or Draw) ratio")
ax2.set_xlabel("Team names")
ax2.legend([it,g,e,s,f], ["Italy","Germany","England","Spain","France"])

ax3 = fig.add_subplot(gs[1,1:3])
rects3 = ax3.bar(range(top_n), team_home_wdl_ratio.head(top_n).values,align = "center", color = colorwdl)
ax3.set_xticks(range(top_n))
ax3.set_xticklabels(team_home_wdl_ratio.head(top_n).index,rotation = "vertical")
ax3.set_title("Team home to loss ratio (with draws considered as a loss)")
ax3.set_ylabel("(Win or draw) to loss ratio")
ax3.set_xlabel("Team names")
ax3.legend([it,g,e,s,f], ["Italy","Germany","England","Spain","France"])
plt.tight_layout()


# Keeping with the trends, the two giants enjoy a considerably large (11 to 14) home win to loss ratio. Even if we consider draws, FC Barcelona and Real Madrid win at least 5 times more games at home than lose or draw. Following these two is FC Bayern Munich (from Germany) which also wins 4 times more than lose or draw at home.
# 
# Other big teams in the top 15 also have significantly large (4+) home win to loss ratio. However, the home win to (loss or draw) ratio is slightly lower (1.5 to 4) for these teams. Also, we can see that if we carried out the analyses by considering a draw to the number of wins list as opposed to the loss list, we get a much higher ratio (FC Barcelona nearly equal to 16).
# 
# 
# 
# These factors indicate that the top teams enjoy a very high home advantage. We could probably look at trends in each season and check how select teams are performing on this home-advantage factor.
# 
# We should also analyse how other teams at the bottom of the table perform on these factors. 

# In[ ]:




