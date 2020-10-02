#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.set_option('display.max_columns', 100)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# # Premier League Exploratory Data Analysis Part 1 

# ## Table of Contents
# 1. [Import Libraries](#import_libraries)
# 2. [Disecting Dataset](#disect_dataset)
# 3. [Win/Loss/Draw Analysis](#winlossdraw)
# 4. [Goals Scored Analysis](#goalscored)
# 5. [Premier League Table Overview](#pmoverview)

# <a id="import_libraries"></a>
# ### Import Libraries
# ***

# Lets import all the libraries we will need for the upcoming analysis

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display, HTML
get_ipython().run_line_magic('matplotlib', 'inline')


# <a id="disect_dataset"></a>
# ### Disecting Dataset
# ***

# Lets import data into notebook

# In[ ]:


results_file = '/kaggle/input/premier-league/results.csv'
stats_file = '/kaggle/input/premier-league/stats.csv'
results_csv = pd.read_csv(results_file)
stats_csv = pd.read_csv(stats_file)


# This kernel will be exporing results.csv. We can take a look at the top 5 observations in the csv file

# In[ ]:


results_csv.head()


# We can also take a look at the distribution of numeric and categorical variables

# In[ ]:


results_csv.info()


# No missing observation in the dataset. Let's take a closer look at the features.

# In[ ]:


display(results_csv.select_dtypes(include=['object']).describe())
for colname in results_csv.select_dtypes(include=['object']).columns:
    display(pd.DataFrame(results_csv.select_dtypes(include=['object']).loc[:,colname].value_counts()))
# print(75*'*')
# for i in results_csv.loc[:,'result'].value_counts():
#     print("{:.3f}".format(i/results_csv.loc[:,'result'].value_counts().sum()))


# Just from looking at the categorical data, we can see that the only teams that were not relegated in the premier league from the 2006-2007 season to 2017-2018 season were Manchester City, Manchester United, Chelsea, Everton, Liverpool, Tottenham Hotspur and Arsenal.
# We can also see that on average, there is a 46% of winning if you are the home team, 28% chance of winning if you are the away team and 26% chance of drawing the game. 

# Lets take a closer look at the numberical features

# In[ ]:


results_csv.select_dtypes(include=['float64']).describe()


# We can see that the home team has a higher mean goal per game value. A better way of showing this would be to use a boxplot

# In[ ]:


fig,ax6 = plt.subplots(1,1)
fig.set_size_inches(10,7)
sns.boxplot(data=results_csv.select_dtypes(include=['float64']), ax=ax6,showmeans=True)
ax6.set_title("Goal Distribution")
ax6.set_xticklabels(['Home Team',"Away Team"])
ax6.set_ylabel('Avg Goal/Game')
plt.show()


# What immediately jumps to me is that the 25th and 50th percentile for the distribution of the home goals are the same which means that many of the games ended with the home team scoring only one goal. There are also quite a few outliers on the high end for the distribution of the goals scored by the home and away team so the distribution will be skewed. This can also be observed in the difference between the mean and the median values. A bar plot should confirm this. 

# In[ ]:


fig,ax7 = plt.subplots(1,1)
fig.set_size_inches(10,7)
sns.distplot(results_csv.select_dtypes(include=['float64']).loc[:,'home_goals'], ax=ax7,label='Home Goals',hist=True,            kde = False,bins = 10)
sns.distplot(results_csv.select_dtypes(include=['float64']).loc[:,'away_goals'], ax=ax7,label='Away Goals',hist=True,            kde = False, bins = 8)
ax7.set_xlabel("Average Goal/Game")
ax7.set_title("Distribution of Goal/Game")
plt.legend()
plt.show()


# In[ ]:


temp = results_csv.select_dtypes(include=['float64']).loc[:,'home_goals'].value_counts().reset_index()
temp1 = results_csv.select_dtypes(include=['float64']).loc[:,'away_goals'].value_counts().reset_index()
# temp['home_goals_percentage']
temp['home_goals_percentage'] = temp.loc[:,['home_goals']].apply(lambda x: x/temp.home_goals.sum())
temp1['away_goals_percentage'] = temp1.loc[:,['away_goals']].apply(lambda x: x/temp1.away_goals.sum())
fig,ax8 = plt.subplots(2,1)
fig.set_size_inches(10,7)
sns.barplot(data = temp, x = 'index',y='home_goals_percentage',ax=ax8[0])
ax8[0].set_xlabel('Goals')
ax8[0].set_ylabel('')
ax8[0].set_title("Number of Goals Per Game @ Home")
sns.barplot(data = temp1, x = 'index',y='away_goals_percentage',ax=ax8[1])
ax8[1].set_xlabel('Goals')
ax8[1].set_ylabel('')
ax8[1].set_title("Number of Goals Per Game Away")
plt.tight_layout()


# In[ ]:


results_csv_result = pd.get_dummies(results_csv.loc[:,['result']]).iloc[:,[0,2]]
results_csv_result.head()


# In[ ]:


# results_csv = pd.concat([results_csv.drop(['result'],axis=1),results_csv_result],axis=1)
# results_csv.head()


# In[ ]:


results_csv_plt = results_csv.loc[:,['home_team','result','season','home_goals']].groupby(['home_team','result','season']).count().reset_index()


# In[ ]:


season_lsts = results_csv_plt.season.value_counts().index.tolist()
season_lsts.sort()
season_lsts


# <a id="winlossdraw"></a>
# ### Win/Loss/Draw Analysis
# ***

# Now lets begin to look into the number of wins/losses/draw for the home and away teams, per season

# #### Home Team Record
# 

# In[ ]:


home_result = results_csv.loc[:,['home_team','result','season','home_goals']].groupby(['home_team','result','season']).count().sort_values(by=['home_team','season','home_goals']).reset_index()
home_result.loc[:,'result'] = home_result.loc[:,'result'].apply(lambda x: 'W' if x == 'H' else ('L' if x == 'A' else 'D'))
home_result.columns = home_result.columns[0:3].tolist() + ['NumOfGames']
home_result.head()


# In[ ]:


fig, ax1 = plt.subplots(12,1)
fig.set_size_inches(15,100)
for season_lst,i in zip(season_lsts,range(12)):
        home_result.loc[home_result.season == season_lst,['home_team','result','NumOfGames']].set_index('home_team').        pivot_table(values = 'NumOfGames',index = ['home_team'],columns = ['result']).sort_index(ascending = False).        plot(kind = 'barh',stacked = True,ax = ax1[i])
        ax1[i].set_title('Team record at home in {} season'.format(season_lst))
        ax1[i].set_xlabel('Home Team Wins/Losses/Draws in {} season'.format(season_lst))
        ax1[i].set_ylabel('')


# In[ ]:


# fig, ax1 = plt.subplots(12,1)
# fig.set_size_inches(15,150)
# for season_lst,i in zip(season_lsts,range(12)):
#         sns.barplot(data=home_result.loc[home_result.season==season_lst,home_result.columns.to_list()],\
#             x='NumOfGames',y='home_team',hue='result',ax=ax1[i])
#         ax1[i].set_title('Team record at home in {} season'.format(season_lst))
#         ax1[i].set_xlabel('Home Team Wins/Losses/Draws in {} season'.format(season_lst))
#         ax1[i].set_ylabel('')


# In[ ]:


away_result = results_csv.loc[:,['away_team','result','season','away_goals']].groupby(['away_team','result','season']).count().sort_values(by=['away_team','season','away_goals']).reset_index()
away_result.loc[:,'result'] = away_result.loc[:,'result'].apply(lambda x: 'W' if x == 'A' else ('L' if x == 'H' else 'D'))
away_result.columns = away_result.columns[0:3].tolist() + ['NumOfGames']
away_result.head()


# In[ ]:


# fig, ax2 = plt.subplots(12,1)
# fig.set_size_inches(15,150)
# for season_lst,i in zip(season_lsts,range(12)):
#         sns.barplot(data=away_result.loc[away_result.season==season_lst,away_result.columns.to_list()],\
#             x='NumOfGames',y='away_team',hue='result',ax=ax2[i],dodge = True)
#         ax2[i].set_title('Team record away in {} season'.format(season_lst))
#         ax2[i].set_xlabel('Away Team Wins/Losses/Draws in {} season'.format(season_lst))
#         ax2[i].set_ylabel('')


# #### Away Team Record

# In[ ]:


fig, ax2 = plt.subplots(12,1)
fig.set_size_inches(15,100)
for season_lst,i in zip(season_lsts,range(12)):
        away_result.loc[away_result.season == season_lst,['away_team','result','NumOfGames']].set_index('away_team').        pivot_table(values = 'NumOfGames',index = ['away_team'],columns = ['result']).sort_index(ascending = False).        plot(kind = 'barh',stacked = True,ax = ax2[i])
        ax2[i].set_title('Team record away in {} season'.format(season_lst))
        ax2[i].set_xlabel('Away Team Wins/Losses/Draws in {} season'.format(season_lst))
        ax2[i].set_ylabel('')


# As indicated earlier, and as can be seen in the barplots above, the top teams have a very good record at home and not too many losses away from home. A very good indicator for a team that will be relegated, is their record away from home. 
# Let us now extract the teams that were relegated and promoted each season

# In[ ]:


for j in range(1,len(season_lsts)):
    tmp = results_csv.loc[:,['home_team','season','result']].groupby(['season','home_team']).count()
    relegated_teams = set(tmp.loc[season_lsts[j-1],:].index.tolist())-set(tmp.loc[season_lsts[j],:].index.tolist())
    promoted_teams = set(tmp.loc[season_lsts[j],:].index.tolist())-set(tmp.loc[season_lsts[j-1],:].index.tolist())
    teams_tbl = pd.DataFrame({'Relegated Team in '+season_lsts[j-1]:list(relegated_teams),'Promoted Teams in '+season_lsts[j]:list(promoted_teams)})
    display(teams_tbl)
    #     print("Relegated teams in {} season are: {}".format(season_lsts[j-1],relegated_teams))
#     print("Promoted teams in {} season are: {}".format(season_lsts[j],promoted_teams))


# <a id="goalscored"></a>
# ### Goals Scored Analysis
# ***

# Lets now take a look at the number of goals scored by each team at home and away from home

# #### Home Team Goal Scored

# In[ ]:


result_csv_goals_hometeam =results_csv.loc[:,['home_team','home_goals','season']].groupby(['season','home_team']).sum().reset_index().sort_values(by=['season','home_goals'],ascending=[1,0])


# In[ ]:


fig, ax3 = plt.subplots(len(season_lsts),1)
fig.set_size_inches(15,100)
for i in range(len(season_lsts)):
    sns.barplot(data = result_csv_goals_hometeam.loc[result_csv_goals_hometeam.season == season_lsts[i],:],                x='home_goals',y='home_team',ax=ax3[i])
    ax3[i].set_title('Total Goals at home during {} season'.format(season_lsts[i]))
    ax3[i].set_xlabel("Total Goals")
    ax3[i].set_ylabel("Home Team")


# In[ ]:


result_csv_goals_awayteam =results_csv.loc[:,['away_team','away_goals','season']].groupby(['season','away_team']).sum().reset_index().sort_values(by=['season','away_goals'],ascending=[1,0])


# #### Away Team Goal Scored

# In[ ]:


fig, ax4 = plt.subplots(len(season_lsts),1)
fig.set_size_inches(15,100)
for i in range(len(season_lsts)):
    sns.barplot(data = result_csv_goals_awayteam.loc[result_csv_goals_awayteam.season == season_lsts[i],:],                x='away_goals',y='away_team',ax=ax4[i])
    ax4[i].set_title('Total Goals away during {} season'.format(season_lsts[i]))
    ax4[i].set_xlabel("Total Goals")
    ax4[i].set_ylabel("Away Team")


# Something to notice here is that it is not enough to score alot of goals at home. If you look at the 2006-2007 season, Sheffield United was relegated. They scored a decent amount of goals at home but were last when it came to scoring goals way from home. A better parameter to look at would be goal difference

# In[ ]:


results_csv_goaldiffhome = results_csv.loc[:,['season','home_team','home_goals','away_goals']].groupby(['season','home_team']).sum().apply(lambda x: x[0]-x[1],axis=1).reset_index()
results_csv_goaldiffhome.columns = [results_csv_goaldiffhome.columns[0],'TeamName','GoalDiffHome']
results_csv_goaldiffhome.head()


# In[ ]:


results_csv_goaldiffaway = results_csv.loc[:,['season','away_team','away_goals','home_goals']].groupby(['season','away_team']).sum().apply(lambda x: x[0]-x[1],axis=1).reset_index()
results_csv_goaldiffaway.columns = [results_csv_goaldiffaway.columns[0],'TeamName','GoalDiffAway']
results_csv_goaldiffaway.head()


# In[ ]:


results_csv_goaldiff = pd.merge(results_csv_goaldiffhome, results_csv_goaldiffaway, on=['season','TeamName'])
results_csv_goaldiff['GoalDiff'] = results_csv_goaldiff.apply(lambda x: x[2]+x[3], axis=1)
results_csv_goaldiff.head()


# In[ ]:


fig,ax10 = plt.subplots(1,1)
fig.set_size_inches(15,10)
sns.heatmap(results_csv_goaldiff.pivot('TeamName','season','GoalDiff'),ax=ax10,           annot=True,linewidths=1,fmt='.0f')
ax10.set_xticklabels(ax10.get_xticklabels(),rotation=45)
ax10.set_ylabel('Team')
plt.show()


# As I figured, if you take a look at the teams that were relegated and their GD, you can see a high correlation but GD cannot be the only factor used to determine if a team will be relegated. Birmingham City in the 2007-2008 season had a better goal differencial  than Sunderland and were still relegated. 

# <a id="pmoverview"></a>
# ### Premier League Table Overview
# ***

# Now lets take a look at the premier league table from the 2006-2007 season to 2017-2018 season

# In[ ]:


results_csv['Home_Team_Points'] = results_csv['result'].apply(lambda x: 3 if x=='H' else (1 if x == 'D' else 0))
results_csv['Away_Team_Points'] = results_csv['result'].apply(lambda x: 3 if x=='A' else (1 if x == 'D' else 0))
results_csv.head()


# In[ ]:


results_csv_home = results_csv.loc[:,['home_team','season','Home_Team_Points']].groupby(['season','home_team']).sum().reset_index()
results_csv_away = results_csv.loc[:,['away_team','season','Away_Team_Points']].groupby(['season','away_team']).sum().reset_index()


# In[ ]:


results_csv_total_points = results_csv_home.merge(results_csv_away,left_on=['season','home_team'],right_on = ['season','away_team'])


# In[ ]:


results_csv_total_points["total_points"] = results_csv_total_points.loc[:,['Home_Team_Points','Away_Team_Points']].apply(      np.sum,axis=1)
results_csv_total_points.head()


# In[ ]:


fig,ax9 = plt.subplots(1,1)
fig.set_size_inches(15,10)
sns.heatmap(results_csv_total_points.pivot('home_team','season','total_points'),ax=ax9,           annot=True,linewidths=1,fmt='.0f')
ax9.set_xticklabels(ax9.get_xticklabels(),rotation=45)
ax9.set_ylabel('')
plt.show()


# What immediately jumps out to me is the number of points Derby County accumulated in the 2007-2008 season and the number of points Aston Villa accumulated during the 2015-2016 season. Quite abysmal. 
# What is also really cool to see is the sudden drop off in points for Manchester United after the 2012-2013 season which co-incides with the season where Alex Ferguson retired.
# 
# Next we'll be reviewing stats.csv
# 
# Thanks for reading! If you have any comments on how I can improve the notebook, definitely let me know! 
