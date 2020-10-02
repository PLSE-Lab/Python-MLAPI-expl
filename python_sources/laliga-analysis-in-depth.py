#!/usr/bin/env python
# coding: utf-8

# **            Data of 47 years of Spanish League.That's huge. **
# I will explore,analyse and visualize this data only for La Liga.I will leave out the data for the second division matches.
# 
# I will touch the following points of interest of this beautiful game.
# * The points table for the tournament.
# * Trend of winning in home ground and away grounds
# * Trend of goal scoring in home and away matches
# * Season wise laliga winners
# * The masters of winning matches and trophy

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# Let us start by reading the dataset in the name of laliga.
# Then we will filter only the first divison matches from the whole dataset and run our analysis.

# In[ ]:


laliga=pd.read_csv('../input/FMEL_Dataset.csv')
laliga=laliga[laliga.division==1]


# Let us view a sample of this data now.

# In[ ]:


laliga.head()


# If we go for the info of the dataset we can see there are 10 features for each of 16789 matches played in laliga in the given time. The dataset is not having any null values as all the columns are having same length(16789).

# In[ ]:


laliga.info()


# Next, We will create 3 new features in the dataset to see which team has won the match or the match ended as a tie.

# In[ ]:


laliga['local_team_won']=laliga.apply(lambda row: 1 if row['localGoals']>row['visitorGoals'] else 0,axis=1)
laliga['visitor_team_won']=laliga.apply(lambda row: 1 if row['localGoals']<row['visitorGoals'] else 0,axis=1)
laliga['draw']=laliga.apply(lambda row: 1 if row['localGoals']==row['visitorGoals'] else 0,axis=1)


# This time we will check the last 5 matches and see what the new features justify.

# In[ ]:


laliga.tail()


# COOL!!! Our new features say the local team won the first match, the visitor team won the second match and third match was a draw(in the order shown in above table).

# Now we will leave this dataset here and create a new one for further analysis.
# 
# I will name it as **points_table** .
# This will contain the information of each teams' performance season wise.
# 
# To do it let's take a copy of laliga dataset and extract features from this copy to **points_table** dataset.

# In[ ]:


df=laliga.copy()


# Next, we will calculate the number of wins,losses or draws each team/club has registered both on there home grounds and away arenas. For simplicity the results are stored in simple alphabetica variables. 

# In[ ]:


a=df.groupby(['season','localTeam'])['local_team_won'].sum().reset_index().rename(columns={'localTeam': 'club','local_team_won': 'won'})
b=df.groupby(['season','visitorTeam'])['visitor_team_won'].sum().reset_index().rename(columns={'visitorTeam': 'club','visitor_team_won': 'won'})
c=df.groupby(['season','localTeam'])['draw'].sum().reset_index().rename(columns={'localTeam': 'club','draw': 'draw'})
d=df.groupby(['season','visitorTeam'])['draw'].sum().reset_index().rename(columns={'visitorTeam': 'club','draw': 'draw'})
e=df.groupby(['season','localTeam'])['visitor_team_won'].sum().reset_index().rename(columns={'localTeam': 'club','visitor_team_won': 'lost'})
f=df.groupby(['season','visitorTeam'])['local_team_won'].sum().reset_index().rename(columns={'visitorTeam': 'club','local_team_won': 'lost'})


# We will create **points_table** with all these win ,loss and draw information and see how it looks.

# In[ ]:


point_table=a.merge(b,on=['season','club']).merge(c,on=['season','club']).merge(d,on=['season','club']).merge(e,on=['season','club']).merge(f,on=['season','club'])
point_table.head()


# Here the **won_x** feature stands for Win In HomeGround and **won_y** for Win In AwayGround. The same is for loss and draw. So we will rename the columns accordingly. Then calculate the total number of win ,loss and draw. Obviously the total number of matches played in the season by the team will be the sum of the aforesaid numbers.

# In[ ]:


point_table= point_table.rename(columns={'won_x':'home_win','won_y':'away_win','lost_x':'home_loss','lost_y':'away_loss'})
point_table['matches_won']=point_table.home_win+point_table.away_win
point_table['matches_lost']=point_table.home_loss+point_table.away_loss
point_table['matches_drawn']=point_table.draw_x+point_table.draw_y
point_table=point_table.drop(['draw_x','draw_y'],axis=1)
point_table['total_matches']=point_table.matches_won+point_table.matches_lost+point_table.matches_drawn


# It's time to award points for each team as per their performance in the matches. In laliga the winning team gets 3 points. for a decided match. Otherwise each team gets 1 point each if the match ends as a tie.

# In[ ]:


point_table['points']=(point_table.matches_won*3)+(point_table.matches_drawn*1)


# And this is how the points_table looks like after this.

# In[ ]:


point_table.tail()


# Going a little forward we will calculate the goals scored and conceded by each team throughout a season and also the goal difference.

# In[ ]:


g=df.groupby(['season','localTeam'])['localGoals'].sum().reset_index().rename(columns={'localTeam': 'club','localGoals': 'home_goals'})
h=df.groupby(['season','visitorTeam'])['visitorGoals'].sum().reset_index().rename(columns={'visitorTeam': 'club','visitorGoals': 'away_goals'})
i=df.groupby(['season','localTeam'])['visitorGoals'].sum().reset_index().rename(columns={'localTeam': 'club','visitorGoals': 'goals_conceded'})
j=df.groupby(['season','visitorTeam'])['localGoals'].sum().reset_index().rename(columns={'visitorTeam': 'club','localGoals': 'goals_conceded'})


# In[ ]:


point_table=point_table.merge(g,on=['season','club']).merge(h,on=['season','club']).merge(i,on=['season','club']).merge(j,on=['season','club'])


# In[ ]:


point_table['goals_scored']=point_table.home_goals+point_table.away_goals
point_table['goals_conceded']=point_table.goals_conceded_x+point_table.goals_conceded_y
point_table['goal_difference']=point_table.goals_scored-point_table.goals_conceded
point_table= point_table.drop(['goals_conceded_x','goals_conceded_y'],axis=1)


# We will sort the points table so as to get the team rankings in each of the seasons.

# In[ ]:


point_table= point_table.sort_values(by=['season','points','goal_difference']).reset_index().drop('index',axis=1)


# After this let's have look at the points table of the last la liga season(2016-17). Here Real Madrid stand out as the winners followed by Barcelona.

# In[ ]:


point_table.tail(n=20).sort_values('points',ascending=False)


# again taking a copy of this dataframe.

# In[ ]:


df=point_table.copy()


# Looking at the LaLiga winners for each of these years.

# In[ ]:


champs=df[df.groupby(['season'])['points'].transform(max)==df.points].reset_index()
champs.tail(12)


# For 2006-07 season we have 2 entries as both have equal points. So deleting the second entry on to head-to-head goals criteria.

# In[ ]:


champs=champs.drop(champs.index[[37]])
champs.tail(12)


# Counting the trophies each team has won in this time period.  Real Madrid are the leaders here. Plotting the trophy share as a pie chart.

# In[ ]:


champs.club.value_counts().reset_index()


# In[ ]:


champs.club.value_counts().plot(kind='pie', autopct='%2.1f%%', figsize=(7,7))


# Some bar graphs showing the dominance in the league with home wins, total wins.

# In[ ]:


df.groupby(['club'])['home_win'].sum().sort_values(ascending=False).head(20).plot(kind='bar',figsize=(20,8))


# In[ ]:


df.groupby(['club'])['matches_won'].sum().sort_values(ascending=False).head(20).plot(kind='bar',figsize=(20,8))


# This is a ranking of teams based on total goals scored. Two arch rivals are leading the way here.

# In[ ]:


df.groupby(['club'])['goals_scored'].sum().sort_values(ascending=False).head(20)


# In[ ]:


w=df.groupby(['club'])['home_goals'].sum().sort_values(ascending=False).head(20).reset_index()
x=df.groupby(['club'])['away_goals'].sum().sort_values(ascending=False).head(20).reset_index()
y=df.groupby(['club'])['goals_scored'].sum().sort_values(ascending=False).head(20).reset_index()
z=w.merge(x,on=['club']).merge(y,on=['club'])
a=df.groupby(['club'])['home_win'].sum().sort_values(ascending=False).head(20).reset_index()
b=df.groupby(['club'])['away_win'].sum().sort_values(ascending=False).head(20).reset_index()
c=df.groupby(['club'])['matches_won'].sum().sort_values(ascending=False).head(20).reset_index()
z=a.merge(b,on=['club']).merge(c,on=['club']).merge(z,on=['club'])


# Share of home goals and away goals for each team towards the total goals .

# In[ ]:


z.plot(x='club',y=['home_goals','away_goals','goals_scored'], kind="bar",figsize=(15,8))


# The more the number of goals scored, the more is the number of matches won. The following graph proves it.

# In[ ]:


sns.FacetGrid(z, hue="club", size=7).map(plt.scatter, "goals_scored", "matches_won").add_legend()


# Dominance of each team on their home ground and away ground with respect to total matches won.

# In[ ]:


z.plot(x='club',y=['home_win','away_win','matches_won'], kind="barh",figsize=(15,10))


# More analysis and visualization to come soon....
