#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


matches = pd.read_csv("../input/matches.csv")
matches.head()


# In[ ]:


deliveries = pd.read_csv("../input/deliveries.csv")
deliveries.head()


# * Lets look at the players who have scored most 50s among all seasons.

# In[ ]:


aggregatedata = pd.merge(matches,deliveries,left_on = 'id',right_on = 'match_id')
fifty = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','season']).agg({'batsman_runs':'sum'}))
fifty = fifty[np.logical_and(fifty['batsman_runs'] >= 50,fifty['batsman_runs'] <= 100)]
fifty.reset_index(inplace = True)
fifty.head()


# In[ ]:


print('Total 50 scores in all the seasons :',len(fifty))


# In[ ]:


for groups in fifty.groupby(['season']):
    print('Fifties in season ',groups[0], ':',len(groups[1]['batsman_runs']))


# Now lets look at the players who has scored most fifties across all seasons.

# In[ ]:


fifty_season = pd.DataFrame(fifty.groupby(['batsman']).agg({'batsman_runs' : 'count'}))
fifty_season.reset_index(inplace = True)
fifty_season.columns = ['batsman','Total Fifties']
fifty_season = fifty_season.sort_values(by = 'Total Fifties',ascending = False)
fifty_season.head()


# In[ ]:


plt.rcParams['figure.figsize'] = 20,10
sns.barplot(x = 'batsman',y = 'Total Fifties',data = fifty_season.head(15),palette = reversed(sns.cubehelix_palette(15)))


# This tells us about the player who have scored most fifties across all seasons ( excluding the 100s.)

# Now lets look at how they have performed each season.

# In[ ]:


fifty_eseason = pd.DataFrame(fifty.groupby(['season','batsman']).agg({'batsman_runs' : 'count'}))
fifty_eseason.reset_index(inplace = True)
fifty_eseason.columns = ['Season','Batsman','Total Fifties']
fifty_eseason = fifty_eseason.sort_values(by = ['Season','Total Fifties'],ascending = False)
fifty_eseason.head()


# In[ ]:


colors = cm.rainbow(np.linspace(0,1,10))
for title,groups in fifty_eseason.groupby(['Season']):
    groups.head(10).plot(x ='Batsman',y = 'Total Fifties',kind = 'bar',color = colors)
    plt.title('Most fifties in season %s '%title)


# Scoring fifty does not matter always, sometimes it is important that it comes in a winning cause. So lets look at how many fifites came in a winning cause across all season.

# In[ ]:


fifty_win = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','season','winner','batting_team']).agg({'batsman_runs':'sum'}))
fifty_win = fifty[np.logical_and(fifty['batsman_runs'] >= 50,fifty['batsman_runs'] <= 100)]
fifty_win.reset_index(inplace = True)
fifty_win.head(10)


# In[ ]:


fifty_win = fifty_win[fifty_win['winner'] == fifty_win['batting_team']]
columns = ['batsman','season','batsman_runs']
fifty_win = fifty_win[columns]
fifty_win = pd.DataFrame(fifty_win.groupby(['batsman']).agg({'batsman_runs': 'count'}))
fifty_win.reset_index(inplace = True)
fifty_win.head(10)


# In[ ]:


fifty_win = pd.merge(fifty_win,fifty_season,left_on = 'batsman',right_on = 'batsman')
fifty_win.head()


# In[ ]:


fifty_win['Total 50s in Winning Cause'] = fifty_win['batsman_runs']/fifty_win['Total Fifties']*100


# In[ ]:


fifty_win.head()


# Now we have a list of the players and there 50s in winning cause. Let us see how many in total were in winning cause.

# In[ ]:


print('Total fifties in Winning Cause:',np.sum(fifty_win['batsman_runs']))
print('Total % of 50s in Winning Cause:',round(np.sum(fifty_win['batsman_runs'])/np.sum(fifty_win['Total Fifties'])*100,2))     


# So we see that 62% of the fifties were in winning causes.

# To get a substantial idea about each player impact while scoring 50s we will only take those players who has scored atleast 10 fifties.

# In[ ]:


fifty_win = fifty_win[fifty_win['Total Fifties'] >= 10]
fifty_win = fifty_win.sort_values(by = 'Total 50s in Winning Cause',ascending = False)
fifty_win.head(10)


# In[ ]:


sns.barplot(x = 'batsman',y = 'Total 50s in Winning Cause',data = fifty_win.head(10),palette = reversed(sns.color_palette("mako_r", 10)))


# So this are the players whose performance have great impact on their teams winning chances

# Now lets look at the players who have scored most 50s and still their team lost.

# In[ ]:


fifty_win['Total 50s in Losing Cause'] = 100 - fifty_win['Total 50s in Winning Cause']
fifty_win = fifty_win.sort_values(by = ['Total 50s in Losing Cause'],ascending = False)
sns.barplot(x = 'batsman',y = 'Total 50s in Losing Cause',data = fifty_win.head(10),palette = reversed(sns.color_palette("mako_r", 10)))


# This are the players who have the worst rate when it comes to impacting their teams winning chances with their 50s. Strangely Yuvraj Singh who has been such an impact player is at the top of the list. 

# Now only scoring 50s is not enough. Sometime the speed at which you score the 50 is equally important. So now lets look at the strike rate of each of this players when they scored 50s.

# In[ ]:


fifty_sr = pd.DataFrame(aggregatedata.groupby(['match_id','batsman']).agg({'batsman_runs':'sum','ball':'count'}))
fifty_sr = fifty_sr[np.logical_and(fifty_sr['batsman_runs'] >= 50,fifty_sr['batsman_runs'] <= 100)]
fifty_sr.reset_index(inplace = True)
fifty_sr.head()


# In[ ]:


fifty_sr = pd.DataFrame(fifty_sr.groupby(['batsman']).agg({'batsman_runs':'sum','ball' : 'sum'}))
fifty_sr = fifty_sr[fifty_sr['batsman_runs'] > 800] # Equivalent to saying they might have scored atleast 15 fifties
fifty_sr['Strike Rate'] = fifty_sr['batsman_runs']/fifty_sr['ball']*100
fifty_sr.reset_index(inplace = True)
fifty_sr = fifty_sr.sort_values(by = ['Strike Rate'],ascending = False)


# In[ ]:


sns.barplot(x = 'batsman',y = 'Strike Rate',data = fifty_sr.head(10),palette = "Blues_d")


# So we can see that most of the times when YK Pathan and MS Dhone scores fifty they scores it at a great rate almost around 180. It is not strange to see most of the players we have in the list are great strikers of the ball and that is actually part of their natural game.

# There are many more analysis that can be done on the 50s scored by players. We would end it by looking at which team has scored most fifties and which team has conceded most 50s.

# In[ ]:


fifty_scored = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','batting_team']).agg({'batsman_runs':'sum'}))
fifty_scored = fifty_scored[np.logical_and(fifty_scored['batsman_runs'] >= 50,fifty_scored['batsman_runs'] <= 100)]
fifty_scored.reset_index(inplace = True)
fifty_scored.head()


# In[ ]:


fifty_scored = pd.DataFrame(fifty_scored.groupby(['batting_team']).agg({'batsman_runs':'count'}))
fifty_scored.reset_index(inplace = True)
fifty_scored.columns = ['batting_team','Total Fifties']
fifty_scored = fifty_scored.sort_values(by = ['Total Fifties'],ascending = False)
fifty_scored


# So we can see the MI,RCB and CSK are the top 3 teams with most number of players scoring 50s. 

# In[ ]:


fifty_conceded = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','bowling_team']).agg({'batsman_runs':'sum'}))
fifty_conceded = fifty_conceded[np.logical_and(fifty_conceded['batsman_runs'] >= 50,fifty_conceded['batsman_runs'] <= 100)]
fifty_conceded.reset_index(inplace = True)
fifty_conceded.head()


# In[ ]:


fifty_conceded = pd.DataFrame(fifty_conceded.groupby(['bowling_team']).agg({'batsman_runs':'count'}))
fifty_conceded.reset_index(inplace = True)
fifty_conceded.columns = ['bowling_team','Total Fifties']
fifty_conceded = fifty_conceded.sort_values(by = ['Total Fifties'],ascending = False)
fifty_conceded


# And as we can see KXIP has  most 50s scored against them. MI and RCB who were in the top3 of most 50s scored are also in the top 3 here. This might have to do with the fact that they play on pitches that are 

# Now all of the analysis we have done on 50s can be done on 100s. It might not be that interesting since not many 100s have been scored , but it would still be worthwhile to look at it.

# Lets look at the players who have scored most 100s among all seasons.

# In[ ]:


hundred = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','season']).agg({'batsman_runs':'sum'}))
hundred = hundred[hundred['batsman_runs'] >= 100]
hundred.reset_index(inplace = True)
hundred.head()


# In[ ]:


print('Total 100s scores in all the seasons :',len(hundred))


# In[ ]:


for groups in hundred.groupby(['season']):
    print('Hundreds in season ',groups[0], ':',len(groups[1]['batsman_runs']))


# Now lets look at the players who has scored most fifties across all seasons.

# In[ ]:


hundred_season = pd.DataFrame(hundred.groupby(['batsman']).agg({'batsman_runs' : 'count'}))
hundred_season.reset_index(inplace = True)
hundred_season.columns = ['batsman','Total Hundred']
hundred_season = hundred_season.sort_values(by = 'Total Hundred',ascending = False)
#np.sum(hundred_season['Total Hundred'])
hundred_season.head()


# In[ ]:


plt.rcParams['figure.figsize'] = 20,10
sns.barplot(x = 'batsman',y = 'Total Hundred',data = hundred_season.head(15),palette = reversed(sns.cubehelix_palette(15)))


# This tells us about the player who have scored most centuries across all seasons

# In[ ]:


hundred_eseason = pd.DataFrame(hundred.groupby(['season','batsman']).agg({'batsman_runs' : 'count'}))
hundred_eseason.reset_index(inplace = True)
hundred_eseason.columns = ['Season','Batsman','Total Hundred']
hundred_eseason = hundred_eseason.sort_values(by = ['Season','Total Hundred'],ascending = False)
hundred_eseason.head()


# In[ ]:


colors = cm.rainbow(np.linspace(0,1,6))
for title,groups in hundred_eseason.groupby(['Season']):
    groups.head(10).plot(x ='Batsman',y = 'Total Hundred',kind = 'bar',color = colors)
    plt.title('Most hundreds in season %s '%title)


# Scoring hundred does not matter always, sometimes it is important that it comes in a winning cause. So lets look at how many hundred came in a winning cause across all season.

# In[ ]:


hundred_win = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','season','winner','batting_team']).agg({'batsman_runs':'sum'}))
hundred_win = hundred_win[hundred_win['batsman_runs'] >= 100]
hundred_win.reset_index(inplace = True)
hundred_win.head(10)


# In[ ]:


hundred_win = hundred_win[hundred_win['winner'] == hundred_win['batting_team']]
columns = ['batsman','season','batsman_runs']
hundred_win = hundred_win[columns]
hundred_win = pd.DataFrame(hundred_win.groupby(['batsman']).agg({'batsman_runs': 'count'}))
hundred_win.reset_index(inplace = True)
hundred_win.head(10)


# In[ ]:


hundred_win = pd.merge(hundred_win,hundred_season,left_on = 'batsman',right_on = 'batsman')
hundred_win.head()


# In[ ]:


hundred_win['Total 100s in Winning Cause'] = hundred_win['batsman_runs']/hundred_win['Total Hundred']*100
hundred_win.head()


# Now we have a list of the players and there 100s in winning cause. Let us see how many in total were in winning cause.

# In[ ]:


print('Total hundreds in Winning Cause:',np.sum(hundred_win['batsman_runs']))
print('Total % of 100s in Winning Cause:',round(np.sum(hundred_win['batsman_runs'])/53*100,2))     


# So we see that 79.25% of the fifties were in winning causes.

# In[ ]:


hundred_win = hundred_win.sort_values(by = 'Total 100s in Winning Cause',ascending = False)
hundred_win.head(10)


# In[ ]:


hundred_win['Total 100s in Losing Cause'] = 100 - hundred_win['Total 100s in Winning Cause']
hundred_win = hundred_win.sort_values(by = ['Total 100s in Losing Cause'],ascending = False)
sns.barplot(x = 'batsman',y = 'Total 100s in Losing Cause',data = hundred_win.head(2),palette = reversed(sns.color_palette("mako_r", 10)))


# So V Kohli and SR Watson has hundred in winning causes.

# Now only scoring 100s is not enough. Sometime the speed at which you score the 100 is equally important. So now lets look at the strike rate of each of this players when they scored 100s.

# In[ ]:


hundred_sr = pd.DataFrame(aggregatedata.groupby(['match_id','batsman']).agg({'batsman_runs':'sum','ball':'count'}))
hundred_sr = hundred_sr[hundred_sr['batsman_runs'] >= 100]
hundred_sr.reset_index(inplace = True)
hundred_sr.head()


# In[ ]:


hundred_sr = pd.DataFrame(hundred_sr.groupby(['batsman']).agg({'batsman_runs':'sum','ball' : 'sum'}))
hundred_sr['Strike Rate'] = hundred_sr['batsman_runs']/hundred_sr['ball']*100
hundred_sr.reset_index(inplace = True)
hundred_sr = hundred_sr.sort_values(by = ['Strike Rate'],ascending = False)


# In[ ]:


sns.barplot(x = 'batsman',y = 'Strike Rate',data = hundred_sr.head(10),palette = "Blues_d")


# So we have DA Miller and YK PAthan topping the list of scoring 100s with the fastest rate.

# We would end it by looking at which team has scored most hundred and which team has conceded most 100s.

# In[ ]:


hundred_scored = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','batting_team']).agg({'batsman_runs':'sum'}))
hundred_scored = hundred_scored[hundred_scored['batsman_runs'] >= 100]
hundred_scored.reset_index(inplace = True)
hundred_scored.head()


# In[ ]:


hundred_scored = pd.DataFrame(hundred_scored.groupby(['batting_team']).agg({'batsman_runs':'count'}))
hundred_scored.reset_index(inplace = True)
hundred_scored.columns = ['batting_team','Total Hundred']
hundred_scored = hundred_scored.sort_values(by = ['Total Hundred'],ascending = False)
hundred_scored


# So RCB has scored the most 100s as expected since they have the 3 best T20 players in their team named V Kohli, AB De Villiers, CH Gayle

# In[ ]:


hundred_conceded = pd.DataFrame(aggregatedata.groupby(['match_id','batsman','bowling_team']).agg({'batsman_runs':'sum'}))
hundred_conceded = hundred_conceded[hundred_conceded['batsman_runs'] >= 100]
hundred_conceded.reset_index(inplace = True)
hundred_conceded.head()


# In[ ]:


hundred_conceded = pd.DataFrame(hundred_conceded.groupby(['bowling_team']).agg({'batsman_runs':'count'}))
hundred_conceded.reset_index(inplace = True)
hundred_conceded.columns = ['bowling_team','Total Hundred']
hundred_conceded = hundred_conceded.sort_values(by = ['Total Hundred'],ascending = False)
hundred_conceded


# So KKR and GL has been on receiving side the most number of times

# In[ ]:




