#!/usr/bin/env python
# coding: utf-8

# # How has the nature of the NBA changed?

# **Conclusion: There has been an increasing number of threes in the past few years. PG are obvious but it also seems somewhat beneficial although not necessary for centers to also have the capability of shooting threes.**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# ## Importing Data

# In[ ]:


season_stats = pd.read_csv('../input/Seasons_Stats.csv')
season_stats.sort_values(by='Year',ascending=False).head()
season_stats.loc[season_stats.Player== 'Ivica Zubac']


# ## Side Track: Lebron James

# In[ ]:


# Small Curious side thing, how has Lebron's Threes been progressing
lebron_stats = season_stats.loc[season_stats.Player == 'LeBron James']
lebron_stats


# In[ ]:


columnsarr = ['FG','FGA','FG%','3P%','3PA','2P%','2PA','FT%','AST','STL','BLK','TOV']
columnsarr[1]

plt.figure(figsize=[20,20])
for s in range(len(columnsarr)):
    plt.subplot(3,4,s+1)
    plt.scatter(lebron_stats.Year,lebron_stats[columnsarr[s]])
    plt.title(columnsarr[s])


# It is interesting to note that Lebron isn't taking more shots.
# **His field goal percentage is going up because he is taking smarter shots and his attempts are going down.**

# ## NBA as a whole

# ### Compare complete number of each stat

# In[ ]:


sums_per_year = season_stats.groupby('Year').sum()

plt.figure(figsize=[20,20])
for s in range(len(columnsarr)):
    plt.subplot(3,4,s+1)
    plt.scatter(sums_per_year.index,sums_per_year[columnsarr[s]])
    plt.title(columnsarr[s])


# Every single stat is going up by a lot.  Lets first look at the scoring. 2PA seems to be plateuing while 3PA seems to be going up linearly.

# ### Three Points

# In[ ]:


season_stats = season_stats[np.isfinite(season_stats['3PA'])]

sums_per_year = season_stats.groupby('Year').sum()
mean_per_year = season_stats.groupby('Year').mean()
median_per_year = season_stats.groupby('Year').median()

plt.figure(figsize =[20,20])

#Total sum
plt.subplot(3,1,1)
threep = sums_per_year['3PA']
twop = sums_per_year['2PA']

plt.bar(sums_per_year.index,threep)
plt.bar(sums_per_year.index,twop,bottom=threep)
plt.legend(('3PA','2PA'))
plt.title('Sum of points')

#Mean
plt.subplot(3,1,2)
threep = mean_per_year['3PA']
twop = mean_per_year['2PA']

plt.bar(mean_per_year.index,threep)
plt.bar(mean_per_year.index,twop,bottom=threep)
plt.legend(('3PA','2PA'))
plt.title('Mean of points')

#Median
plt.subplot(3,1,3)
threep = median_per_year['3PA']
twop = median_per_year['2PA']

plt.bar(median_per_year.index,threep)
plt.bar(median_per_year.index,twop,bottom=threep)
plt.legend(('3PA','2PA'))
plt.title('Median of points')


# From the median plot, from **2012-2017 just recently the number of players shooting 3s have gone so much higher whereas 2PA have been going down.**

# In[ ]:


# Lets check the points distribution for each position
posarray = ['PG','SG','PF','SF','C']

plt.figure(figsize =[20,20])

for pos in range(len(posarray)):
    pos_stats = season_stats.loc[season_stats.Pos == posarray[pos]]
    
    sums_per_year = pos_stats.groupby('Year').sum()
    mean_per_year = pos_stats.groupby('Year').mean()
    median_per_year = pos_stats.groupby('Year').median()
    
    plt.subplot(5,3,pos*3+1)
    plt.bar(sums_per_year.index,sums_per_year['3PA'])
    plt.bar(sums_per_year.index,sums_per_year['2PA'],bottom=sums_per_year['3PA'])
    plt.title('Sum' + " " + posarray[pos])
    plt.subplot(5,3,pos*3+2)
    plt.bar(sums_per_year.index,mean_per_year['3PA'])
    plt.bar(sums_per_year.index,mean_per_year['2PA'],bottom=mean_per_year['3PA'])
    plt.title('Mean' + " " + posarray[pos])
    plt.subplot(5,3,pos*3+3)
    plt.bar(sums_per_year.index,median_per_year['3PA'])
    plt.bar(sums_per_year.index,median_per_year['2PA'],bottom=median_per_year['3PA'])
    plt.title('Median' + " " + posarray[pos])

#Note to self: I should learn how to put bar plots side by side later to make the std bars visible


# **First of all the fact that the total points have been rising or at least staying about the same while the median points have been going down suggests that there are more superpower players scoring points.
# Second I found that the median number of threes for centers have been consistently small while the sum has been going up. Does this mean the winning teams usually have centers that can shoot threes?**

# ### Importance of centers that can shoot threes

# In[ ]:


#It is hard to quantify how good a player is but we'll just use wins as a metric
center_stats = season_stats.loc[season_stats.Pos == 'C']
test_case = center_stats.sort_values('WS',ascending=False)[0:1000]


# In[ ]:


#Now lets see how these players shoot
test_case = test_case.groupby('Year').head(10).sort_values('Year')

sums_per_year = test_case.groupby('Year').sum()
mean_per_year = test_case.groupby('Year').mean()
median_per_year = test_case.groupby('Year').median()

plt.figure(figsize = (20,20))
plt.subplot(3,1,1)
plt.bar(sums_per_year.index,sums_per_year['3PA'])
plt.bar(sums_per_year.index,sums_per_year['2PA'],bottom=sums_per_year['3PA'])
plt.title('Sum' + " " + posarray[pos])

plt.subplot(3,1,2)
plt.bar(sums_per_year.index,mean_per_year['3PA'])
plt.bar(sums_per_year.index,mean_per_year['2PA'],bottom=mean_per_year['3PA'])
plt.title('Mean' + " " + posarray[pos])

plt.subplot(3,1,3)
plt.bar(sums_per_year.index,median_per_year['3PA'])
plt.bar(sums_per_year.index,median_per_year['2PA'],bottom=median_per_year['3PA'])
plt.title('Median' + " " + posarray[pos])


# Lets look at the past 2 years at how WS correlates with 3PA for centers

# In[ ]:


test = center_stats.loc[(center_stats.Year == 2015) | (center_stats.Year == 2016) | (center_stats.Year == 2017)]


# In[ ]:


plt.scatter(test.WS,test['3PA'])
plt.ylabel('3PA')
plt.xlabel('WS')
plt.title('WS vs 3PA')

plt.axvline(test.WS.median(),0,400)


# Right of the reference median line there seems to be much more centers that can shoot 3 points.

# In[ ]:


testshot = test.loc[test.WS > test.WS.median()]
shooters = testshot.loc[testshot['3PA'] > 150]
nonshooters = testshot.loc[testshot['3PA'] <= 150]


print(str(len(shooters)/len(testshot)) + " " + "are shooters above WS mark") 
print(str(len(nonshooters)/len(testshot)) + " " + "are nonshooters above WS mark") 

testshot = test.loc[test.WS < test.WS.median()]
shooters = testshot.loc[testshot['3PA'] > 150]
nonshooters = testshot.loc[testshot['3PA'] <= 150]

print(str(len(shooters)/len(testshot)) + " " + "are shooters below WS mark") 
print(str(len(nonshooters)/len(testshot)) + " " + "are nonshooters below WS mark") 


# **It seems like as much of 10% of centers with a WS higher than the median WS are shooters with a 3PA while for less than the median it is 1%.  While it may still be rare for center shooters and not completely necessary, it seems like a center that can shoot could at much value to a team.**

# # Part 2: Python Practice with Height and Weight

# In[ ]:


player_data = pd.read_csv('../input/player_data.csv')
players = pd.read_csv('../input/Players.csv')
season_stats = pd.read_csv('../input/Seasons_Stats.csv')
player_data.head()
players.head()


# In[ ]:


player_data.position = player_data.position.apply(lambda s: str(s).split('-')[0])
player_data.head()


# In[ ]:


year_players = season_stats.loc[season_stats.Year==1950.0].Player.unique()
year_stats = players.loc[players.Player.isin(year_players)]

positions = player_data.position.unique()
colors = dict(zip(positions,['Blue','Red','Green']))

for row in range(len(year_stats)):
    color = colors[player_data.loc[player_data.name == year_stats.loc[row].Player].position.values[0]]
    plt.scatter(year_stats.loc[row].height.item(),year_stats.loc[row].weight.item(),c=color)


# In[ ]:


#We have a problem, probably cause a null value in our data so lets take it out
player_data.position.isna().sum()
nullrow = player_data.loc[player_data.position.isnull()]
nullrow


# In[ ]:


player_data = pd.read_csv('../input/player_data.csv')
player_data = player_data.dropna(subset=['position'])
player_data.position = player_data.position.apply(lambda s: str(s).split('-')[0])
player_data.head()


# In[ ]:


#To make our legend after
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

year_players = season_stats.loc[season_stats.Year==1950.0].Player.unique()
year_stats = players.loc[players.Player.isin(year_players)]

positions = player_data.position.unique()
colors = dict(zip(positions,['Blue','Red','Green']))

player_ready = player_data['name'].unique()

#Sometimes players does not have a player from year_stats. IDK why cause the overview doesn't go over this
year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
year_stats = year_stats.reset_index()
################################DEBUGGING CODE#############################################
#for row in range(len(year_stats)):
    #if year_stats.Player[row] in player_ready:
#    color = colors[player_data.loc[player_data.name == year_stats.loc[row].Player].position.values[0]]
#    plt.scatter(year_stats.loc[row].height.item(),year_stats.loc[row].weight.item(),c=color)
        
#custom_lines = [plt.scatter(0,0, color='Blue'),plt.scatter(0,0, color='Green'),plt.scatter(0,0, color='Red')]
#plt.legend(custom_lines,['C','G','P'])

#year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
#year_stats = year_stats.reset_index()
#year_stats
###############################################################################################


# In[ ]:


#Plotting
plt.scatter(year_stats.height,year_stats.weight,c=year_stats.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]))
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')

red_patch = mpatches.Patch(color='red', label='Centers')
blue_patch = mpatches.Patch(color='blue', label='Forwards')
green_patch = mpatches.Patch(color='green', label='Guards')
plt.legend(handles=[red_patch, blue_patch, green_patch])


# In[ ]:


#Lets try to include season stats. Maybe WS could be a good metric. We average the WS of each player
metric = '2PA'
player_average = season_stats.groupby('Player')[metric].mean()
player_average_df = pd.DataFrame(player_average)
year_stats
year_stats_WS = year_stats.merge(player_average_df,left_on='Player',right_on='Player')
min(year_stats_WS[metric])
year_stats.WS = year_stats_WS[metric] + 5


# In[ ]:


#Plotting
plt.scatter(year_stats_WS.height,year_stats_WS.weight,c=year_stats_WS.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]),s=0.2*year_stats_WS[metric])
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')

red_patch = mpatches.Patch(color='red', label='Centers')
blue_patch = mpatches.Patch(color='blue', label='Forwards')
green_patch = mpatches.Patch(color='green', label='Guards')
plt.legend(handles=[red_patch, blue_patch, green_patch])


# In[ ]:


#Now for the final lets include a scroll for the year
import matplotlib.animation as animation
from matplotlib.widgets import Slider

year = 1950
year_players = season_stats.loc[season_stats.Year==year].Player.unique()
year_stats = players.loc[players.Player.isin(year_players)]

positions = player_data.position.unique()
colors = dict(zip(positions,['Blue','Red','Green']))

player_ready = player_data['name'].unique()

#Sometimes players does not have a player from year_stats. IDK why cause the overview doesn't go over this
year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
year_stats = year_stats.reset_index()
year_stats

#Size : Proportional to points**2
metric = 'PTS'
player_average = season_stats.groupby('Player')[metric].mean()
player_average_df = pd.DataFrame(player_average)
year_stats
year_stats_WS = year_stats.merge(player_average_df,left_on='Player',right_on='Player')
min(year_stats_WS[metric])
year_stats.WS = year_stats_WS[metric] + 5

#Plotting
plt.figure(figsize=(20,20))
plot = plt.scatter(year_stats_WS.height,year_stats_WS.weight,c=year_stats_WS.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]),s=0.001*(year_stats_WS[metric])**2)
plt.xlabel('height (cm)')
plt.ylabel('weight (kg)')

red_patch = mpatches.Patch(color='red', label='Centers')
blue_patch = mpatches.Patch(color='blue', label='Forwards')
green_patch = mpatches.Patch(color='green', label='Guards')
plt.legend(handles=[red_patch, blue_patch, green_patch])

#Add a slider for the year

#season_stats.head()


# In[ ]:


i=0 #For the subplots
plt.figure(figsize=(20,20))

for year in range(1950,2016,4):

    #year = 1950
    year_players = season_stats.loc[season_stats.Year==year].Player.unique()
    year_stats = players.loc[players.Player.isin(year_players)]

    positions = player_data.position.unique()
    colors = dict(zip(positions,['Blue','Red','Green']))

    player_ready = player_data['name'].unique()

    #Sometimes players does not have a player from year_stats. IDK why cause the overview doesn't go over this
    year_stats = year_stats.drop(year_stats[~year_stats.Player.isin(player_data.name)].index)
    year_stats = year_stats.reset_index()
    year_stats

    #Size : Proportional to points**2
    metric = 'PTS'
    player_average = season_stats.groupby('Player')[metric].mean()
    player_average_df = pd.DataFrame(player_average)
    year_stats
    year_stats_WS = year_stats.merge(player_average_df,left_on='Player',right_on='Player')
    min(year_stats_WS[metric])
    year_stats.WS = year_stats_WS[metric] + 5

    #Plotting
    #plt.figure(figsize=(20,20))
    i = i + 1
    plt.subplot(4,5,i)
    plot = plt.scatter(year_stats_WS.height,year_stats_WS.weight,c=year_stats_WS.Player.apply(lambda s: colors[player_data.loc[player_data.name == s].position.values[0]]),s=0.0001*(year_stats_WS[metric])**2)
    plt.xlabel('height (cm)')
    plt.ylabel('weight (kg)')
    plt.ylim((70,130))
    plt.xlim((170,220))
    plt.title(year)

    red_patch = mpatches.Patch(color='red', label='Centers')
    blue_patch = mpatches.Patch(color='blue', label='Forwards')
    green_patch = mpatches.Patch(color='green', label='Guards')
    plt.legend(handles=[red_patch, blue_patch, green_patch])


# **This seems to confirm that basketball is becoming a guard dominated game.**
