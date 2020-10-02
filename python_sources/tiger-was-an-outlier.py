#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# This data set contains weekly data on the top 1000 players in the official world golf rankings from September 2000 to April 2015. It has 3 columns, the ranking, name, and average points of the player. We can use the data to answer some interesting quesitons about the careers of individual players as well as about the statistical profile required to be ranked in a given position.
# 
# In this kernel I will address the anomalies in the data, a method clean these anomalies, and I will compare the careers of a number of top players during the time frame. 

# In[ ]:


owgr = pd.read_csv('../input/ogwr_historical.csv')
owgr['index'] = owgr.index // 1000
print(owgr.isna().sum())


# In[ ]:


#Utility functions
def exponential_fit(x, A, B, C):
    return A*np.exp(-B*x)+C

#Function to check if a list decreases monotonically
def check_monotonicity(L):
    return all(x>=y for x, y in zip(L, L[1:]))

def get_bad_values(L, idx):
    bools = [x>=y for x,y in zip(L,L[1:])]
    shift = idx * 1000
    bad_vals = []
    for i in range(len(bools)):
        if not bools[i]:
            bad_vals.append(shift+i)
    return bad_vals


# Start exploring data by plotting ranking vs. average number of ranking points

# In[ ]:


plt.plot(owgr['rank'], owgr['avg_points'], 'o', label='data')
plt.xlabel('Ranking')
plt.ylabel('Average OWGR Points')


# There is a funny looking blip around x=60, y=1, lets zoom in.

# In[ ]:


plt.plot(owgr['rank'], owgr['avg_points'], 'o', label='data')
plt.axis([50, 300, 0, 4])
plt.xlabel('Ranking')
plt.ylabel('Average OWGR Points')


# This bottom feature is strange likely comes from an error in the data.
# Hypothesis: Despite the data not having any missing values, there are detectable anomalies in the values
# The most straight forward thing to check is if the rankings are in numerical order, since a point-based ranking system should decrease monotonically with increasing position.
# First, we do a quick check on a single week of data to see if any values are far from thier expectation. We average the points before and after the point of interest, and calculate how far that point is from this average. A spike in the plot indicates a value may not be correct.

# In[ ]:


date_hyp = owgr[owgr['date'] == '05-06-05'] #Get data from the week of 05-06-05
test_hyp = []
for i in range(1, 999):
    ave = (date_hyp['avg_points'].iloc[i-1] + date_hyp['avg_points'].iloc[i+1]) / 2
    test_hyp.append(ave - date_hyp['avg_points'].iloc[i])
    #print(i, ave - date_tmp['avg_points'].iloc[i])
#plt.axis([0,100,-0.5,0.8])
plt.plot(test_hyp)
plt.arrow(x=150, y=-1.0, dx=-60, dy=0.5,
          width=0.05, head_width=0.2, head_length=10, shape='full', color='r')
plt.arrow(x=725, y=0.8, dx=0, dy=-0.5,
          width=0.05, head_width=0.2, head_length=0.1, shape='full', color='r')


# The large peaks at low x-values are from larger variation in average points from player to player. The peaks mark with red arrows are anomalies in the data, where a value is far from its expected value.
# 
# Below we will find all values where the average points are not in numerical order and drop them from our data

# In[ ]:


# Find all the bad average point values based on if rankings are numerically ordered
bad_vals = []
for i in range(len(owgr.date.unique())):
    date_tmp = owgr[owgr['index'] == i]
    mono = check_monotonicity(date_tmp['avg_points'])
    if not mono:
        bad_vals_slice = get_bad_values(date_tmp['avg_points'], i)
        [bad_vals.append(i) for i in bad_vals_slice]
print(len(bad_vals))


# In[ ]:


owgr_clean = owgr.drop(owgr.index[bad_vals])


# In[ ]:


# Compare clean and original data
plt.plot(owgr['rank'], owgr['avg_points'], 'bo', label='original')
plt.plot(owgr_clean['rank'], owgr_clean['avg_points'], 'g.', label='clean')
plt.axis([50, 400, 0, 4])
plt.xlabel('Ranking')
plt.ylabel('Average OWGR Points')
plt.legend()


# Now with clean data, lets look at the point distribtuions for the top ten players in the world.

# In[ ]:


# Violin plot for distribtuion of OWGR points for each fo top ten players (w/ Tiger)
top_ten = owgr_clean[owgr_clean['rank'] <= 10]
sns.violinplot(x='rank', y='avg_points', data=top_ten, inner=None)


# There is huge variance in the number 1 spot (largely due to Tiger Woods, analysis below). Removing tiger from dataset provides the following distibtutions. The top players still have largest variance, but it decreases with rank.

# In[ ]:


# Violin plot for distribution of OWGR points for each of top ten players (w/o Tiger)
top_ten = owgr_clean[owgr_clean['rank'] <= 10]
top_ten = top_ten[top_ten['name'] != 'TigerWoods']
sns.violinplot(x='rank', y='avg_points', data=top_ten, inner=None)


# How dominant was Tiger? 
# 
# Below we build a new dataframe with career statistics of each player who reached the number 1 ranked player in the world

# In[ ]:


#Build dataframe with info about players who reached the number 1 ranking
top_players = owgr_clean[owgr_clean['rank'] == 1]['name'].unique()

top_players_dict = {}
for i in top_players:
    player_data = []
    player_data.append(i)
    player_df = owgr_clean[owgr_clean['name'] == i]
    player_data.append(player_df['avg_points'].mean()) #Average points for career
    player_data.append(player_df['rank'].mean())
    player_data.append(player_df.count()['rank']) #Total weeks
    player_data.append(player_df[player_df['rank'] == 1].count()['rank']) #Total weeks at 1
    top_players_dict[i] = player_data
top_players_df = pd.DataFrame.from_dict(top_players_dict, orient='index',
                                        columns=['player', 'ave_points_career', 'ave_rank', 'total_weeks', 'total_weeks_at_1'])
top_players_df['per_weeks_at_1'] = 100 * (top_players_df['total_weeks_at_1'] / top_players_df['total_weeks'])
print(top_players_df.head())


# In[ ]:


fig, ax = plt.subplots()
ax.scatter(top_players_df['player'], top_players_df['per_weeks_at_1'], 
           color='k', s=30)
ax.set_ylabel('Percent at #1 (%)', color='k')
ax.tick_params(axis='y', labelcolor='k')
plt.xticks(rotation=60)
ax2 =ax.twinx()
ax2.set_ylabel('Average World Ranking', color='b')
ax2.tick_params(axis='y', labelcolor='b')
ax2.scatter(top_players_df['player'], top_players_df['ave_rank'],
            color='b', s=30)


# Tiger was the number 1 player in the world for over 70% of the time between September 2000 and April 2015. In the top_players_df I have compiled some other stats that can be used to tell a similar story, modify plotting to explore thos values.
# 
# The below plots show distibutions of the average number of points for the #1 player in the world. Tiger and everyone else are separated. Not only was Tiger #1 the majority of the time, he acculated more average world ranking points than the other number 1 players, reaching values more than double what anyone else has ever reached.

# In[ ]:


tw = owgr[(owgr['rank'] == 1) & (owgr['name'] == 'TigerWoods')]
notw = owgr[(owgr['rank'] == 1) & (owgr['name'] != 'TigerWoods')]
sns.kdeplot(tw['avg_points'], shade=True, label='Tiger')
sns.kdeplot(notw['avg_points'], shade=True, label='Everyone Else')
plt.xlabel('OWGR Points')
plt.ylabel('Frequency')
plt.title('Average World Ranking Points of #1 Player')


# Finally, lets compare the career trajectories of multiple players. Below shows the week-by-week average world ranking points for tiger, phil, and rory. X-axis is the week enumerated from 1 to 758.

# In[ ]:


# Tiger vs. Phil vs. Rory over time
tw = owgr_clean[owgr_clean['name'] == 'TigerWoods']
pm = owgr_clean[owgr_clean['name'] == 'PhilMickelson']
rm = owgr_clean[owgr_clean['name'] == 'RoryMcIlroy']

ax = tw.plot(x='index', y='avg_points', color='Red', label='Tiger')
pm.plot(x='index', y='avg_points', color='Green', label='Phil', ax=ax)
rm.plot(x='index', y='avg_points', color='Blue', label='Rory', ax=ax)
plt.ylabel('OWGR Points')
plt.xlabel('Week')


# Many other insights can be taken from this relatively simple data set. A couple examples include, quantifying the level of parity in the game or identifying a given players prime/peak years.
