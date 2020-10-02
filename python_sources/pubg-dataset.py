#!/usr/bin/env python
# coding: utf-8

# Hi all,This is beginners work.. Feel free to upvote it if u like ... thank you 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


df= pd.read_csv("../input/pubg-finish-placement-prediction/sample_submission_V2.csv")
test= pd.read_csv("../input/pubg-finish-placement-prediction/test_V2.csv")
train= pd.read_csv("../input/pubg-finish-placement-prediction/train_V2.csv")


# * let us go through the data once..

# In[ ]:


train.head()


# In[ ]:


train.shape


# In[ ]:


train.info()


# FOR BETTER UNDERSTANING OF THE DATASET LET US GO THROUGH THE COLUMNS IN THE DATASET..

# * groupId - Players team ID
# * matchId - Match ID
# * assists - Number of assisted kills. The killed is actually scored for the another teammate.
# * boosts - Number of boost items used by a player. These are for example: energy dring, painkillers, adrenaline syringe.
# * damageDealt - Damage dealt to the enemy
# * DBNOs - Down But No Out - when you lose all your HP but you're not killed yet. All you can do is only to crawl.
# * headshotKills - Number of enemies killed with a headshot
# * heals - Number of healing items used by a player. These are for example: bandages, first-aid kits
# * killPlace - Ranking in a match based on kills.
# * killPoints - Ranking in a match based on kills points.
# * kills - Number of enemy players killed.
# * killStreaks - Max number of enemy players killed in a short amount of time.
# * longestKill - Longest distance between player and killed enemy.
# * matchDuration - Duration of a mach in seconds.
# * matchType - Type of match. There are three main modes: Solo, Duo or Squad. In this dataset however we have much more categories.
# * maxPlace - The worst place we in the match.
# * numGroups - Number of groups (teams) in the match.
# * revives - Number of times this player revived teammates.
# * rideDistance - Total distance traveled in vehicles measured in meters.
# * roadKills - Number of kills from a car, bike, boat, etc.
# * swimDistance - Total distance traveled by swimming (in meters).
# * teamKills - Number teammate kills (due to friendly fire).
# * vehicleDestroys - Number of vehicles destroyed.
# * walkDistance - Total distance traveled on foot measured (in meters).
# * weaponsAcquired - Number of weapons picked up.
# * winPoints - Ranking in a match based on won matches.

# **our target variable is winPlacePerc**

# lets create some basic descreptive statistics for each column. These will be usefull to set the visualisation parameters, to filter out the outliers and to get the feeling about the ranges and scales.

# In[ ]:


train.describe()


# now, lets check if there is any missing data

# In[ ]:


missing_data=train.isna().sum()
missing_data.columns=['Missing data']


# there are no missing values in the dataset.So lets start doing EDA.

# EXPLORATORY DATA ANALYSIS

# In[ ]:


m_types = train.loc[:,"matchType"].value_counts().to_frame().reset_index()
m_types.columns = ["Type","Count"]
m_types


# In[ ]:


number_of_matches = train.loc[:,"matchId"].nunique()
number_of_matches


# * There are 47965 matches in our dataset

# In PUBG there are essentially three main modes of game: Solo, Duo and Squad.
# 
# In a squad mode, you play in a group of 4 players. Here we can see that the match types are further broken down taking into account view modes:
# 
#     -FPP - First Person Perspective
#     -TPP - Thirst Peron Perspective

# In[ ]:


plt.figure(figsize=(15,8))
ticks = m_types.Type.values
ax = sns.barplot(x="Type", y="Count", data=m_types)
ax.set_xticklabels(ticks, rotation=60, fontsize=14) # this helps us in rotating the font.
ax.set_title("Match types")
plt.show()


# - The graph above shows that the most popular game modes are squad and duo. Next I will aggregate all these individual types into three main categories (squad, duo and solo).

# In[ ]:


m_types2 = train.loc[:,"matchType"].value_counts().to_frame()
aggregated_squads = m_types2.loc[["squad-fpp","squad","normal-squad-fpp","normal-squad"],"matchType"].sum()
aggregated_duos = m_types2.loc[["duo-fpp","duo","normal-duo-fpp","normal-duo"],"matchType"].sum()
aggregated_solo = m_types2.loc[["solo-fpp","solo","normal-solo-fpp","normal-solo"],"matchType"].sum()
aggregated_mt = pd.DataFrame([aggregated_squads,aggregated_duos,aggregated_solo], index=["squad","duo","solo"], columns =["count"])
aggregated_mt


# In[ ]:


fig1, ax1 = plt.subplots(figsize=(5, 5))
labels = ['squad', 'duo', 'solo']
wedges, texts, autotexts = ax1.pie(aggregated_mt["count"],textprops=dict(color="w"), autopct='%1.1f%%')
ax1.legend(wedges, labels,title="Types",loc="center left",bbox_to_anchor=(1, 0, 0.5, 1))
plt.setp(autotexts, size=12, weight="bold")
plt.show()


# the above pie chart shows that 54% of all the matches were played in squad mode

# In[ ]:


plt.figure(figsize=(15,8))
Aa=sns.distplot(train['numGroups'])
plt.title('Number of groups')
plt.show()


# The graph allows to clearly notice distribution three spikes referring to squad games, duo games and solo games.

# In[ ]:


plt.figure(figsize=(15,8))
Az=sns.boxplot(x='kills',y='damageDealt',data=train)
plt.title('Damage Dealth VS Number of kills')
plt.show()


# There is an obvious correlation between number of kills and damage dealt. We see also that there are some outliers, more in the lower range. As the number of kills increases number of outliers reduces. The maximum kills is 72 which is much bigger than the wast majority of players scores.

# In[ ]:


train[train['kills']>60][["Id","assists","damageDealt","headshotKills","kills","longestKill"]]


# In[ ]:


headshots = train[train['headshotKills']>0]
plt.figure(figsize=(15,5))
sns.countplot(headshots['headshotKills'].sort_values())
print("Maximum number of headshots that the player scored: " + str(train["headshotKills"].max()))


# DBNO - Down But Not Out. How many enemies DBNOs an average player scores.

# In[ ]:


headshots = train[train['DBNOs']>0]
plt.figure(figsize=(15,5))
sns.countplot(headshots['DBNOs'].sort_values())
print("Mean number of DBNOs that the player scored: " + str(train["DBNOs"].mean()))


# now lets try to find if there is any correlation between kills and DBNOs

# In[ ]:


plt.figure(figsize=(15,8))
ax2 = sns.boxplot(x="DBNOs",y="kills", data = train)
ax2.set_title("Number of DBNOs vs. Number of Kills")
plt.show()


# It seems that DBNOs are correlated with kills. That makes sense as usually if player is not killed by headshoot yu have to finish him while he's in DBNO state. Interesting is the first observation in the plot - apparently there is a number of players who scored a kill without DBNOs - this is usually a headshot or a vechicle explosion.

# MAXIMUM DISTANCE

# TO GIVE A CLEAR EXAMPLE ABOUT THE DISTANCE IN THHIS GAME ..

# ![](https://i.imgur.com/js8kQpU.jpg)

# In[ ]:


dist = train[train['longestKill']<200]
#plt.rcParams['axes.axisbelow'] = True
dist.hist('longestKill', bins=20, figsize = (16,8))
plt.show()


# In[ ]:


print("Average longest kill distance a player achieve is {:.1f}m, 95% of them not more than {:.1f}m and a maximum distance is {:.1f}m." .format(train['longestKill'].mean(),train['longestKill'].quantile(0.95),train['longestKill'].max()))


# In[ ]:


walk0 = train["walkDistance"] == 0
print('number of players dint walk at all:',walk0.sum())


# In[ ]:


ride0 = train["rideDistance"] == 0
print('number of players who dint ride at all:',ride0.sum())


# In[ ]:


swim0=train['swimDistance']==0
print('number of players who dint swim at all',swim0.sum())


# now lets create the sum of walking,swimming and driving distances

# In[ ]:


travel_dist = train["walkDistance"] + train["rideDistance"] + train["swimDistance"]
travel_dist = travel_dist[travel_dist<5000]
travel_dist.hist(bins=40, figsize = (15,10))
plt.show()


# NOW LETS ANALYSE THE TOP 10 PLAYERS IN THE GIVEN DATASET

# In[ ]:


top10 = train[train["winPlacePerc"]>0.9]
print("TOP 10% overview\n")
print("Average number of kills: {:.1f}\nMinimum: {}\nThe best: {}\n95% of players within: {} kills." 
      .format(top10["kills"].mean(), top10["kills"].min(), top10["kills"].max(),top10["kills"].quantile(0.95)))


# In[ ]:


print("On average the best 10% of players have the longest kill at {:.3f} meters, and the best score is {:.1f} meters."
      .format(top10["longestKill"].mean(), top10["longestKill"].max()))

