#!/usr/bin/env python
# coding: utf-8

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


import matplotlib.pyplot as plt
import seaborn as sns
import collections


# ### Team Data
# 
# Team name and Team ID. 

# In[ ]:


WTeams = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WTeams.csv')

WTeams


# ### Seasons
# 
# + Day
# + Region

# In[ ]:


WSeasons = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WSeasons.csv')

WSeasons


# ### TouneySeeds

# In[ ]:


WTouneySeeds = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySeeds.csv')

WTouneySeeds


# ### RegularSeasonResults
# 
# **Attention**
# + all games played from DayNum 0 through 132
# + 'RegularSeasonGame' are defined to be all games played on DayNum=132 or earlier (DayNum = 133 is Selection Monday)
# 
# + WTeamID : This is id number of the team that won the game
# + WScore : This is the number of points scored by the winning team
# + LTeamID : looser
# + LScore : points scored by the lost the game
# + NumOT : overtime
# + WLoc : this is the "location" of the winning team. If the winning team was the home team, this value will be "H". If the winning team was the visiting team, this value will be "A". If it was played on a neutral court, then this value will be "N".

# In[ ]:


SeasonResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonCompactResults.csv')

SeasonResults


# ### TourneyCompact
# 
# 2017 season through 2020 season:
# + Round 1 = days 137/138 (Fri/Sat)
# + Round 2 = days 139/140 (Sun/Mon)
# + Round 3 = days 144/145 (Sweet Sixteen, Fri/Sat)
# + Round 4 = days 146/147 (Elite Eight, Sun/Mon)
# + National Seminfinal = day 151 (Fri)
# + National Final = day 153 (Sun)
# 
# 2015 season and 2016 season:
# + Round 1 = days 137/138 (Fri/Sat)
# + Round 2 = days 139/140 (Sun/Mon)
# + Round 3 = days 144/145 (Sweet Sixteen, Fri/Sat)
# + Round 4 = days 146/147 (Elite Eight, Sun/Mon)
# + National Seminfinal = day 153 (Sun)
# + National Final = day 155 (Tue)
# 
# 2003 season through 2014 season:
# + Round 1 = days 138/139 (Sat/Sun)
# + Round 2 = days 140/141 (Mon/Tue)
# + Round 3 = days 145/146 (Sweet Sixteen, Sat/Sun)
# + Round 4 = days 147/148 (Elite Eight, Mon/Tue)
# + National Seminfinal = day 153 (Sun)
# + National Final = day 155 (Tue)
# 
# 1998 season through 2002 season:
# + Round 1 = days 137/138 (Fri/Sat)
# + Round 2 = days 139/140 (Sun/Mon)
# + Round 3 = day 145 only (Sweet Sixteen, Sat)
# + Round 4 = day 147 only (Elite Eight, Mon)
# + National Seminfinal = day 151 (Fri)
# + National Final = day 153 (Sun)

# In[ ]:


TouneyResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyCompactResults.csv')

TouneyResults


# ### TeamBoxScore
# This section provides game-by-game stats at a team level (free throws attempted, defensive rebounds, turnovers, etc.) for all regular season.
# + WFGM - field goals made (by the winning team)
# + WFGA - field goals attempted (by the winning team)
# + WFGM3 - three pointers made (by the winning team)
# + WFGA3 - three pointers attempted (by the winning team)
# + WFTM - free throws made (by the winning team)
# + WFTA - free throws attempted (by the winning team)
# + WOR - offensive rebounds (pulled by the winning team)
# + WDR - defensive rebounds (pulled by the winning team)
# + + + WAst - assists (by the winning team)
# + + WTO - turnovers committed (by the winning team)
# + WStl - steals (accomplished by the winning team)
# + WBlk - blocks (accomplished by the winning team)
# + WPF - personal fouls committed (by the winning team)

# In[ ]:


RegularDetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WRegularSeasonDetailedResults.csv')

RegularDetailedResults


# In[ ]:


NCAADetailedResults = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneyDetailedResults.csv')

NCAADetailedResults


# ### Geography
# This section provides city locations of all regular season, conference tournament, and NCAA tournament games since the 2009-19 season.
# 
# ### Cities
# This file provides a master list of cities that have been locations for games played.
# 
# ### GameCities
# This file identifies all games, starting with the 2010 season, along with the city that the game was played in.

# In[ ]:


Cities = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/Cities.csv')

Cities


# In[ ]:


GameCities = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WGameCities.csv')

GameCities


# ### Play-by-play
# This section proveides play-by-play event logs for more than 99% of each year's regular season and NCAA tournament, and secondary tournament women's games since the 2014-15 season - including plays by individual players.
# 
# + EventID
# + Season
# + DayNum
# + WTeamID
# + LTeamID
# + WFinalScore = WScore
# + LFinalScore = LScore
# + WCurrentScore
# + LCurrentSCore
# + ELapsedSeconds : This is the number of seconds that have elapsed from the start of the game until the event occurred.
# + EventTeamID
# + EventPlayerID 
# + EventType
# + EventSubType
# 
# ### Players
# + PlayerID
# + LastName
# + TeamID

# In[ ]:


WEvents2015 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2015.csv')
WEvents2016 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2016.csv')
WEvents2017 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2017.csv')
WEvents2018 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2018.csv')
WEvents2019 = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WEvents2019.csv')

WEvents2015


# In[ ]:


Player = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WPlayers.csv')

Player


# ### Supplements
# This section contains additional supporting information, including alternative team name spellings and representations of bracket structure.
# 
# ### TeamSpelling
# + TeamNameSpelling : This is the spelling of the team name.
# + TeamID
# 
# ### Slots
# This file identifies th mechanism by which teams  are paired against each other, depending upon their seeds, as the tounament proceeds through its rounds. It can be of use in identifying, for a given historical game, what round it occurred in, and what the seeds/slots were for the two teams (the meaning of "slots" is described below).
# + slots :  this uniquely identifies one of the tournament games. It is a four-character string, where the first two characters tell you which round the game is (R1, R2, R3, R4, R5, or R6) and the second two characters tell you the expected seed of the favored team. Thus the first row is R1W1, identifying the Round 1 game played in the W bracket, where the favored team is the 1 seed. As a further example, the R2W1 slot indicates the Round 2 game that would have the 1 seed from the W bracket, assuming that all favored teams have won up to that point. The slot names are different for the final two rounds, where R5WX identifies the national semifinal game between the winners of regions W and X, and R5YZ identifies the national semifinal game between the winners of regions Y and Z, and R6CH identifies the championship game. The "slot" value is used in other columns in order to represent the advancement and pairings of winners of previous games.
# 
# + StrongSeed : This indicates the expected stronger-seeded team that playsin this game.
# + WeakSeed
# 
# ### Conferenses
# This file indicates the Division 1 conferences that have existed over the years since 1985.
# + ConfAbbrev : shorter text name
# + Description : longer text name
# 
# ### TeamConferences
# This file indicates the conference affiliations for each team during each season. Some conferences have changed their names from year to year, and/or changed which teams are part of the conference.
# + season
# + TeamID
# + ConfAbbrev

# In[ ]:


Slots = pd.read_csv('/kaggle/input/google-cloud-ncaa-march-madness-2020-division-1-womens-tournament/WDataFiles_Stage1/WNCAATourneySlots.csv')

Slots


# ## Data Wrangling

# In[ ]:


WTouneySeeds


# In[ ]:


WTeams


# In[ ]:


pd.merge(WTouneySeeds,WTeams,on='TeamID').set_index('TeamID')


# In[ ]:


SeasonResults


# ## Visualization
# ### SeasonData
# basketball performance indicators according to home/away games and winning and losing teams

# In[ ]:


collections.Counter(SeasonResults['WLoc'])


# In[ ]:


SeasonResults['WLoc'].value_counts()


# In[ ]:


SeasonResults['WLoc'].value_counts().plot(kind='bar')
plt.legend()
plt.show()


# In[ ]:


SeasonResults['WScore'].value_counts()


# In[ ]:


fig,ax=plt.subplots(1,figsize=(15,10))

sns.kdeplot(SeasonResults['WScore'],color='green',shade=True,ax=ax)
sns.kdeplot(SeasonResults['LScore'],shade=True,ax=ax)

plt.legend()
plt.show()


# In[ ]:


SortWScore= SeasonResults.sort_values('WScore',ascending=False).head(10)

SortWScore


# In[ ]:


sns.barplot(x='WScore',y='WTeamID',data=SortWScore,palette='Set3_r',orient="h")


# In[ ]:


fig,ax=plt.subplots(1,figsize=(15,10))

sns.kdeplot(SeasonResults.loc[(SeasonResults['WLoc']=='H'),'WScore'],color='green',shade=True,ax=ax)
sns.kdeplot(SeasonResults.loc[(SeasonResults['WLoc']=='A'),'WScore'],color='red',shade=True,ax=ax)
sns.kdeplot(SeasonResults.loc[(SeasonResults['WLoc']=='N'),'WScore'],color='blue',shade=True,ax=ax)

plt.show()


# In[ ]:


WScoreFrequency = SeasonResults['WScore'].value_counts()
WScoreFrequency.index.names = ['WScore']

LScoreFrequency = SeasonResults['LScore'].value_counts()


# In[ ]:


SortW = SeasonResults.sort_values('WScore',ascending=False)
SortL = SeasonResults.sort_values('LScore',ascending=False)

SortW['WScore'].plot(kind='hist',bins=50,label='WScore',alpha=0.5)
SortL['LScore'].plot(kind='hist',bins=50,label='LScore',alpha=0.5)
plt.title('Score Frequency')
plt.legend()
plt.show()


# In[ ]:


SeasonResults.groupby(['WTeamID','LTeamID']).size().unstack().fillna(0).style.background_gradient(axis=1)


# In[ ]:


SeasonResults['counter']=1
SeasonResults.groupby('WTeamID')['counter']     .count()     .sort_values()     .tail(20)     .plot(kind='barh',figsize=(15,8),xlim=(400,680))
plt.show()


# ## Event Data

# In[ ]:


WEvents2015.head()


# In[ ]:


WEvents2015['counter']=1
WEvents2015.groupby('EventType')['counter']     .sum()     .sort_values(ascending=False)    .plot(kind='bar',figsize=(15,5),title='Event Type Frequency 2015')
plt.show()

