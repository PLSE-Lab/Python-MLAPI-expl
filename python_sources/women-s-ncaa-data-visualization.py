#!/usr/bin/env python
# coding: utf-8

# # Loading Libraries

# In[ ]:


## Importing various Python libraries##

# Importing libraries for Numerical Python to perform linear algebra
# Importing Pandas library to perform data processing via panda's dataframes.

import numpy as np
import pandas as pd
from sklearn import *

# Importing libraries for Plotting my work

import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('ggplot')

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# # **Loading Data into different variables via pd.read_csv**

# In[ ]:


WCities = pd.read_csv('../input/WCities.csv')
WGameCities = pd.read_csv('../input/WGameCities.csv')
WNCAATourneyCompactResults = pd.read_csv('../input/WNCAATourneyCompactResults.csv')
WNCAATourneySeeds = pd.read_csv('../input/WNCAATourneySeeds.csv')
WNCAATourneySlots = pd.read_csv('../input/WNCAATourneySlots.csv')
WRegularSeasonCompactResults = pd.read_csv('../input/WRegularSeasonCompactResults.csv')
WSeasons = pd.read_csv('../input/WSeasons.csv')
WTeamSpellings = pd.read_csv('../input/WTeamSpellings.csv', engine='python')
WTeams = pd.read_csv('../input/WTeams.csv')


# In[ ]:


WNCAATourneyCompactResults.head(n=5)


# In[ ]:


WRegularSeasonCompactResults.head(n=5)


# In[ ]:


# Convert Tourney Seed to a Number
WNCAATourneySeeds['SeedNumber'] = WNCAATourneySeeds['Seed'].apply(lambda x: int(x[-2:]))


# In[ ]:


WNCAATourneySeeds.head(n=5)


# In[ ]:


# Merging information for better anlaytics
WNCAATourneyCompactResults['WSeed'] =     WNCAATourneyCompactResults[['Season','WTeamID']].merge(WNCAATourneySeeds,
                                                      left_on = ['Season','WTeamID'],
                                                      right_on = ['Season','TeamID'],
                                                      how='left')[['SeedNumber']]
WNCAATourneyCompactResults['LSeed'] =     WNCAATourneyCompactResults[['Season','LTeamID']].merge(WNCAATourneySeeds,
                                                      left_on = ['Season','LTeamID'],
                                                      right_on = ['Season','TeamID'],
                                                      how='left')[['SeedNumber']]

WNCAATourneyCompactResults =     WNCAATourneyCompactResults.merge(WGameCities,
                                how='left',
                                on=['Season','DayNum','WTeamID','LTeamID'])

WRegularSeasonCompactResults['WSeed'] =     WRegularSeasonCompactResults[['Season','WTeamID']].merge(WNCAATourneySeeds,
                                                        left_on = ['Season','WTeamID'],
                                                        right_on = ['Season','TeamID'],
                                                        how='left')[['SeedNumber']]
WRegularSeasonCompactResults['LSeed'] =     WRegularSeasonCompactResults[['Season','LTeamID']].merge(WNCAATourneySeeds,
                                                        left_on = ['Season','LTeamID'],
                                                        right_on = ['Season','TeamID'],
                                                        how='left')[['SeedNumber']]

WRegularSeasonCompactResults =     WRegularSeasonCompactResults.merge(WGameCities,
                                  how='left',
                                  on=['Season',
                                      'DayNum',
                                      'WTeamID',
                                      'LTeamID'])

# Add Season Results
WRegularSeasonCompactResults = WRegularSeasonCompactResults.merge(WSeasons,
                                                        how='left',
                                                        on='Season')
WNCAATourneyCompactResults = WNCAATourneyCompactResults.merge(WSeasons,
                                                    how='left',
                                                    on='Season')

# Add Team Names
WRegularSeasonCompactResults['WTeamName'] =     WRegularSeasonCompactResults[['WTeamID']].merge(WTeams,
                                               how='left',
                                               left_on='WTeamID',
                                               right_on='TeamID')[['TeamName']]
WRegularSeasonCompactResults['LTeamName'] =     WRegularSeasonCompactResults[['LTeamID']].merge(WTeams,
                                               how='left',
                                               left_on='LTeamID',
                                               right_on='TeamID')[['TeamName']]

WNCAATourneyCompactResults['WTeamName'] =     WNCAATourneyCompactResults[['WTeamID']].merge(WTeams,
                                             how='left',
                                             left_on='WTeamID',
                                             right_on='TeamID')[['TeamName']]
WNCAATourneyCompactResults['LTeamName'] =     WNCAATourneyCompactResults[['LTeamID']].merge(WTeams,
                                             how='left',
                                             left_on='LTeamID',
                                             right_on='TeamID')[['TeamName']]
    
WNCAATourneyCompactResults['ScoreDiff'] = WNCAATourneyCompactResults['WScore'] - WNCAATourneyCompactResults['LScore']


# In[ ]:


WRegularSeasonCompactResults.head(n=5)


# In[ ]:


WNCAATourneyCompactResults.head(n=5)


# # **Plot WNCAATorneySeeds for Top 15**

# In[ ]:


# Calculate the Average Team Seed
averageseed = WNCAATourneySeeds.groupby(['TeamID']).agg(np.mean).sort_values('SeedNumber')
averageseed = averageseed.merge(WTeams, left_index=True, right_on='TeamID') #Add Teamnname
averageseed.head(15).plot(x='TeamName',
                          y='SeedNumber',
                          kind='bar',
                          figsize=(15,5),
                          title='Top 15 Average Tournament Seed')


# In[ ]:


# Pairplot of the WNCAATourneyCompactResults file for all the Scores and Seeds. HUE as Season
sns.pairplot(WNCAATourneyCompactResults[['WScore',
                                    'LScore',
                                    'ScoreDiff',
                                    'WSeed',
                                    'LSeed',
                                    'Season']], hue='Season')


# In[ ]:


# Pairplot of the WRegularSeasonCompactResults file for all the Scores and Seeds. HUE as WSeed
regseason_in_tourney = WRegularSeasonCompactResults.dropna(subset=['WSeed','LSeed'])
sns.pairplot(data = regseason_in_tourney,
             vars=['WScore','LScore','WSeed','LSeed'],
             hue='WSeed')


# In[ ]:


WRegularSeasonCompactResults2017 = WRegularSeasonCompactResults.loc[WRegularSeasonCompactResults['Season'] == 2017]


# In[ ]:


bins = np.linspace(0, 120, 61)
plt.figure(figsize=(15,5))
plt.title('Last years W-L Score Distribution')
plt.hist(WRegularSeasonCompactResults2017['WScore'], bins, alpha=0.5, label='W-Score')
plt.hist(WRegularSeasonCompactResults2017['LScore'], bins, alpha=0.5, label='L-Score')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


bins = np.linspace(0, 120, 61)
plt.figure(figsize=(15,5))
plt.title('All Years W-L Score Distribution for Regular Seasons')
plt.hist(WRegularSeasonCompactResults['WScore'], bins, alpha=0.5, label='W-Score')
plt.hist(WRegularSeasonCompactResults['LScore'], bins, alpha=0.5, label='L-Score')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


bins = np.linspace(0, 120, 61)
plt.figure(figsize=(15,5))
plt.title('All Years W-L Score Distribution for Tournements')
plt.hist(WNCAATourneyCompactResults['WScore'], bins, alpha=0.5, label='W-Score')
plt.hist(WNCAATourneyCompactResults['LScore'], bins, alpha=0.5, label='L-Score')
plt.legend(loc='upper left')
plt.show()


# In[ ]:


fig,(ax1,ax2) = plt.subplots(nrows=2, figsize=(12,10))
a = sns.distplot(WNCAATourneyCompactResults['WScore'],label='Winning Score',ax=ax1)
a = sns.distplot(WNCAATourneyCompactResults['LScore'],ax=a,label='Losing Score')
a.set_xlabel("Score distribution-Tourney Results")

b = sns.distplot(WRegularSeasonCompactResults['WScore'],label='Winning Score',ax=ax2)
b = sns.distplot(WRegularSeasonCompactResults['LScore'],ax=b,label='Losing Score')
b.set_xlabel("Score distribution-RegSeason Results")

plt.legend();


# # **Plotting Seeds from WNCAATourneyCompactResults (Winners and Loosers)**

# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(12,4))
ax1 = sns.countplot(x=WNCAATourneyCompactResults['WSeed'],ax=ax1)
ax1.set_title("Winners - WNCAATourneyCompactResults")
ax2 = sns.countplot(x=WNCAATourneyCompactResults['LSeed'],ax=ax2);
ax2.set_title("Loosers - WNCAATourneyCompactResults")

plt.legend();


# # **Plotting Seeds from WRegularSeasonCompactResults (Winners and Loosers)**

# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(12,4))
ax1 = sns.countplot(x=WRegularSeasonCompactResults['WSeed'],ax=ax1)
ax1.set_title("Winners - WRegularSeasonCompactResults")
ax2 = sns.countplot(x=WRegularSeasonCompactResults['LSeed'],ax=ax2)
ax2.set_title("Loosers - WRegularSeasonCompactResults")

plt.legend();


# In[ ]:


fig,(ax1,ax2) = plt.subplots(ncols=2, figsize=(12,4))
ax1 = sns.countplot(x=WRegularSeasonCompactResults['WLoc'],ax=ax1)
ax1.set_title("Regular Season")
ax1.set_xlabel('Winning location')
ax2 = sns.countplot(x=WNCAATourneyCompactResults['WLoc'],ax=ax2)
ax2.set_title("Tournaments")
ax2.set_xlabel('Winning location')

plt.legend();


# # **Great Teams and Not so Great Teams from the entire WRegularSeasonCompactResults dataset**

# In[ ]:


# Not So Great Teams
count_of_losses = WRegularSeasonCompactResults.groupby('LTeamID')['LTeamID'].agg('count')
count_of_losses = count_of_losses.sort_values(ascending=False)
team_loss_count = pd.DataFrame(count_of_losses).merge(WTeams, left_index=True, right_on='TeamID')[['TeamName','LTeamID']]
team_loss_count.rename(columns={'LTeamID':'Loss Count'}).head(10)


# In[ ]:


# Great Teams
count_of_wins = WRegularSeasonCompactResults.groupby('WTeamID')['WTeamID'].agg('count')
count_of_wins = count_of_wins.sort_values(ascending=False)
team_wins_count = pd.DataFrame(count_of_wins).merge(WTeams, left_index=True, right_on='TeamID')[['TeamName','WTeamID']]
team_wins_count.rename(columns={'WTeamID':'Win Count'}).head(10)


# In[ ]:


Win_Loss_Start = pd.merge(team_wins_count, team_loss_count, how='outer')


# In[ ]:


Win_Loss_Start.sort_values('WTeamID', ascending=False).head(40)


# # **Great Teams From Last Year's Season**

# In[ ]:


# Great teams from 2017
Winning_Count2017 = WRegularSeasonCompactResults2017.groupby('WTeamID')['WTeamID'].agg('count')
Winning_Count2017 = Winning_Count2017.sort_values(ascending=False)
Winning_Teams_Count2017 = pd.DataFrame(Winning_Count2017).merge(WTeams, left_index=True, right_on='TeamID')[['TeamName','WTeamID']]
Winning_Teams_Count2017 = Winning_Teams_Count2017.rename(columns={'WTeamID':'Win Count'})

Lossing_Count2017 = WRegularSeasonCompactResults2017.groupby('LTeamID')['LTeamID'].agg('count')
Lossing_Count2017 = Lossing_Count2017.sort_values(ascending=False)
Loosing_Teams_Count2017 = pd.DataFrame(Lossing_Count2017).merge(WTeams, left_index=True, right_on='TeamID')[['TeamName','LTeamID']]
Loosing_Teams_Count2017 = Loosing_Teams_Count2017.rename(columns={'LTeamID':'Loss Count'})

WinLoss2017 = pd.merge(Winning_Teams_Count2017, Loosing_Teams_Count2017, how='outer')
WinLoss2017.sort_values('Win Count', ascending=False).head(26)
WinLoss2017 = WinLoss2017.fillna(0)
WinLoss2017.head(20)


# # **ScoreDiff Distribution**

# In[ ]:


WRegularSeasonCompactResults['ScoreDiff'] = WRegularSeasonCompactResults['WScore'] - WRegularSeasonCompactResults['LScore']


# In[ ]:


WRegularSeasonCompactResults['ScoreDiff'].hist(bins=40)


# # **Histogram of WNCAATourneyCompactResults for HOME, Away & Neutral Site games

# In[ ]:


# Distributionbased on WLoc field.
a = WNCAATourneyCompactResults.loc[WNCAATourneyCompactResults['WLoc'] == 'H']['ScoreDiff'].hist()
b = WNCAATourneyCompactResults.loc[WNCAATourneyCompactResults['WLoc'] == 'A']['ScoreDiff'].hist()
c = WNCAATourneyCompactResults.loc[WNCAATourneyCompactResults['WLoc'] == 'N']['ScoreDiff'].hist()

