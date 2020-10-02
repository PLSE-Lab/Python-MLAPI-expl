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


# # --------------------NFL Surface Analytics-------------
# # -------------------[By Anurag Aiyer]------------------

# ## The NFL is currently interested in seeing the factors that different playing surfaces have towards player performance and injury. In this notebook, we will be conducting exploratory analysis to solidify and understand the issues that ascertain towards key categorization of factors toward injuries. The goal is to try to gain an understanding into how we can find relationships between injuries and NFL turfs efficient, so that we can try to create a better understanding and perspective of how better regulations can be implemented. As of right now, there are concussion-sustained injuries that virtually last a life time and there is not much physicians or sport health specialists that can address the complete root cause.

# # Here is the following breakdown of the notebook:
# 1. Looking at the different frequency of injuries for each player. What is the general outlook overall?
#    
# 2. Understanding or viewing any association between surfaces and injuries
# 
# 3. Potentially discovering any classifications or further interpretations of the data that can be made
# 
# 4. Maybe viewing fluctuations or other alternatives that could be considered to enforcing strict NFL protocols

# # **[1]** MOST FREQUENT INJURIES FOR EACH POSITION

# In[ ]:


#READING all the csv downloaded from Kaggle API
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cmap
import seaborn as sns 


# In[ ]:


playerlist = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayList.csv')
injury_record = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/InjuryRecord.csv')
trackerdata = pd.read_csv('/kaggle/input/nfl-playing-surface-analytics/PlayerTrackData.csv')


# In[ ]:


relationship = playerlist.drop_duplicates('PlayerKey', keep='first').merge(injury_record, on='PlayerKey')
relationship['RosterPosition'].value_counts().plot(kind='bar', title='INJURIES VS Position of Player')


# ## Out of **105** injured players within the past 2 seasons, it is clear that the two positions for the highest number of injuries are **WIDE RECEIVER** and **LINEBACKER**. Majority of those injuries are either knee or ankle cases.

# ## Let us **view** the relationship between each position and specific injuries

# In[ ]:


df = relationship[['RosterPosition','Surface','BodyPart']]
dfnew = df.groupby(['RosterPosition', 'BodyPart']).count()
#
#result = df.pivot(index='Surface', columns='RosterPosition', values='value')
dfrev = dfnew['Surface'].reset_index().rename(columns={'Surface':'Count'})
ax = sns.heatmap(dfrev.pivot_table(index='RosterPosition', columns = 'BodyPart', values='Count'))


# # Linebackers have taken a major beating in terms of facing adversity and not fully rebouncing or recovering from the depths of injuries. It is important to understad other features as well that affect health well-being.

# # **[2]** NFL SURFACES CONTRIBUTION TO INJURIES

# # Here is a relationship between RosterPosition and the different surfaces that reflects how injuries are happening. We can conclude that synthetic surfaces have more potential injuries than natural surfaces. This is especially the case for cornerbacks.
# 
# 

# In[ ]:


surfaceinjuries = relationship[['RosterPosition', 'Surface']].groupby(['RosterPosition', 'Surface']).size().reindex()
lst = []
lsttwo = []
counter = 0
countertwo = 1
#Natural Surface Injuries  
while counter < len(surfaceinjuries):
    lst.append(int(surfaceinjuries[counter]))
    counter = counter + 2
    
    
#Synthetic Surface Injuries
while countertwo < len(surfaceinjuries)+1:
    lsttwo.append(surfaceinjuries[countertwo])
    countertwo = countertwo + 2
    
# set width of bar
barWidth = 0.25
 
# set height of bar
bars1 = lst
bars2 = lsttwo
 
# Set position of bar on X axis
r1 = np.arange(len(bars1))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, bars1, color='#7f6d5f', width=barWidth, edgecolor='white', label='Natural')
plt.bar(r2, bars2, color='#557f2d', width=barWidth, edgecolor='white', label='Synthetic')
#plt.bar(r3, bars3, color='#2d7f5e', width=barWidth, edgecolor='white', label='var3')
 
# Add xticks on the middle of the group bars
plt.xlabel('Position', fontweight='bold')
plt.ylabel('Injuries', fontweight='bold')

plt.xticks([r + barWidth for r in range(len(bars1))], ['Cornerback', 'Defensive Lineman', 'Linebacker', 'Offensive Lineman', 'Running Back', 'Safety', 'Tight End', 'Wide Receiver'])

plt.xticks(rotation='vertical')
plt.title('NFL Surface Lower-Limb Injuries', fontweight='bold')
# Create legend & Show graphic
plt.legend()
plt.show()


#Creating Stacked Bar with SurfaceInjuries among Lower-Limb NFL candidates
#relationship[['RosterPosition', 'Surface']].groupby(['RosterPosition', 'Surface'])
    
#for x in range(0, len(surfaceinjuries), 2):
    #natural = lst.append(surfaceinjuries[x])
#list(surfaceinjuries.to_frame().columns)


# # Since we went through a general rundown of how RosterPositions and injuries have a few associations, now let us further explore and deepen our understanding with how NFL turfs are affecting player movement. Some potential factors to think about our whether the speed or orientation is impacting the movement through a turf. If so,does higher velocity speed or a larger angle of inclination with respect to the ground emphasize a potential concern for injuries. In fact, what would be the minimum orientation such that a player would sustain a lower-limb injury? Realistically, this raises speculations to what kind of improvements can be made so that NFL players are much safer on the field. 

# # With a few questions of these in mind, let's start our analysis. 
# 
# 

# # **[3]** DISCOVERY STAGE

# In[ ]:


numplays = trackerdata.groupby('PlayKey').size().to_frame().reset_index()
injury_record
merged_set = pd.merge(injury_record, numplays, on='PlayKey')
#For 77 NFL lower-limb injures, looking at any possible trends or other signficant factors that highly affect the outcome
#of the total # of plays and number of days missed during practices


# In[ ]:


merged_set['sum'] = merged_set.iloc[:,5:9].sum(axis = 1)


# In[ ]:


missed_time = []
for x in merged_set['sum']:
    if x == 4:
        missed_time.append('DM_42')
    elif x == 3:
        missed_time.append('DM_28')
    elif x == 2:
        missed_time.append('DM_7')
    else:
        missed_time.append('DM_1')
merged_set['NumDaysMissed'] = missed_time


# In[ ]:


missed_time


# # Average Number of Plays for Day of Missed Games(77 injured players)
# 

# In[ ]:


merged_set = merged_set.iloc[:, [9, 11]].groupby('NumDaysMissed').mean()





# In[ ]:


merged_set.reindex().rename(columns={'0':'AverageNumPlays'}).plot(kind='bar', title='Average Plays and Relationship with Num Days Missed for Injured NFL Players', legend=None)
plt.xlabel('NumDaysMissed')
plt.ylabel('AveragePlays')


# # There seems to be a positive association between the number of days missed from participating in NFL games and the average number of plays injured players participated in a particular game. It would be interesting to intertwine and look at further examples for safety measurements. Now, it is imperative to view how orientation of an average play is measured and constructed with respect to the speed of a particular individual.

# In[ ]:


numplays = trackerdata.groupby('PlayKey').size().to_frame().reset_index()


# In[ ]:


trackerdata


# In[ ]:


playerlist


# In[ ]:


playerlist['PlayType']


# In[ ]:


injury_record


# In[ ]:


playerlist.loc[playerlist['PlayerKey']==26624]


# # Most played fieldtypes that reflect the current atmosphere and different feel into the various fieldtypes. As an expample, we were able to find how we could figure the way in which FieldType provides useful insights into how different stadiums affect the larger chaos the world seems to be undergoing. 

# In[ ]:


playerlist.iloc[:, [ 6, 7]].groupby(['StadiumType']).count().reset_index().sort_values(by='FieldType', ascending=False).iloc[0:5].set_index('StadiumType').plot.bar(y='FieldType', rot=0)


# # There is some research that has provided further segway into he we could manipulate and get something fotunate about our life. Something to be cognizant about is the vulnerability for NFL players to be prone to inury until somethinbg possibly fes around out. I'm sure that the NFL injured citizens will not take anything granted. This is a place of development and belief in yoursef. 

# In[ ]:




