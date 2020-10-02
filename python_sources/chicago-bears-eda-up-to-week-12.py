#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd# data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use('fivethirtyeight')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
pd.options.mode.chained_assignment = None

# Any results you write to the current directory are saved as output.


# In the Preseason the Chicago Bears were heralded as playoff favorites and Super Bowl Contenders. 12 Weeks into the season the Bears are 5-7, technically in the hunt but ESPN predicts them to have a 1% chance of making it in. Throughout the Offseason, fans heard that the offense was taking steps into the '202' progression of plays and most expected a big jump from 2018 considering they were average in almost every statistical category. This report investigates the inner workings of the Chicago Bears offense. 
# 
# Expected Points Added is used to add more value to the limited nature of yards gained by adding field position and down & distance. 
# 
# Gaining three yards on  2nd down and 10, and gaining three yards on 4th down and 3, will both register as +3 in yards gained. But, the three yards gained with three yards to go is more valuable than the three yards gained with 10 yards to go, because it creates a first down and keeps the chains moving. The same principle applies with field position. Taking a sack in general is bad but taking one in your own terriotory vs taking one in field goal range are different and should be reflected in statistics. A loss of 10 on your 35 is graded differently than a loss of 10 on your opponents 35 using EPA. 
# 
# Calculating EPA is used from next points scored for total plays. As the points are added up for and against the offensive team, the differential point advantage can be found for any down and distance.
# ![](http://)
# 
# Datafiles are provided by NflscrapR. 

# ![](http://https://www.chicagobears.com/)

# ![61g9IpO3TJL._SL1024_.jpg](attachment:61g9IpO3TJL._SL1024_.jpg)
# 

# In[ ]:


import pandas as pd
data = pd.read_csv("../input/nfl-data-up-to-week-12/Week12NFLData.csv")


# In[ ]:


data = data.loc[(data.play_type.isin(['pass','run', 'no_play'])) &
               data.epa.isna()==False]


# Creating a dataframe with Run, Pass and NoPlays (that are penalties). 

# In[ ]:


data[data['play_type'] =='no_play']


# In[ ]:


data.loc[data.desc.str.contains('left end|left tackle|left guard|up the middle|right guard|right tackle|right end|rushes'),
'play_type'] = 'run'

data.loc[data.desc.str.contains('scrambles|sacked|pass'), 'play_type'] = 'pass'
data.reset_index(drop = True, inplace = True)


# Searching the "description" column in the data to find plays that contain a position or "scramble" or "sacked" to included in runs and passes respectively. 

# In[ ]:


#Create a smaller dataframe with plays where rusher_player_name is null
rusher_nan = data.loc[(data['play_type'] == 'run') &
         (data['rusher_player_name'].isnull())]
         
#Create a list of the indexes/indices for the plays where rusher_player_name is null
rusher_nan_indices = list(rusher_nan.index)

for i in rusher_nan_indices:
    #Split the description on the blank spaces, isolating each word
    desc = data['desc'].iloc[i].split()
    #For each word in the play description
    for j in range(0,len(desc)):
        #If a word is right, up, or left
        if desc[j] == 'right' or desc[j] == 'up' or desc[j] == 'left':
            #Set rusher_player_name for that play to the word just before the direction
            data['rusher_player_name'].iloc[i] = desc[j-1]     
        else:
            pass
#Create a smaller dataframe with plays where passer_player_name is null
passer_nan = data.loc[(data['play_type'] == 'pass') &
         (data['passer_player_name'].isnull())]
#Create a list of the indexes/indices for the plays where passer_player_name is null
passer_nan_indices = list(passer_nan.index)

for i in passer_nan_indices:
    #Split the description on the blank spaces, isolating each word
    desc = data['desc'].iloc[i].split()
    #For each word in the play description
    for j in range(0,len(desc)):
        #If a word is pass
        if desc[j] == 'pass':
            data['passer_player_name'].iloc[i] = desc[j-1]            
        else:
            pass
#Change any backwards passes that incorrectly labeled passer_player_name as Backward
data.loc[data['passer_player_name'] == 'Backward', 'passer_player_name'] == float('NaN')

receiver_nan = data.loc[(data['play_type'] == 'pass') & 
                        (data['receiver_player_name'].isnull()) &
                        (data['desc'].str.contains('scrambles|sacked|incomplete')==False)]

receiver_nan_indices = list(receiver_nan.index)

for i in receiver_nan_indices:
    desc = data['desc'].iloc[i].split()

    for j in range(0,len(desc)):
        if (desc[j]=='left' or desc[j]=='right' or desc[j]=='middle') and (desc[j] != desc[-1]):
            if desc[j+1]=='to':
                data['receiver_player_name'].iloc[i] = desc[j+2]
        else:
            pass


# In[ ]:


data.loc[data['epa'] > 0, 'success'] = 1


# In[ ]:


bears_stats = data.loc[data['posteam'] == 'CHI']
bears_stats


# Loading only the Chicago Bears Data

# In[ ]:


plt.figure(1, figsize = (10,6))
plt.hist(bears_stats['epa'].loc[bears_stats['play_type'] == 'pass'], bins = 50, label = 'Pass', color = 'orange')
plt.hist(bears_stats['epa'].loc[bears_stats['play_type'] == 'run'], bins = 50, label = 'Run', alpha = .7, color = 'darkblue')
plt.xlabel('Expected Points Added')
plt.ylabel('Number of plays')
plt.title('EPA Distribution Based on Play Type - Bears 2019')
plt.text(6,50,'Data from nflscrapR', fontsize=10, alpha=.7)
#Will show the colors and labels of each histogram
plt.legend()
plt.show()


# Distribution of EPA on Runs vs Pass plays. As we can see the Bears have more runs than passes and they seem to be more spread and slightly more successful. Running the ball has been a struggle for the Bears all season, injuries on the Offensive Line and regression from older players appear to be the biggest factors. According the Pro-Football Reference the Bears rank 20th in passing and 26th in rushing. 

# In[ ]:


sns.lmplot(data = bears_stats, x = 'yardline_100', y = 'epa', 
                  fit_reg = False, hue = 'play_type',
                    height = 8, 
                  scatter_kws = {'s':200})
plt.show()


# Looking at play distribution by where the Bears are on the field. Typically the closer to 0 (opposing teams goalline) teams should have high EPA rushing plays since that is the easiest way to score. But the Bears have a wide range of passing results and average rushing results. 

# In[ ]:


selected_column = ['pass_length', 'pass_location', 'run_location', 'run_gap', 'play_type']
for c in selected_column:
    print(bears_stats[c].value_counts(normalize=True).to_frame(), '\n')


# The Bears are a predominantly short passing team, preferring checkdowns over coverage beaters. Most passes going to the Quarterback's right which is similar to all NFL QB's since most are right handed. The Bears also are a horizontal rushing team, running to the Right and Left at similar clips but avoiding the middle. Passing calls of nearly ~50% is high, most NFL Coordinators prefer a 50/50 game and the league average currently is 60/40.

# In[ ]:


plt.figure(figsize=(16,9))
sns.swarmplot(x= bears_stats['qtr'],
              y = bears_stats['yards_gained'],
              hue = bears_stats['play_type'])
plt.legend(loc='upper right')
plt.show()


# The Bears offense in the first half has been abysmal this season, setting several records for longest time without touchdowns and points scored this season. But they gain more yards in the 3rd quarter through the ground and the air.

# In[ ]:


nfcnorth = data.loc[(data['posteam']== 'CHI') | (data['posteam']== 'DET') | 
                   (data['posteam']== 'GB') | (data['posteam']== 'MIN')]


# In[ ]:


sns.barplot( x = 'posteam', y = 'epa', data = nfcnorth)


# Comparing the Bears EPA to their divisional rivals.

# In[ ]:


nfl_wrs = data.loc[(data['play_type']=='pass') & (data['down']<=4)].groupby(by='receiver_player_name')[['epa','success','yards_gained', 'td_prob', 'air_yards', 'yardline_100','qtr', 'yac_epa']].mean()


# In[ ]:


bears_wrs = data.loc[(data['posteam']=='CHI') & (data['play_type']=='pass') & (data['down']<=4)].groupby(by='receiver_player_name')[['epa','success','yards_gained', 'td_prob', 'air_yards', 'yardline_100','qtr', 'yac_epa']].mean()


# In[ ]:


#Add new column
nfl_wrs['attempts'] = data.loc[(data['play_type']=='pass') & 
                        (data['down']<=4)].groupby(by='receiver_player_name')['epa'].count()

#Sort by mean epa
nfl_wrs.sort_values('epa', ascending=False, inplace=True)

#Filter by attempts
nfl_wrs = nfl_wrs.loc[nfl_wrs['attempts'] > 10] 

#
bears_wrs['attempts'] = data.loc[(data['posteam']=='CHI') & (data['play_type']=='pass') & 
                        (data['down']<=4)].groupby(by='receiver_player_name')['epa'].count()

#Sort by mean epa
bears_wrs.sort_values('epa', ascending=False, inplace=True)

#Filter by attempts
bears_wrs = bears_wrs.loc[bears_wrs['attempts'] > 10] 


# Specifically looking at the NFL and Bears WRs epa

# In[ ]:


bears_wrs


# Chart with all of the Bears WRs going over the epa, success (whether epa is greater than 0 or not), yards gained, probability of scoring a touchdown on that play, air yards gained, distance to opposing teams endzone, quarter, yards after catch epa.

# In[ ]:


plt.figure(figsize=(16,9))
plt.title('Receiver Usage for the Bears')
sns.countplot(bears_stats['receiver_player_name'])


# Counting the amount of attempts aka 'passing plays involved' the Bears receiving core had. Allen Robinson and Tarik Cohen are by far the Bears highest usage players. While Taylor Gabriel has had injuries this season holding his attempts down and 2nd year Anthony Miller is still figuring out the offense so isnt being used as frequently. The Tight Ends has been a troubled position all season with Burton, Holtz, Horsted, Braunecker and Shaheen all rotating. Shaheen and Burton were put on IR this week so the others should get a boost in attempts in coming weeks. 

# In[ ]:


plt.figure(figsize=(16,9))
plt.title('RunningBack Usage for the Bears')
sns.countplot(bears_stats['rusher_player_name'])


# Looking at the running backs rookie Montgomery is the clear #1 back with Cohen filling in for a change of pace. FA signee Davis barely got used through 12 weeks.  Along with the Bears using WRs on occasion for trick run plays. 

# In[ ]:


wr_table = pd.pivot_table(bears_stats, index = ['receiver_player_name'],
               columns = ['defteam'],
               values = ['epa'], aggfunc = [np.mean], fill_value = 0)        
wr_table


# Looking at all the Bears WRs EPA by Game this season. 

# In[ ]:


plt.figure(figsize=(16,9))
plt.rcParams['font.size'] = 10
bg_color = (0.88,0.85,0.95)
plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['axes.facecolor'] = bg_color
fig, ax = plt.subplots(1)
p = sns.heatmap(wr_table,
                cmap='coolwarm',
                annot=True,
                fmt=".1f",
                annot_kws={'size':16},
                ax=ax)
plt.xlabel('Vs Defense')
plt.ylabel('Receiver')
ax.set_ylim((0,15))
plt.text(3,13.3, "Bears WRs Heat Map", fontsize = 20, color='Black', fontstyle='italic')
plt.show()


# Heat map of who was playing well vs the Bears opponents this season. A few standout games from Montgomery, Miller, Wims and Braunecker. With Robinson and Gabriel having consistently positive EPA and Miller showing flashes but remains rocky with his production. But even these players are producing average EPA, for the offense to play better they have to make more plays. 

# In[ ]:


bears_rbs_pivot = pd.pivot_table(bears_stats,
            index = ['rusher_player_name'],
            columns = ['run_location'],
            values = ['epa'], 
            aggfunc = [np.mean],
            fill_value = 0)


# In[ ]:


plt.figure(figsize=(16,9))
plt.rcParams['font.size'] = 10
bg_color = (0.88,0.85,0.95)
plt.rcParams['figure.facecolor'] = bg_color
plt.rcParams['axes.facecolor'] = bg_color
fig, ax = plt.subplots(1)
p = sns.heatmap(bears_rbs_pivot,
                cmap='Blues',
                annot=True,
                fmt=".1f",
                annot_kws={'size':18},
                ax=ax)
plt.xlabel('Run Direction')
plt.ylabel('Receiver')
ax.set_ylim((0,15))
plt.text(1,10, "Bears RBs Heat Map", fontsize = 20, color='Black', fontstyle='italic')
plt.show()


# The rushing attack is not good with most players means being around 0. But from this set Montgomery is consistently solid running and Tarik is slightly hurting the rushing attack. QB Trubisky is positive rushing to his left. While using WRs for "trick" plays has generally been unsuccessful when ran with Anthony Miller but running with Gabriel and Patterson to the left has shown a slight uptick in EPA. 

# In[ ]:


sns.set(style="white", context="talk")
plt.figure(figsize=(20,9))
sns.violinplot(data = data[data.posteam =='CHI'],
                   x ='receiver_player_name', y = 'epa')
plt.show()


# In[ ]:


bears_wrs.epa.mean()


# In[ ]:


nfl_wrs.epa.mean()


# Bears WRs are performing below the NFL average at EPA. Biggest indicator of this is dropped passes which the Bears lead the NFL in. 

# In[ ]:


plt.figure(figsize=(16,9))
sns.set(style="white", context="talk")
sns.catplot(x="qtr", y='epa', hue="receiver_player_name", kind="swarm", data=bears_stats);
plt.show()


# Conclusions: The Bears offense looks to be stalling in 2019 for multiple reasons. 
# 
# The plays being called are skewed heavily towards pass, which can make it easier for the opposing team to defend. But when runs are called they are largely unsuccessful and not adding to the net EPA. 
# 
# This hinderance can be further explained by looking at the production from the Bears skill positions specifically the receiving core. Except for Allen Robinson none are performing close to the league average and it appears by having several receivers with middling attemps the coaching staff is unsure of who they can get production out of.
# 
# The QB was ignored in this report due to the dozens of analysis on him already available, examining the players and play calls around him seemed prudent given the teams record. 
