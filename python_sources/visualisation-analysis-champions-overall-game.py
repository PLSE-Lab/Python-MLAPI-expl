#!/usr/bin/env python
# coding: utf-8

# ![Image of Yaktocat](https://images.contentstack.io/v3/assets/blt731acb42bb3d1659/blt45d6c2043ff36e28/5e21837f146ca8115b2d3332/Champion-List.jpg)
# 
# # Gameplan  
# 1. Clean data + make it relevant to info to be extracted.
# 2. Visualisation - Will first do Champion specific stuff (win rates and positional pick frequency), then game wide what factors are important for winning overall, and in short and long games.

# In[ ]:


import sklearn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


# # **Read in data**

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


filepath = r'/kaggle/input/league-of-legends-ranked-matches/matches.csv'
matches = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/champs.csv'
champs = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/participants.csv'
participants = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/stats1.csv'
stats1 = pd.read_csv(filepath)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/stats2.csv'
stats2 = pd.read_csv(filepath)
stats = stats1.append(stats2)
filepath = r'/kaggle/input/league-of-legends-ranked-matches/teamstats.csv'
teamstats = pd.read_csv(filepath)


# # Merging together the bits of data I want to use

# In[ ]:


df = pd.merge(participants, stats, how = 'left', on = ['id'], suffixes=('', '_y'))
df = pd.merge(df, champs, how = 'left', left_on = 'championid', right_on = 'id', suffixes=('', '_y'))
df = pd.merge(df, matches, how = 'left', left_on = 'matchid', right_on = 'id', suffixes=('', '_y'))
df.head()


# In[ ]:


pd.options.display.max_columns = None
df.columns


# In[ ]:


# selecting a smaller subset of factors to look at
# using the describe function as a preliminary way to check the data seems alright (i.e. spot any errors)

df = df[['matchid', 'player', 'name', 'position', 'win', 'kills', 'deaths', 'assists', 'largestkillingspree', 'largestmultikill', 'longesttimespentliving', 'totdmgdealt', 'totdmgtochamp', 'totdmgtaken', 'turretkills', 'totminionskilled', 'goldearned', 'wardsplaced', 'duration', 'firstblood', 'seasonid']]
df.describe()


# # Dataset cleaning
# 
# 1. Will use just season 8, as makes up majority of data, and is most relevant.
# 2. Will remove anomalous totdmgdealt value (shouldn't be -ve).
# 3. Remove games that are under 5 minutes long, as they probably include afk players.

# In[ ]:


# roughly 150k rows removed i.e. ~ 15,000 matches

print(df.shape)
df = df[df['seasonid'] == 8]
df = df[df['duration'] >= 300]
df = df[df['totdmgdealt'] >= 0]

print(df.shape)


# # Champions position and win-rate anlysis

# In[ ]:


pd.options.display.float_format = '{:,.1f}'.format

df_win_rate = df.groupby('name').agg({'win': 'sum', 'name': 'count', 'kills': 'mean', 'deaths': 'mean', 'assists': 'mean'})
df_win_rate.columns = ['win matches', 'total matches', 'K', 'D', 'A']
df_win_rate['win rate'] = df_win_rate['win matches'] /  df_win_rate['total matches'] * 100
df_win_rate['KDA'] = (df_win_rate['K'] + df_win_rate['A']) / df_win_rate['D']
df_win_rate = df_win_rate.sort_values('win rate', ascending = False)
df_win_rate = df_win_rate[['total matches', 'win rate', 'K', 'D', 'A', 'KDA']]

# adding position that champions are most commonly played in

df_test = df.groupby('name').position.apply(lambda x: x.mode())
df_new = pd.merge(df_win_rate, df_test, how = 'left', on = ['name'], suffixes=('', '_y'))
df_new

print('Top 10 win rate')
print(df_new.head(10))
print('Bottom 10 win rate')
print(df_new.tail(10))


# # Conclusions  
# 
# 1. The game is overall pretty well balanced.  
# 2. Ryze is a trash pick. I don't know much about the game, but I hope he's recieved a buff by now.

# In[ ]:


# this cell allows visulisation of the distribution of positions played for each champion. Shows most champions have a clear
# favorite lane that they are played in.

champ = 'Anivia'

df_test = df[df['name'] == champ]
print(df_test['position'].value_counts())
plt.figure(figsize=(12,8))
plt.title('Distribution of position played for '+str(champ))
plt.ylabel('# of games position picked')
sns.countplot(df_test['position'])


# # Correlation between factors

# In[ ]:


df_corr = df._get_numeric_data()
df_corr = df_corr.drop(['matchid', 'player'], axis = 1)


# In[ ]:


corr = df_corr.corr()
plt.figure(figsize=(15,10))
ax= plt.subplot()

mask = np.zeros_like(df_corr.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

sns.heatmap(corr, ax=ax, annot=True,square=True, linewidths=.5, center = 0, mask = mask, cmap=cmap, fmt = '.2f')


# # Conclusions  
# 1. Turret kills strongest correlation with being on winning team. There is also positive correlation with gold earned, kills (and related stats), and negative correlation with deaths (obviously).  
# 2. Wards placed, minions killed, first blood, and damage taken have little correlation with winning.  
#   
# Game duration has no correlation with winning (makes sense) - would be interesting to see how correlation changes when different game lengths are looked at. Maybe for shorter games things like gold earned/kills/deaths will be more important, as can get stronger items before other team and use advantage to win. Also, maybe things like first blood more important for shorter games.

# # Match duration distribution

# In[ ]:


# average game time is ~ 31 minutes

df.loc[:,'duration'].describe()


# In[ ]:


df.hist(column='duration', bins=40)


# (The early peak might be due to games ending early when people are AFK.) --> this got removed by adjustment to data cleaning
# 
# Next bit of analysis will look at short games vs long games i.e. above and below the average time.

# # Shorter Game Analysis
# Looking at games a chunk of time less than average (<25mins)

# In[ ]:


df_corr_2 = df._get_numeric_data()
df_corr_2 = df_corr_2.drop(['matchid', 'player'], axis = 1)
# for games less than 25mins
df_corr_2 = df_corr_2[df_corr_2['duration'] <= 1500]


# In[ ]:


corr = df_corr_2.corr()
plt.figure(figsize=(15,10))
ax= plt.subplot()

mask = np.zeros_like(df_corr_2.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

sns.heatmap(corr, ax=ax, annot=True,square=True, linewidths=.5, center = 0, mask = mask, cmap=cmap, fmt = '.2f')


# # Conclusions  
# 1. As hypothesised, there is magnification of correlation of certain factors (e.g. gold earned, kills, deaths). Implies in short games actions are more meaningful e.g. a kill/death is what tips the balance that allows a team to win a short game. On the other hand, in longer, slower games, each death isn't as meaningful to the overall result.
# 2. Jump in first blood/win correlation. Makes sense, reiterates previous notion.
# 3. There is the implication that increased vision from warding seems to aid positioning for assists, as well as enabling you to stay alive for longer.
# 4. Gold earned strongly correlates with a lot of attributes that indicate the benefits of having good items e.g. you can deal more damage, get more kills, fo on big killing sprees. Still doesn't as strongly correlate to overall win, implying teamwork is important in league.

# # Longer Games
# Longer than 40 mins

# In[ ]:


df_corr_3 = df._get_numeric_data()
df_corr_3 = df_corr_3.drop(['matchid', 'player'], axis = 1)
df_corr_3 = df_corr_3[df_corr_3['duration'] >= 2400]


# In[ ]:


corr = df_corr_3.corr()
plt.figure(figsize=(15,10))
ax= plt.subplot()

mask = np.zeros_like(df_corr_3.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
cmap = sns.diverging_palette(240, 10, n=9, as_cmap=True)

sns.heatmap(corr, ax=ax, annot=True,square=True, linewidths=.5, center = 0, mask = mask, cmap=cmap, fmt = '.2f')


# # Conclusions  
# 1. Overall conclusions can be made that the longer the game drags on, the lower the importance of individual things dictating the overall result. This makes inuitive sense, as longer games are likely to be more even stats wise (if one team has much better stats than the other it is likely to win quickly). Still, the same correlation patterns are seen with what is indicative of a winning team member e.g. turret kills, gold earned, kills, less deaths, etc.
# 2. Interesting -ve correlation between wards placed and kills. Perhaps having more vision makes you a more cautious player and less likely to get into positions where you get kills. Also -ve correlation with total damage dealt; again, there is possibly less engagment with increased vision.

# Please let me know if you agree/disagree with any of the conclusions, + if there any other things that'd be of interest to analyse/visualise.
