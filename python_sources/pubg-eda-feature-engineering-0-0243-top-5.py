#!/usr/bin/env python
# coding: utf-8

# # PUBG : EDA & Feature Engineering - 0.0243 (Top 5%)
# 
# ### *In this kernel, I will present the feature engineering I did for the PUBG placement ranking predictions.*
# #### So far, I've reached 0.0243 using those, which is top 5% approximately.
# 
# I have also played this game quite a lot, I will try to expose the things I learned playing that helped me design these features.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_palette("husl")
sns.set_style('white')


# ## Loading data

# In[ ]:


train_df = pd.read_csv("../input/train_V2.csv")


# In[ ]:


print("Size of train dataset : " + str(len(train_df)))


# In[ ]:


train_df.head()


# I can already separate columns in some categories :

# In[ ]:


features = ['assists', 'boosts', 'damageDealt', 'DBNOs', 'headshotKills', 'heals', 'killPlace', 'kills', 'killStreaks', 
            'longestKill', 'revives', 'rideDistance', 'roadKills', 'swimDistance', 'teamKills', 'vehicleDestroys', 'walkDistance', 'weaponsAcquired']
infos = ['matchDuration', 'matchType', 'maxPlace', 'numGroups']
ELO = ['rankPoints', 'killPoints', 'winPoints']
label = ['winPlacePerc']


# ### Correlation between features

# In[ ]:


sample = train_df.sample(100000)

f,ax = plt.subplots(figsize=(15, 12))
sns.heatmap(sample[ELO + features + label].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Considering the last column and my intuition, I can already detect features that I'll drop later :
# teamKills, swimDistance, roadKills, vehicleDestroys.
# 
# ELO variables also seem pretty useless. That is because matchmaking is based on those features, therefore people with similar Points will be put in the same game.
# However I have the intuition they can still be used. A first thing to do is to sum them into a single feature.

# ## Gametypes
# I merge gametypes to 5 categories : solo, duo, squad, crash and flare. 
# I create one column with indexes, that will be used for the model, and an other one with names, for visualisation purposes.

# In[ ]:


plt.figure(figsize=(15,10))
train_df['matchType'].value_counts().sort_values(ascending=False).plot.bar()
plt.show()


# In[ ]:


game_index_dic = {"solo": 1, "solo-fpp": 1, 'normal-solo': 1, "normal-solo-fpp": 1,
       "duo": 2, "duo-fpp": 2, 'normal-duo': 2,"normal-duo-fpp": 2,
       "squad": 3, "squad-fpp": 3, 'normal-squad': 3,"normal-squad-fpp": 3,
       "crashfpp" :4, "crashtpp" :4,
       "flarefpp" :5, "flaretpp":5
      }

game_name_dic = {"solo": "solo", "solo-fpp": "solo", 'normal-solo': "solo", "normal-solo-fpp": "solo",
                   "duo": "duo", "duo-fpp": "duo", 'normal-duo': "duo","normal-duo-fpp": "duo",
                   "squad": "squad", "squad-fpp": "squad", 'normal-squad': "squad","normal-squad-fpp": "squad",
                   "crashfpp": "crash", "crashtpp": "crash",
                   "flarefpp": "flare", "flaretpp": "flare"
      }


# In[ ]:


train_df['matchTypeName'] = train_df['matchType'].apply(lambda x: game_name_dic[x])
train_df['matchType'] = train_df['matchType'].apply(lambda x: game_index_dic[x])


# In[ ]:


f, ax = plt.subplots(figsize=(12, 8))
ax.set(yscale="log")
train_df['matchTypeName'].value_counts().sort_values(ascending=False).plot.bar(ax=ax)
plt.show()


# Flare games can easily be considered the same way as squad games, but crash games are a bit different.

# ## Played map
# 
# Indeed, there are 3 different maps in PUBG, which are of different size. The bigger the map, the longer the match, and the more distance players have to travel.

# In[ ]:


plt.figure(figsize=(12,8))
sns.distplot(train_df['matchDuration'])
plt.show()


# There are two Gaussians, I assume the first one corresponds to the "Mini Royale" mode played on the Sanhok map which is considerably smaller and faster paced. The second one corresponds to the normal maps.
# 
# We can create a feature saying whether we are in Mini Royale or not.

# In[ ]:


def get_map(x):
    if x > 1600:
        return 0
    else: 
        return 1


# In[ ]:


train_df['miniRoyale'] = train_df['matchDuration'].transform(get_map)


# In[ ]:


data = train_df
f, ax = plt.subplots(figsize=(12, 8))
ax.set(yscale="log")
sns.countplot(x='miniRoyale', hue='matchTypeName', ax=ax, data=data, palette=sns.color_palette(n_colors=5))
plt.show()


# Because they are shorter, crash games are labelled as Mini Royale, although they are not. But this gamemode is a bit special as mentionned before.

# ## Creating group level features
# 
# ### Group sizes 

# In[ ]:


agg = train_df.groupby(['matchId', 'groupId']).size().to_frame('teamSize')
train_df = train_df.merge(agg, how='left', on=['matchId', 'groupId'])


# In[ ]:


train_df.head()


# In[ ]:


data = train_df[['teamSize']].copy()
data.loc[data['teamSize'] >= 9] = '9+'
plt.figure(figsize=(15,10))
sns.countplot(data['teamSize'].astype('str').sort_values())
plt.show()


# Teams larger than 4 members ? Weird...

# ### It is important to consider the size of the team in different gamemodes
# 
# * Solos : 1 player
# * Duos : 2 players
# * Squad : 1 to 4 players
# * Crash : 1 to 4 players
# * Flaredrop : 1 to 4 players
# 
# A full squad will have more chances to win, obviously. 
# 
# The following dictionary precises the maximum number of players in a group depending on the gamemode.
# 
# 

# In[ ]:


size_dic = {1:1, 2:2, 3:4, 4:4, 5:4}


# #### But groups are indexed weirdly :

# In[ ]:


sample = train_df[train_df['matchType'] == 1]
plt.figure(figsize=(15, 6))
sns.countplot(sample['teamSize'])
plt.show()


# Teams in solos have up to 64 players, while they should all have 1 player.
# 
# Let us consider a single solo game : 

# In[ ]:


sample = train_df[train_df['matchId'] == "6dc8ff871e21e6"]
plt.figure(figsize=(10,4))
sns.countplot(sample['teamSize'])
plt.show()


# There are two players who have been incorrectly put in the same group.

# In[ ]:


fig,ax = plt.subplots(figsize=(12,8))
ax.set_xticklabels([])
sns.scatterplot(x='Id', y='winPlacePerc', hue='teamSize', data=sample, ax=ax)
plt.show()


# This plot let us notice that players in the same group have the same placement, despite the fact that they are not in the same team (because it's solos)

# > $  \text{matchTeamSize} = \max( \frac{\text{teamSize}}{\text{size(matchType)}} ,2) $

# In[ ]:


train_df['matchTeamSize'] = train_df['teamSize'] / train_df['matchType'].apply(lambda x: size_dic[x])


# With merged groups, this feature takes way too high values, therefore I threshold it at 2.

# In[ ]:


train_df['matchTeamSize'] = train_df['matchTeamSize'].apply(lambda x: max(2, x))


# ## Creating player level features

# ### PUBG is a skilled based game ...
# Here is a feature evaluating the skill of a player. It takes into consideration the fact that players teamkilling are usually trolling.
# > $ \text{skill} = \text{headshotKills} + 0.01 \cdot \text{longestKill} - \frac{\text{teamKills}}{\text{kills}+1}  $

# In[ ]:


train_df['skill'] = train_df['headshotKills'] + 0.01 * train_df['longestKill'] - train_df['teamKills']/(train_df['kills']+1)


# In[ ]:


data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='skill', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()


# ### A good player tends to aim for the head
# > $ \text{hsRatio} = \frac{\text{headshotKills}}{\text{Kills}} $

# In[ ]:


train_df['hsRatio'] = train_df['headshotKills'] / train_df['kills']
train_df['hsRatio'].fillna(0, inplace=True)


# In[ ]:


data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='hsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()


# Not very convinced with this one, as people who get a ratio of one are mostly the ones who only got one kill, it could be luck. The interesting part of this feature is in $ ] 0;0.5[  \cup  ] 0.5; 1 [ $. Therefore we modify it a bit.

# In[ ]:


def transform_hsRatio(x):
    if x == 1 or x == 0:
        return 0.5
    else: 
        return x


# In[ ]:


train_df['hsRatio'] = train_df['hsRatio'].apply(transform_hsRatio)


# In[ ]:


data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='hsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette("husl", n_colors=5))
plt.show()


# There we have it, a nice Gaussian-like repartition with skilled players on the right and less skilled players on the left !

# #### The more distance the travel, the longer you are supposed to stay alive
# 
# > $ \text{distance} = (\text{walkDistance} + 0.4 \cdot \text{rideDistance} + \text{swimDistance} ) \cdot \frac{1}{matchDuration} $

# It is important to normalize by the length of the game.

# In[ ]:


train_df['distance'] = (train_df['walkDistance'] + 0.4 * train_df['rideDistance'] + train_df['swimDistance'])/train_df['matchDuration']


# In[ ]:


data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='distance', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()


# #### The importance of boosts
# During end game, good players will always keep their boost jauge to the max. Movements are faster and it provides life regeneration. Furthermore, having a lot of boost means being well stuffed, therefore it increases the chances of winning. Therefore you want to be "full boost" when on foot.
# > $ \text{boostRatio} = \frac{\text{boosts}^2}{\sqrt{\text{walkDistance}}}$

# In[ ]:


train_df['boostsRatio'] = train_df['boosts']**2 / train_df['walkDistance']**0.5
train_df['boostsRatio'].fillna(0, inplace=True)
train_df['boostsRatio'].replace(np.inf, 0, inplace=True)


# In[ ]:


data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='boostsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()


# #### Healing means living
# > $ \text{healsRatio} = \frac{\text{heals}}{\text{matchDuration}^{0.1}}$

# In[ ]:


train_df['healsRatio'] = train_df['heals'] / train_df['matchDuration']**0.1
train_df['healsRatio'].fillna(0, inplace=True)
train_df['healsRatio'].replace(np.inf, 0, inplace=True)


# In[ ]:


data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='healsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()


# #### Good players go for kills
# > $ \text{killsRatio} = \frac{\text{kills}}{\text{matchDuration}^{0.1}}$

# In[ ]:


train_df['killsRatio'] = train_df['kills'] / train_df['matchDuration']**0.1
train_df['killsRatio'].fillna(0, inplace=True)
train_df['killsRatio'].replace(np.inf, 0, inplace=True)


# In[ ]:


data = train_df.sample(10000)
plt.figure(figsize=(15,10))
sns.scatterplot(x='killsRatio', y='winPlacePerc', hue='matchTypeName', data=data, palette=sns.color_palette(n_colors=5))
plt.show()


# ##### To be continued...
# >A lot more of clever features can be designed, but I'm gonna stick with these for now.

# In[ ]:


engineered = ['matchTeamSize', 'skill', 'hsRatio', 'distance', 'boostsRatio', 'healsRatio', 'killsRatio']


# ### Correlation map of our new features

# In[ ]:


sample = train_df.sample(100000)

f,ax = plt.subplots(figsize=(15, 12))
sns.heatmap(sample[engineered + label].corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# Focusing on the last line/column, there seems to be work to be done on the matchTeamSize and hsRatio feature. The first one is highly constrained by the problem with group ids ...

# # Following work
# 
# After the features engineering, I did some more preprocessing. I chose not to include the code, but here are the ideas :
# 
# ### Grouping by squads
# 
# As mentionned before, scores are the same for the each members of the team. Therefore it is clever to consider groups instead of players.
# 
# Depending on the pertinence of the feature, I am going to use the merge to increase the size of the feature space. 
# I usually take the maximum, mean, median and minimum in each group for the feature, except the ones that will (almost) always have 0 as minimum.
# Some data don't change in inbetween a group or a game, therefore only one way of merging is needed.
# 
# ### Normalizing when needed
# 
# I normalize (with a min/max scale) features that need to be.
# 
# ### Features based on rankings
# 
# I also increase the size of our feature space by considering the rank of the group inside its match for each feature. 

# ## * Feel free to ask me anything in the comments.*

# **Thanks for reading my work. As this is my first kernel, any advice is highly appreciated. **
# 
# **Don't forget to leave an upvote, it is always appreciated as well, as doing this kernel took me quite a lot of time.**
#  
# **Cheers, **
# 
# **Theo.**
