#!/usr/bin/env python
# coding: utf-8

# # Feature Engineering of PUBG data
# 
# We previously created a notebook going through some [exploratory anaylsys] of the PUBG data. We went through many of the different features avalailable and displayed an interesting plot describing the data and potential correlation with the target variable.
# 
# * We found that there was one missing value for the target variable and decided that this row of data should be removed, as there was only one player for the match identified by the missing value.
# 
# * We also made a few decisions about creating new features and one important way of breaking the data up to gain higher correllations with our features for seperate match types.
# 
# 
# [exploratory anaylsys]: https://www.kaggle.com/beaubellamy/pubg-eda#

# ## Import libraries
# We import the required libraries and import the data

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
sns.set()
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

import warnings
warnings.filterwarnings('ignore')


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')
test = pd.read_csv('../input/test_V2.csv')


# Lets check out the data again.

# In[ ]:


train.head()


# ## Missing Data
# Based on our EDA, we found a row that had a NULL value for teh target variable. We will remove the irrelevant row of data.

# In[ ]:


# Remove the row with the missing target value
train = train[train['winPlacePerc'].isna() != True]


# ## Lets Engineer some features
# ### PlayersJoined
# We can determine the number of players that joined each match by grouping the data by matchID and counting the players.

# In[ ]:


# Add a feature containing the number of players that joined each match.
train['playersJoined'] = train.groupby('matchId')['matchId'].transform('count')


# ### Normalised Features
# This feature now allows us to create more features based on other game features. This will allow us to normalise these features, as it might be easier to find an enemy when there are 100 players, than it is when there are 50 players.
# 

# In[ ]:


train['killsNorm'] = train['kills']*((100-train['playersJoined'])/100 + 1)
train['headshotKillsNorm'] = train['headshotKills']*((100-train['playersJoined'])/100 + 1)
train['killPlaceNorm'] = train['killPlace']*((100-train['playersJoined'])/100 + 1)
train['killPointsNorm'] = train['killPoints']*((100-train['playersJoined'])/100 + 1)
train['killStreaksNorm'] = train['killStreaks']*((100-train['playersJoined'])/100 + 1)
train['longestKillNorm'] = train['longestKill']*((100-train['playersJoined'])/100 + 1)
train['roadKillsNorm'] = train['roadKills']*((100-train['playersJoined'])/100 + 1)
train['teamKillsNorm'] = train['teamKills']*((100-train['playersJoined'])/100 + 1)
train['damageDealtNorm'] = train['damageDealt']*((100-train['playersJoined'])/100 + 1)
train['DBNOsNorm'] = train['DBNOs']*((100-train['playersJoined'])/100 + 1)
train['revivesNorm'] = train['revives']*((100-train['playersJoined'])/100 + 1)

    


# We can now remove the original features that the normalised ones are based on

# In[ ]:


# Features to remove
train = train.drop([ 'kills', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks', 
 'longestKill', 'roadKills', 'teamKills', 'damageDealt', 'DBNOs', 'revives'],axis=1)


# In[ ]:


train.head()


# ### TotalDistance
# An additional feature we can create is the total distance the player travels. This is a combination of all the distance features in the original data set.

# In[ ]:


# Total distance travelled
train['totalDistance'] = train['walkDistance'] + train['rideDistance'] + train['swimDistance']


# # Standardize the matchType feature
# Here we decided that many of the existing 16 seperate modes of game play were just different versions of four types of game.
# 
# 1. Solo: Hunger Games style, last man/women standing.
# 2. Duo: Teams of two against all other players.
# 3. Squad: Teams of up to 4 players against All other players
# 4. Other: These modes consist of custom and special events modes

# In[ ]:


# Normalise the matchTypes to standard fromat
def standardize_matchType(data):
    data['matchType'][data['matchType'] == 'normal-solo'] = 'Solo'
    data['matchType'][data['matchType'] == 'solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-solo-fpp'] = 'Solo'
    data['matchType'][data['matchType'] == 'normal-duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'normal-duo'] = 'Duo'
    data['matchType'][data['matchType'] == 'duo-fpp'] = 'Duo'
    data['matchType'][data['matchType'] == 'squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad'] = 'Squad'
    data['matchType'][data['matchType'] == 'normal-squad-fpp'] = 'Squad'
    data['matchType'][data['matchType'] == 'flaretpp'] = 'Other'
    data['matchType'][data['matchType'] == 'flarefpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashtpp'] = 'Other'
    data['matchType'][data['matchType'] == 'crashfpp'] = 'Other'

    return data


data = standardize_matchType(train)
#print (set(data['matchType']))


# In[ ]:


# We can do a sanity check of the data, making sure we have the new 
# features created and the matchType feature is standardised.
data.head()


# ## Seperate the data
# Here, we will create four seperate data sets which describes the matchType.

# In[ ]:


# Seperate the data into the matchTypes
solo = data[data['matchType'] == 'Solo']
duo = data[data['matchType'] == 'Duo']
squad = data[data['matchType'] == 'Squad']
other = data[data['matchType'] == 'Other']


# ## Feature Selection
# Here we use our previous EDA to determine the list of features that we want to keep for each data set.

# In[ ]:


# SOLO: Features to keep
solo_features = ['boosts','heals', 'rideDistance','walkDistance','weaponsAcquired',
                 # Engineered Features
                 'damageDealtNorm','headshotKillsNorm','killPlaceNorm',
                 'killsNorm','killStreaksNorm','longestKillNorm',
                 'playersJoined','totalDistance']

solo = solo[solo_features]
solo.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(solo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


# DUO: Features to keep
duo_features = ['assists','boosts', 'heals','rideDistance','walkDistance',
                'weaponsAcquired',
                # Engineered Features
                'damageDealtNorm','DBNOsNorm', 'killPlaceNorm',
                'killsNorm','killStreaksNorm','longestKillNorm',
                'revivesNorm', 'playersJoined','totalDistance']

duo = duo[duo_features]
duo.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(duo.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


# SQUAD: Features to keep
squad_features = ['assists','boosts','heals','rideDistance',
                  'walkDistance','weaponsAcquired',
                  # Engineered Features
                  'damageDealtNorm','DBNOsNorm', 'killPlaceNorm',
                  'killsNorm','killStreaksNorm','longestKillNorm',
                  'revivesNorm','playersJoined','totalDistance']

squad = squad[squad_features]
squad.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(squad.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# In[ ]:


# OTHER: Features to keep
other_features = ['assists','boosts','heals','rideDistance',
                  'walkDistance','weaponsAcquired',
                  # Engineered Features
                  'damageDealtNorm','DBNOsNorm','headshotKillsNorm',
                  'killPlaceNorm','killsNorm','killStreaksNorm','longestKillNorm',
                  'revivesNorm','playersJoined','totalDistance']

other = other[other_features]
other.head()


# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(other.corr(), annot=True, linewidths=.5, fmt= '.2f',ax=ax)
plt.show()


# # Model Development
# This is were we will develop our machine learning model

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# If you liked this post, please upvote.
