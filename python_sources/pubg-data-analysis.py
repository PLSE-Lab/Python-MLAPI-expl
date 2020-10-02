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
import itertools
import warnings
warnings.filterwarnings("ignore")


# In[ ]:


Train_df =  pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/train_V2.csv")
test_df = pd.read_csv("/kaggle/input/pubg-finish-placement-prediction/test_V2.csv")


# In[ ]:


Train_df


#  # before reduce Data

# In[ ]:


Train_df.info() 


# In[ ]:


print("TRAIN ",Train_df.memory_usage().sum()/1024**2)
print("TEST ",test_df.memory_usage().sum()/1024**2)


# In[ ]:


"""
Check Data Type 
"""
col = Train_df['assists'].dtype
type(col)


# In[ ]:


'''
All type of min and max value in INT & floar
'''
print('INT8 ',np.iinfo(np.int8))
print('INT16 ',np.iinfo(np.int16))
print('INT32 ',np.iinfo(np.int32))
print('INT64 ',np.iinfo(np.int64))
print('FLOAT32 ',np.finfo(np.float32))
print('FLOAT64 ',np.finfo(np.float64))


# # Function for reduce Data

# In[ ]:


"""
Reduce Memory function
"""
def reduce_memory(df):
    total_memoryIN_mebi = df.memory_usage().sum()/1024**2 # Convert Bytes to Mebibyte
    
    for col in df.columns: # get column one by one
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()  # min value of column
            c_max = df[col].max() # max value of column
            
            if str(col_type)[:3] == 'int': # convert numpy.dtype to string
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    after_reduce = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB --> {:.2f} MB (Decreased by {:.1f}%)'.format(
        total_memoryIN_mebi, after_reduce, 100 * (total_memoryIN_mebi - after_reduce) / total_memoryIN_mebi))
    return df


# # After Redduce Data

# In[ ]:


get_ipython().run_cell_magic('time', '', '\ntrain = reduce_memory(Train_df)\ntest = reduce_memory(test_df)\nprint(train.shape,test.shape)')


# In[ ]:


train.info()


# In[ ]:


train.columns


# - **killPlace** - Your ranking in match in terms of number of enemy players killed.
# - **killPoints** - Kills-based external ranking of player. (Ranking where only winning matters).
# - **kills** - Number of enemy players killed.
# - **killStreaks** - Max number of enemy players killed in a short amount of time. A Killstreak is earned when a player acquires a certain number of kills in a row without dying.
# - **longestKill** - Longest distance between player and player killed at time of death. This may be misleading, as downing a - player and driving away may lead to a large longestKill stat.
# - **maxPlace** - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# - **numGroups** - Number of groups we have data for in the match.
# - **revives** - Number of times you revived your teammates.
# - **rideDistance** - Total distance traveled in vehicles (measured in meters).
# - **roadKills** - Number of enemy killed while travelling in a vehicle.
# - **swimDistance** - Total distance traveled by swimming (measured in meters).
# - **teamKills** - Number of times you are killed your teammate.
# - **vehicleDestroys** - Number of vehicles destroyed.
# - **walkDistance** - Total distance traveled on foot (measured in meters).
# - **weaponsAcquired** - Number of weapons picked up.
# - **winPoints** - Win-based external ranking of player. (Ranking where only winning matters).
# - **winPlacePerc** - The target of prediction **(Target Variable)**. This is a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match.

# #Unique Count of ID,Group, MatchID
# 1. Why unique because of rejoining player

# In[ ]:


for i in ['Id','groupId','matchId']:  # Name in List formate
    print(f'unique [{i}] count:', train[i].nunique()) #Getting Unique Data from Data Sets


# ### Exploring Different Match Type
# PUBG offers 3 different game modes:
# - Solo - One can play alone (solo,solo-fpp,normal-solo,normal-solo-fpp)
# - Duo - Play with a friend (duo,duo-fpp,normal-duo,normal-duo-fpp,crashfpp,crashtpp)
# - Squad - Play with 4 friends (squad,squad-fpp,normal-squad,normal-squad-fpp,flarefpp,flaretpp)

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(12, 4))


# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(30,6))
train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[0])

mapper = lambda x: 'solo' if ('solo' in x) else 'duo' if ('duo' in x) or ('crash' in x) else 'squad'
train['matchType'] = train['matchType'].apply(mapper)
train.groupby('matchId')['matchType'].first().value_counts().plot.bar(ax=ax[1])


# ###Player Analysis
# - players in match and group
# 
# During a game, 100 players join the same server,  so in the case of duos the max teams are 50 and in the case of squads the max teams are 25.

# In[ ]:


def mergeList(list1,list2):
    return list(itertools.product(list1,list2))
match = train.groupby(['matchType','matchId']).size().to_frame('players in match')
group = train.groupby(['matchType','matchId','groupId']).size().to_frame('players in group')
pd.concat([match.groupby('matchType').describe()[mergeList(['players in match'],['min','mean','max'])], 
           group.groupby('matchType').describe()[mergeList(['players in group'],['min','mean','max'])]], axis=1)


# In[ ]:


print(group['players in group'].nlargest())


# In[ ]:


''' ex) matchId=='3e029737889ce9', groupId=='b8275198faa03b'
'''
subset = train[train['matchId']=='3e029737889ce9']
sub_grp = subset[subset['groupId']=='b8275198faa03b']

print('matchId ==\'3e029737889ce9\' & groupId ==\'b8275198faa03b\'')
print('-'*50)
print('players:',len(subset))
print('groups:',subset['groupId'].nunique())
print('numGroups:',subset['numGroups'].unique())
print('maxPlace:',subset['maxPlace'].unique())
print('-'*50)
print('max-group players:',len(sub_grp))
print('max-group winPlacePerc:',sub_grp['winPlacePerc'].unique())
print('-'*50)
print('winPlacePerc:',subset['winPlacePerc'].sort_values().unique())


# In[ ]:


corr = train.corr()
f,ax = plt.subplots(figsize=(20, 15))
sns.heatmap(train.corr(), annot=True, fmt= '.1f',ax=ax, cmap="BrBG")
sns.set(font_scale=1.25)
plt.show()


# In[ ]:




