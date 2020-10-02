#!/usr/bin/env python
# coding: utf-8

# <h1><center>Improving predictions using winPlacePerc's definition</center></h1>

# The <a href="https://www.kaggle.com/c/pubg-finish-placement-prediction/data">competition data definition says</a>  that the target we are to predict, winPlacePerc, is "a percentile winning placement, where 1 corresponds to 1st place, and 0 corresponds to last place in the match. It is calculated off of maxPlace, not numGroups, so it is possible to have missing chunks in a match"
# 
# More precisely, in <a href="https://www.kaggle.com/c/pubg-finish-placement-prediction/discussion/67742">a discussion</a>, Michael Apers from kaggle team gives the following formula: "winPlacePerc is calculated by winPlacePerc = (maxPlace-winPlace)/(maxPlace-1), where winPlace is the player's team's placement in the game (1 for first, 2 for second, and so on)"
# 
# Tl;dr = indeed winPlacePerc = (maxPlace-winPlace)/(maxPlace-1).  As a side benefit, we have spooted what to predict for winPlacePerc when numGroups = 1 or 2. 
# 

# In[ ]:


import numpy as np 
import pandas as pd 
import gc


# In[ ]:


train = pd.read_csv('../input/train_V2.csv')
train.drop(train.columns.difference(['matchId','groupId','maxPlace','numGroups','winPlacePerc']),axis=1,inplace=True)


# # Initial check: do all Ids with same groupId have the same winPlacePerc?

# In[ ]:


init_check = train .groupby(['matchId','groupId'])[['winPlacePerc']].nunique()
init_check.head()
print(init_check[['winPlacePerc']].max())


# All good: all members of a same group have the same winPlacePerc so we can keep one member per groupId and discard the rest to work on a smaller dataset

# In[ ]:


print('Train # of samples before dropping duplicates {:,}'.format(train.shape[0]))
train.drop_duplicates(subset=['matchId','groupId'],inplace=True)
print('Train # of samples after dropping duplicates {:,}'.format(train.shape[0]))


# Now let us check the winPlacePerc formula mentioned by Michael Apers in two situations:
# 1. when maxPlace = numGroups
# 2. when maxPlace != numGroups
# 
# Indeed, in the data defintion we find: 
# numGroups - Number of groups we have data for in the match. 
# maxPlace - Worst placement we have data for in the match. This may not match with numGroups, as sometimes the data skips over placements.
# 

# In[ ]:


train_same = train[train['maxPlace'] == train['numGroups']].copy()
train_diff = train[train['maxPlace'] != train['numGroups']].copy()
del train
gc.collect();


# # Formula check when maxPlace = numGroup

# We check if there are any winPlacePerc missing values in this case... and indeed there is one, so we remove it.

# In[ ]:


print(train_same['winPlacePerc'].isnull().sum())
train_same.dropna(subset = ['winPlacePerc'], inplace = True)
print(train_same['winPlacePerc'].isnull().sum())


# In[ ]:


print('# of samples for which maxPlace = numGroups: {:,}'.format(train_same.shape[0]))


# Let us define winPlaceRank = (maxPlace-winPlace)
# 
# Now we can check that, the winPlacePerc formula holds by looking at whether winPlaceRank computed as winPlacePerc x (maxPlace-1) is indeed an integer up to the second decimal place
# 
# Indeed according to the formula winPlacePerc = (maxPlace-winPlace)/(maxPlace-1) 

# In[ ]:


train_same['winPlaceRank'] = train_same['winPlacePerc'] * (train_same['maxPlace'] - 1)
train_same['winPlaceRank_frac'] = np.mod(np.around(train_same['winPlaceRank'] * 100).astype(int),100)
train_same['winPlaceRank_frac'].value_counts().sort_values()


# All good: we have just shown that maxPlace-1 is indeed the denominator of winPlacePerc.
# 
# To be thorough, we check that all winPlaceRank are indeed between 0 and maxPlace - 1 and it is the case

# In[ ]:


train_same[ (train_same['winPlaceRank'] > (train_same['maxPlace'] - 1)) | (train_same['winPlaceRank'] < 0) ]


# # Formula check when maxPlace != numGroup
# 
# Let us start by finding whether it can happen that maxPlace < numGroups

# In[ ]:


train_diff[train_diff['maxPlace'] < train_diff['numGroups']]


# Ok, it cannot so now let us look at what happens for small groups

# In[ ]:


train_diff['numGroups'].min()


# In[ ]:


train_diff[train_diff['numGroups'] == 1]['winPlacePerc'].unique()


# In[ ]:


np.sort(train_diff[train_diff['numGroups'] == 1]['maxPlace'].unique())


# So in case numGroups = 1, the group is always considered to have lost, no matter how much maxPlace is. Per se, this is not incoherent with the winPlacePerc formula, just a point to keep in mind

# In[ ]:


train_diff[train_diff['numGroups'] == 2]['winPlacePerc'].unique()


# In[ ]:


np.sort(train_diff[train_diff['numGroups'] == 2]['maxPlace'].unique())


# So in case numGroups = 2, one group is always considered to have lost and the other to have won, no matter how much maxPlace is. Per se, this is also not incoherent with the winPlacePerc formula, just a point to keep in mind.
# 
# No let us focus on all other cases

# In[ ]:


#Renaming train_diff to make for shorter code

train = train_diff.copy()
del train_diff
gc.collect();


# In[ ]:


train = train[train['numGroups'] > 2]
print('# of samples we are going to examine: {:,}'.format(train.shape[0]))


# In[ ]:


train.head()


# Our strategy to check the formula shall be to find "the" denominator that makes the gaps between winPlacePerc values of a same matchId be integer fractions

# In[ ]:


train.sort_values(by=['matchId','winPlacePerc'],inplace=True)

winPlacePerc_gaps = train.groupby('matchId')[['winPlacePerc']].diff()
winPlacePerc_gaps.dropna(inplace=True)
winPlacePerc_gaps = winPlacePerc_gaps.join(train[['matchId','maxPlace','numGroups']],how='left')
winPlacePerc_gaps['winPlacePerc'] = np.around(winPlacePerc_gaps['winPlacePerc'] * 10000).astype(int)

winPlacePerc_gaps_agg = winPlacePerc_gaps.groupby('matchId').agg({'winPlacePerc' : lambda x: set(x), 'maxPlace' : 'mean', 'numGroups' : 'mean'})
winPlacePerc_gaps_agg.head()


# In[ ]:


winPlacePerc_gaps_agg['winPlacePerc'] = winPlacePerc_gaps_agg['winPlacePerc'].map(lambda x: np.sort(np.array(list(x))))
winPlacePerc_gaps_agg['winPlacePerc'] = 10000 / winPlacePerc_gaps_agg['winPlacePerc']
winPlacePerc_gaps_agg['winPlacePerc'] = winPlacePerc_gaps_agg['winPlacePerc'].map(lambda x: set(np.around(x)))
winPlacePerc_gaps_agg.head()


# Now we look at which of maxPlace-1 or numGroups-1 seem to be the denominator

# In[ ]:


winPlacePerc_gaps_agg['maxPlace_vote'] = winPlacePerc_gaps_agg.apply(lambda x: (x['maxPlace'] - 1) in x['winPlacePerc'], axis=1)
winPlacePerc_gaps_agg['numGroups_vote'] = winPlacePerc_gaps_agg.apply(lambda x: (x['numGroups'] - 1) in x['winPlacePerc'], axis=1)


# In[ ]:


print('# of samples we are examining: {:,}'.format(train.shape[0]))
print('# of samples for which maxPlace may not work: {:,}'.format(winPlacePerc_gaps_agg[~winPlacePerc_gaps_agg['maxPlace_vote']].shape[0]))
print('# of samples for which numGroups may not work: {:,}'.format(winPlacePerc_gaps_agg[~winPlacePerc_gaps_agg['numGroups_vote']].shape[0]))
print('# of samples for which both maxPlace and numGroups work: {:,}'.format(winPlacePerc_gaps_agg[winPlacePerc_gaps_agg['numGroups_vote'] & winPlacePerc_gaps_agg['maxPlace_vote']].shape[0]))


# In[ ]:


winPlacePerc_gaps_agg[~winPlacePerc_gaps_agg['maxPlace_vote']]


# In[ ]:


train[train['matchId'] == '668560ba6622c2']


# As Michael Apers kindly pointed out, for this particular matchId 668560ba6622c2, (maxPlace - 1) still works because 4 = 2 x 2 (so in terms of winPlacePerc Here 1.0 = (5-1)/(5-1); 0.5 = (5-3)/(5-1); 0.0 = (5-5)/(5-1) to quote Michael). So although, (maxPlace -1) is not the denominator of the smallest gap in winPlacePerc for this match, it is a divisor of this denominator

# In[ ]:


train[train['numGroups'] == 3].sort_values(by=['matchId','winPlacePerc'])


# # And what about the test set?

# In[ ]:


test = pd.read_csv('../input/test_V2.csv')


# In[ ]:


print('# of samples in the test set: {:,}'.format(test.shape[0]))
print('# of samples in the test set for which numGroups is less than or equal to 3: {:,}'.format(test[test['numGroups'] <= 3].shape[0]))


# In[ ]:


test[test['numGroups'] <=3].groupby('numGroups')['groupId'].size()

