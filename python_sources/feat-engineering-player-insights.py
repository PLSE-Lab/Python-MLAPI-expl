#!/usr/bin/env python
# coding: utf-8

# **PUBG**
# 
# This kernel does feature engineering and then runs a LightGBM model in order to produce a high score. Many of the engineered features are based on insights I have from playing this game.

# In[ ]:



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import warnings
warnings.filterwarnings("ignore")

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error
import gc

from sklearn.model_selection import GridSearchCV


# In[ ]:


train = pd.read_csv("../input/train_V2.csv")
test = pd.read_csv("../input/test_V2.csv")


# In[ ]:


# Thanks and credited to https://www.kaggle.com/gemartin who created this wonderful mem reducer
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.        
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    
    return df


# In[ ]:


concat = pd.concat([train, test])

#Delete train and test for now; this kernel is very memory hungry!
del train
del test
gc.collect()


# In[ ]:


concat = reduce_mem_usage(concat)


# In[ ]:


concat.head().T


# In[ ]:


concat.describe()


# In[ ]:


concat['typeCount'] = concat.groupby('matchType')['Id'].transform('size')
concat.head().T


# In[ ]:


concat['typeCountLog'] = concat['typeCount'].apply(np.log)


# In[ ]:


concat = concat.drop(['matchType', 'typeCount'], axis=1)


# It is helpful to know how many people are on each team and how many people are in each match in the data provided. These can also be used to build other features that we will see later.

# In[ ]:


def count_transform(df, cols):
    for c in cols:
        df[c + "_count"] = df.groupby(c)[c].transform('count')
    
    return df
        


# In[ ]:


concat = count_transform(concat, ['groupId', 'matchId'])


# **Player Experience - Two Main Strategies**
# 
# From my experience with the game, there are people that jump into hotly contested zones (e.g. prison and school areas) and face the action right away, and others that jump into sparcely populated areas where they are safe from other players (at least initially). To try and tease out these 2 different types of players, I track several stats divided by their distance moved. The idea being that people who acquired a lot of weapons without walking far probably dropped in a highly contested zone.

# In[ ]:


per_dist_stats = ['assists', 'boosts', 'damageDealt', 'DBNOs',
       'headshotKills', 'heals', 'kills',
       'teamKills', 'vehicleDestroys', 'weaponsAcquired']


# In[ ]:


concat['LogWalk'] = np.log1p(concat['walkDistance'])


# In[ ]:


for stat in per_dist_stats:
    concat[stat + '_perLogWalk'] = concat[stat] / concat['LogWalk']


# **Varying Team Sizes**
# 
# Interestingly, not all team sizes are the same. This next feature looks at the size of the team relative to the average team size for the match. (See, I said it would be helpful to know team sizes =D)

# In[ ]:


concat['grpSizeMult'] = concat['groupId_count'] / (concat['matchId_count'] / concat['numGroups'])


# **Match Relative Stats**
# 
# For this kernel, I am tracking relative stats for each match. A person may have a high winPoints, but that means less if everyone else in the match also has high winPoints.

# In[ ]:


match_stats = ['DBNOs',
 'assists',
 'boosts',
 'damageDealt',
 'headshotKills',
 'heals',
 'killPlace',
 'killPoints',
 'rankPoints',
 'killStreaks',
 'kills',
 'longestKill',
 'revives',
 'rideDistance',
 'roadKills',
 'swimDistance',
 'vehicleDestroys',
 'walkDistance',
 'weaponsAcquired',
 'winPoints',
 'LogWalk',
 'assists_perLogWalk',
 'boosts_perLogWalk',
 'damageDealt_perLogWalk',
 'DBNOs_perLogWalk',
 'headshotKills_perLogWalk',
 'heals_perLogWalk',
 'kills_perLogWalk',
 'teamKills_perLogWalk',
 'vehicleDestroys_perLogWalk',
 'weaponsAcquired_perLogWalk',]


# In[ ]:


for stat in match_stats:
    concat['matchRel_' + stat] = concat[stat] / (concat.groupby('matchId')[stat].transform('mean') + .001)


# In[ ]:


drop_features = ["winPlacePerc", "Id", "groupId", "matchId"]
feats = [c for c in concat.columns if c not in drop_features]


# In[ ]:


concat.to_csv('processed_concat.csv', index=False)


# **Aggregate at Group Level**
# 
# All members on a team get the same score, so it is helpful to aggregate the features at the group level. After we have our predictions, we can map them back to the original Ids for our submission.
# 

# In[ ]:


aggs = {
    'grpSizeMult' : ['mean'],
    'groupId_count' : ['mean'],
    'matchId_count' : ['mean'],
    'winPlacePerc' : ['mean'],
    'typeCountLog' : ['mean'],
}

for c in feats:
    if c not in aggs:
        aggs[c] = ['mean', 'min', 'max', 'std']
        
new_cols = [k + '_' + agg for k in aggs.keys() for agg in aggs[k]]


# In[ ]:


groups = concat.groupby('groupId').agg(aggs)
groups.columns = new_cols


# In[ ]:


del concat
gc.collect()


# In[ ]:


groups.to_csv('processed_groups.csv')


# Do you have any comments or questions? Please let me know!
