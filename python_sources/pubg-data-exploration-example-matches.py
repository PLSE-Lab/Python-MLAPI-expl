#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/train_V2.csv')

data.loc[:5,]


# **TLD;DR**
# 
# Each match may not follow the limit set my the mode e.g. there are groups of 7 in a squad match when the limit is group of 4.
# There may also be missing groups for each match as numGroup does not equal to maxPlace.

# I want to take a look at an example solo match.

# In[ ]:


data_solo_example = data.loc[data.matchId == data.loc[data.matchType=='solo','matchId'].iloc[0],:].sort_values(by='winPlacePerc',ascending=False)
print(data_solo_example)

print('This example has {} players.'.format(len(data_solo_example.Id)))
print('This example has {} groups.'.format(len(data_solo_example.groupId.unique())))


# This solo match has 92 players but with 89 groups. This suggests that some players are in groups.

# I went to investigate if the difference between consecutive winPlacePerc is consistent.

# In[ ]:


[x-data_solo_example.winPlacePerc.iloc[i+1] for i, x in enumerate(data_solo_example.winPlacePerc) if i < len(data_solo_example.winPlacePerc)-1]


# From the above calculation, I can see that it is not exactly consistent. 
# 
# I also found out that two players had the same winPlacePerc. Both players are also in the same team.
# 
# Let's a look at groupId count of more than 1.

# In[ ]:


data_solo_example.loc[data_solo_example.groupId.isin(
    list(data_solo_example.groupId.value_counts().loc[data_solo_example.groupId.value_counts()>1].index
        ))]


# The table shows that these players seems to be unique. Therefore, there might have been data collection errors or cheating involved.

# In[ ]:


data_duo_example = data.loc[data.matchId == data.loc[data.matchType=='duo','matchId'].iloc[0],:].sort_values(by='winPlacePerc',ascending=False)
print(data_duo_example)

print('This example has {} players.'.format(len(data_duo_example.Id)))
print('This example has {} groups.'.format(len(data_duo_example.groupId.unique())))


# In[ ]:


data_duo_example.maxPlace


# This duo example match shows that maxPlace need not be the same as numGroup.
# 
# Again, there are groups with number of players exceeding the duo limit.

# In[ ]:


data_duo_example.loc[data_duo_example.groupId.isin(
    list(data_duo_example.groupId.value_counts().loc[data_duo_example.groupId.value_counts()>2].index
        ))]


# Let's take a look at an example squad match.

# In[ ]:


data_squad_example = data.loc[data.matchId == data.loc[data.matchType=='squad','matchId'].iloc[0],:].sort_values(by='winPlacePerc',ascending=False)
print(data_squad_example)

print('This example has {} players.'.format(len(data_squad_example.Id)))
print('This example has {} groups.'.format(len(data_squad_example.groupId.unique())))


# In[ ]:


data_squad_example.maxPlace


# This squad example match also shows that maxPlace need not be the same as numGroup.
# 
# Again, there are groups with number of players exceeding the squad limit.

# In[ ]:


data_squad_example.loc[data_squad_example.groupId.isin(
    list(data_squad_example.groupId.value_counts().loc[data_squad_example.groupId.value_counts()>4].index
        ))]


# **Conclusion**
# 
# The dataset shows that not all matches are following the limits of group size for each match. Also there may have been missing groups in each match as numGroup does not equal to maxPlace.
# 

# In[ ]:




