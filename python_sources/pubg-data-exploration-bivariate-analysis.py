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


# In[ ]:


data.loc[:5,]


# **TL;DR**
# 
# Attention should be given to groupId with more than the actual limit (4 for squad mode, 2 for duo mode).
# There are 4 variables that have higher correlation with winPlacePerc.
# There are also some highly correlated variables e.g. winPoints and killPoints.

# **Categorical variables**
# 
# Let's explore some relationships between categorical variables.

# In[ ]:


data_squad_fpp = data.loc[data.matchType=='squad-fpp']

print(data_squad_fpp.groupId.value_counts().describe())

data_squad_fpp.groupId.value_counts().plot(kind='box')


# In[ ]:


data_squad = data.loc[data.matchType=='squad']

print(data_squad.groupId.value_counts().describe())

data_squad.groupId.value_counts().plot(kind='box')


# It seems strange that there are groupId with more than 4 occurences because the squad limit is only 4.

# In[ ]:


data_duo_fpp = data.loc[data.matchType=='duo-fpp']

print(data_duo_fpp.groupId.value_counts().describe())

data_duo_fpp.groupId.value_counts().plot(kind='box')


# In[ ]:


data_duo = data.loc[data.matchType=='duo']

print(data_duo.groupId.value_counts().describe())

data_duo.groupId.value_counts().plot(kind='box')


# Similarly, it is strange that there are groupId with more than 2 occurences because the duo limit is only 2.

# In[ ]:


data_solo_fpp = data.loc[data.matchType=='solo-fpp']

print(data_solo_fpp.groupId.value_counts().describe())

data_solo_fpp.groupId.value_counts().plot(kind='box')


# In[ ]:


data_solo = data.loc[data.matchType=='solo']

print(data_solo.groupId.value_counts().describe())

data_solo.groupId.value_counts().plot(kind='box')


# Again, it is strange to have repeated groupId in solo games. Perhaps, groupId are being reused.

# In[ ]:


data_others = data.loc[(data.matchType!='solo')|(data.matchType!='solo-fpp')|(data.matchType!='duo')|(data.matchType!='duo-fpp')|(data.matchType!='squad')|(data.matchType!='squad-fpp')]

print(data_others.groupId.value_counts().describe())

data_others.groupId.value_counts().plot(kind='box')


# I want to know how many groups there are for every match.

# In[ ]:


data_match_group_player = data.groupby('matchId').groupId.nunique()
data_match_group_player


# In[ ]:


data_match_group_player.describe()

data_match_group_player.plot(kind='box')

print('The median number of groups per match is {}.'.format(data_match_group_player.median()))


# In[ ]:


data.groupby('groupId').matchId.nunique().sort_values(ascending=False)


# groupId is not reused in different matches. Therefore, some groupId is deemed to have more than 4 or 2 players in squad and duo mode respectively.

# **Continuous Variable**
# 
# Let's do a correlation matrix of continuous variables.

# In[ ]:


data_continuous = data.loc[:,[data.columns[i] for i, x in enumerate(data.dtypes) if x != 'object']]
data_continuous.iloc[:10]


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

data_continuous_corr = data_continuous.corr()

# Generate a mask for the upper triangle
mask = np.zeros_like(data_continuous_corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 11))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(data_continuous_corr, mask=mask, cmap=cmap, vmax=.3, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})


# From the correlation matrix above, you can see that the most obvious relationships are between rankPoints and killPoints, and rankPoints and winPoints. This is to be expected as the are closely related in the game.
# 
# Let's take a closer look at the relationship between winPlacePerc, the dependent variables, and the other independent variables.

# In[ ]:


data_continuous_corr.loc['winPlacePerc',:].sort_values()

print('Variables with correlation greater than 0.5, or less than -0.5, with winPlacePerc are {}.'.format([data_continuous_corr.index[i] for i, x in enumerate(data_continuous_corr.loc['winPlacePerc',:]) if abs(x) > 0.5 ]))


# In[ ]:


high_corr_var = {}
for x in data_continuous_corr.index:
    for y in data_continuous_corr.columns:
        if abs(data_continuous_corr.loc[x,y]) > 0.5:
            high_corr_var[(x,y)] = data_continuous_corr.loc[x,y]
pd.DataFrame([x for x in high_corr_var.values()], index=high_corr_var.keys(), columns = ['Correlation'])


# From the table above, you might notice that there are some variables are highly correlated to each other, e.g. winPoints and killPoints correlation is 0.9834.

# In[ ]:


print('When winPoints = 0, killPoints are also {}.'.format(
    sum(data_continuous.loc[data_continuous.winPoints==0, 'killPoints'])))

print('When winPoints = 0, the minimum rankPoints is {}.'.format(
    min(data_continuous.loc[data_continuous.winPoints==0, 'rankPoints'])))


# In[ ]:


print('When killPoints = 0, winPoints are also {}.'.format(
    sum(data_continuous.loc[data_continuous.killPoints==0, 'winPoints'])))

print('When killPoints = 0, the minimum rankPoints is {}.'.format(
    min(data_continuous.loc[data_continuous.killPoints==0, 'rankPoints'])))


# In[ ]:


print('When rankPoints = -1, the minimum winPoints and killPoints is {} and {} respectively.'.format(
    min(data_continuous.loc[data_continuous.rankPoints==-1, 'winPoints']),
    min(data_continuous.loc[data_continuous.rankPoints==-1, 'killPoints'])
))


# In[ ]:


print('When rankPoints > 0, the minimum winPoints and killPoints is {} and {} respectively.'.format(
    min(data_continuous.loc[data_continuous.rankPoints>0, 'winPoints']),
    min(data_continuous.loc[data_continuous.rankPoints>0, 'killPoints'])
))


# From the findings above, we know that when winPoints and killPoints are non-zero, there will be rankPoints.
# When rankPoints is either -1 or 0, there will be a non-zero winPoints and killPoints.
# There are no cases where rankPoints, winPoints and killPoints are 'non-zero' together.

# **Conclusion**
# 
# With my[ univariate analysis](https://www.kaggle.com/teemingyi/pubg-data-exploration-univariate-analysis) and my bivariate analysis, I will be able to conduct some data cleaning before building a model for prediction.

# In[ ]:




