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


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


df = pd.read_csv('../input/fifa19/data.csv', encoding='utf-8')
df.head()


# In[ ]:


df = df.drop(columns=['Unnamed: 0'])
df.keys()


# In[ ]:


df.shape


# This is a relatively huge dataset.
# We begin our EDA with looking at the overall distribution of numbers, perhaps age, stats, value, etc.

# In[ ]:


for_age = df.copy()

plt.figure(figsize=(13,7))
plt.title("HOW OLD?")
sns.distplot(for_age['Age'])


# Overall Smooth Curve. 
# Concentration on 20s, leaving its long tail in late 30s and 40s.

# In[ ]:


for_overall = df.copy()

plt.figure(figsize=(13,7))
plt.title("TALENTS DIST")
sns.distplot(for_overall['Overall'])


# Scores follow Normal Distribution. Common sense confirmed.

# In[ ]:


for_money = df.copy()

for_money['Value'].head()


# In[ ]:


ending = [x['Value'][-1] for i, x in for_money.iterrows()]
set(ending)


# It's string, so we'll have to convert it into numbers. Endings are either 0, K, or M.

# In[ ]:


def value_cal(num):
    if num[-1] == '0':
        return 0
    elif num[-1] == 'K':
        return int(num[1:-1])
    else:
        return int(float(num[1:-1])*1000)


# In[ ]:


for_money['Money_Value'] = for_money['Value'].apply(value_cal)

plt.figure(figsize=(15,8))
plt.title("CUANTO CUESTA?")
sns.distplot(for_money['Money_Value'])


# Those of you who are fond of football may easily guess, but let's see some highest worth players.

# In[ ]:


bigones = for_money.sort_values(by='Money_Value', ascending=False)[['Name', 'Money_Value']][:15]
bigones


# ## Now take a look at some more specific stats

# In[ ]:


df.columns


# It's huge, so let's focus on field players for now.

# In[ ]:


stats = df[['Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle', 'SlidingTackle']].copy()


# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(stats.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


talent_hub = dict()
stats_ = stats.corr()
for ind, val in stats_.iterrows():
    talent_hub[ind] = val.sum()
    
talent_hub


# Perhaps a talent most central to soccer

# In[ ]:


[k for k,v in talent_hub.items() if v == max(talent_hub.values())][0]


# What about the talent most isolated from other soccer abilities

# In[ ]:


[k for k,v in talent_hub.items() if v == min(talent_hub.values())][0]


# What a surprise. Strength turned out to be least correlated with other soccer abilities. Ball control taking the first place well makes sense.

# ## STATS-MONEY
# 
# Now let's see how stats are correlated with monetary values.

# In[ ]:


for_money.columns


# In[ ]:


stats.columns


# In[ ]:


money = for_money.iloc[:500].copy()


# In[ ]:


print("For World Top 500 Players: ")
print()

for col in stats.columns:
    print(" {0}'s correlation with value: ".format(col), "{0:.2f}".format(np.corrcoef(np.array(money[col]), np.array(money['Money_Value']))[0][1]))


# In[ ]:




