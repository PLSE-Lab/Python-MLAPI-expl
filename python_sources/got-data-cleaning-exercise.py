#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


main_gun_data = pd.read_csv('../input/gun-violence-data/gun-violence-data_01-2013_03-2018.csv')
main_gun_data.head()


# In[3]:


main_gun_data.shape


# In[4]:


main_gun_data.isnull().sum()


# In[5]:


game_of_thrones = pd.read_csv('../input/game-of-thrones/battles.csv')
game_of_thrones.head(10)


# In[6]:


game_of_thrones.shape


# In[7]:


game_of_thrones.isnull().sum()


# In[8]:


character_deaths = pd.read_csv('../input/game-of-thrones/character-deaths.csv')
character_deaths.head()


# In[9]:


character_deaths.shape
character_deaths.info()


# In[10]:


character_deaths.isnull().sum()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as pyplot
get_ipython().run_line_magic('matplotlib', 'inline')

# len(game_of_thrones)
wars_missing_percentage = (game_of_thrones.isnull().sum()/len(game_of_thrones))*100
# wars_missing_percentage[wars_missing_percentage == 0] 
wars_missing_percentage = wars_missing_percentage.drop(wars_missing_percentage[wars_missing_percentage==0].index).sort_values(ascending=False)
wars_missing_percentage


# In[12]:


game_of_thrones['attacker_king'].value_counts()


# In[13]:


game_of_thrones['defender_king'].value_counts()


# In[14]:


fig,ax = pyplot.subplots(figsize=(12,8))
pyplot.xticks(rotation='90')
sns.barplot(x=wars_missing_percentage.index,y=wars_missing_percentage)
pyplot.show()


# In[15]:


#drop columns where missing_values are atleast 60% of data
thresh = len(game_of_thrones)*.4
game_of_thrones_dropped = game_of_thrones.dropna(thresh=thresh,axis=1,inplace=True)
game_of_thrones.head()


# In[16]:


game_of_thrones['attacker_size'].fillna(game_of_thrones['attacker_size'].mean(),inplace=True)
game_of_thrones['defender_size'].fillna(game_of_thrones['defender_size'].mean(),inplace=True)


# In[17]:


#We will also specify that any missing value which does not have any value directly under it will be filled with 0
game_of_thrones['summer'] = game_of_thrones['summer'].fillna(method='bfill').fillna(0)


# In[18]:


game_of_thrones['attacker_king']=game_of_thrones['attacker_king'].fillna(game_of_thrones['attacker_king'].mode()[0])
game_of_thrones['defender_king']=game_of_thrones['defender_king'].fillna(game_of_thrones['defender_king'].mode()[0])
game_of_thrones['attacker_commander']=game_of_thrones['attacker_commander'].fillna(game_of_thrones['attacker_commander'].mode()[0])
game_of_thrones['defender_commander']=game_of_thrones['defender_commander'].fillna(game_of_thrones['defender_commander'].mode()[0])
game_of_thrones['location']=game_of_thrones['location'].fillna(game_of_thrones['location'].mode()[0])


# In[19]:


game_of_thrones['defender_king'].mode()[0]


# In[20]:


game_of_thrones.info()


# In[21]:


defender_king = game_of_thrones[game_of_thrones['defender_1'].isnull()]['defender_king']
defender_king.values[0]


# In[22]:


count=1
for name, group in game_of_thrones.groupby(['defender_king', 'defender_1']):
    if name[0] == defender_king.values[0]:
            if group.shape[0]>=count:
                count=group.shape[0]
                defender_1 = name[1]
                
print(defender_1)
game_of_thrones.loc[game_of_thrones['defender_1'].isnull(),'defender_1']=defender_1

