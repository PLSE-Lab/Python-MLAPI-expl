#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


heroes_data = pd.read_csv("/kaggle/input/dota-heroes/p5_training_data.csv")


# Dota2 is a 5vs5 MOBA (multiplayer online battle arena) game. There are lots of heroes in this game to pick everytime the game starts. Lets see how many heroes exist. But first, we need to take a look at the columns of our data.

# In[ ]:


print(heroes_data.columns)


# There is a column called 'name' so we can arrange the heroes using their names.

# In[ ]:


name_series = heroes_data['name']
print(name_series)


# In[ ]:


hero_count = len(name_series)
print("Hero Count: {}".format(hero_count))


# As we can see, there are 99 heroes in this game. Of course every of them has their very own special abilities to win the fight. Some of them are deadly fighters, some of them are good at defending their team mates etc. But basicly there are 3 types of heroes. Strength, agility and intelligence. Not all of them but generally strength heroes are durable fighters and can survive longer according to other heroes. Agility heroes have high DPS (damage per second) points with their fast attacks and moves. So generally the category 'carry heroes' (they carry the game to win the game as they are ultimate killers) are mostly chosen from agility heroes. Intelligence heroes are good at magic skills so most of them are 'support heroes' (they support their teammates, heal them etc. to take advantage for their team at team fights).
# 
# Every hero has both three of the stats. An intelligence hero can be surprisingly more durable than a strength character. But mostly, we don't see this situation.
# 
# So when we think a bit, we can see that generally 'strength heroes' should have high health stats ; 'Agility heroes' should have high attack speed and 'Intelligence heroes' should have high mana points. Lets analyse this.

# In[ ]:


print(heroes_data.loc[:,"name":"type"])


# Type 0 is 'Strength',
# 
# Type 1 is 'Agility' and
# 
# Type 2 is 'Intelligence' in this list.
# 
# Lets use pandas' filter method to categorize these.

# In[ ]:


strength_filter = heroes_data["type"] == 0
agility_filter = heroes_data["type"] == 1
intelligence_filter = heroes_data["type"] == 2

strength_heroes = heroes_data[strength_filter]
agility_heroes = heroes_data[agility_filter]
intelligence_heroes = heroes_data[intelligence_filter]

# ----------------------------------------------
# A function that draws a horizontal line at the screen:
def line():
    print()
    print(100*"-")
    print()
# ----------------------------------------------

line()

print("STRENGTH HEROES")
print(strength_heroes.loc[:,"name":"type"].head())

line()

print("AGILITY HEROES")
print(agility_heroes.loc[:,"name":"type"].head())

line()

print("INTELLIGENCE HEROES")
print(intelligence_heroes.loc[:,"name":"type"].head())

line()


# With using 'head' method of pandas library, we printed our new data frames' first 5 heroes; strenght, agility and intelligence heroes.

# In[ ]:


f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(heroes_data.corr() , annot = True , linewidths = .5 , fmt = '.1f' , ax = ax)
plt.show()


# As we can see at this table, strength stats and range stats are not doing well together. Because in this game, generally big guys (durable strength heroes, 'tank heroes') handle things with melee combat. They are tough and not good at dexterity skills like range attack.
# 
# Because of their thin skin, mage heroes (mostly intelligence heroes) needs to stay away from hand-to-hand combat. So nearly all of them are range-attack heroes. We can see this at this table. Intelligence stats are directly proportional with range stats like 0.5 - 0.6 valuing.

# In[ ]:


f,ax = plt.subplots(figsize = (18,18))
strength_heroes["maxDmg"].plot( kind = 'line' , color = 'red' , label = 'Strength Heroes' , linewidth = 2 , 
                                     alpha = 1 , grid = True , linestyle = '-' , ax=ax )
agility_heroes["maxDmg"].plot( kind = 'line' , color = 'green' , label = 'Agility Heroes' , linewidth = 2 , 
                                     alpha = 1 , grid = True , linestyle = '-' , ax=ax )
intelligence_heroes["maxDmg"].plot( kind = 'line' , color = 'blue' , label = 'Intelligence Heroes' , linewidth = 2 , 
                                     alpha = 1 , grid = True , linestyle = '-' , ax=ax )

plt.legend( loc = 'upper right' )
plt.xlabel('Heroes')
plt.ylabel('Maximum Damage')
plt.title('MAXIMUM DAMAGE OF HEROES')
plt.show()


# We can see that generally strength heroes have more damage than the others. There is a really harsh guy there. At around 85th index, cannot be sure. Lets find him/her.

# In[ ]:


filter = strength_heroes["maxDmg"] > 85
the_harsh_guy = strength_heroes[filter]
print(the_harsh_guy)


# He is 'Treant Protector'. A living tree. Has a strong base damage. Probably a slow guy. Because I think strength and speed should be inversely proportional. Lets compare these stats.

# In[ ]:


f,ax = plt.subplots(figsize = (10,10))
heroes_data.plot( kind = 'scatter' , x = 'baseStr' , y = 'moveSpeed' , alpha = 0.5 , color = 'red' , ax=ax )
plt.xlabel('Base Strength')
plt.ylabel('Move Speed')
plt.title('STRENGTH - SPEED  |  Scatter Plot')
plt.show()


# They are maybe a little bit inversely proportional but not much exactly. Moreover when we look at the correlation map, we can see that they are inversely proportional but not a strong proportional, because the value is only '0.1'. 
# 
# Even so I am sure that strength characters have low range and intelligence characters have high range. Agility heroes should be balanced about this. Lets find out. 

# In[ ]:


strength_heroes["range"].plot( kind = 'hist' , bins = 50 , figsize = (8,8) )
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.title('RANGE HISTOGRAM FOR STRENGTH HEROES')
plt.show()

agility_heroes["range"].plot( kind = 'hist' , bins = 50 , figsize = (8,8) )
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.title('RANGE HISTOGRAM FOR AGILITY HEROES')
plt.show()

intelligence_heroes["range"].plot( kind = 'hist' , bins = 50 , figsize = (8,8) )
plt.xlabel('Range')
plt.ylabel('Frequency')
plt.title('RANGE HISTOGRAM FOR INTELLIGENCE HEROES')
plt.show()


# 150 range actually means it is a melee hero. So when type goes from 0 to 2, there are more ranged heroes avaible. (From strength to intelligence). We can see this at correlation map too. Lets bring it back again:

# In[ ]:


f,ax = plt.subplots(figsize = (18,18))
sns.heatmap(heroes_data.corr() , annot = True , linewidths = .5 , fmt = '.1f' , ax = ax)
plt.show()


# Type and range stats are directly proportional strongly. The value is '0.7'.
