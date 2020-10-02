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


# # Dataset
# The goals of this analysis are to understand the correlations among the sum stat (also known as species strengths), primary type, and six repective stats of pokemons in the video game Pokemon Sword/Shield.
# 
# Attributes:
# * name01: the name of the pokemon in English
# * name02: the name of the pokemon in Japanese
# * HP: the Health Points stat of the pokemon
# * Attack: the Attack stat of the pokemon (corresponding to Defense)
# * Defense: the Defense stat of the pokemon (corresponding to Attack)
# * Sp_Atk: the Special Attack stat of the pokemon (corresponding to Special Defense)
# * Sp_Def: the Special Defense stat of the pokemon (corresponding to Special Attack)
# * Speed: the Speed stat of the pokemon (determining the order of action in combat)
# * sum: the sum of all six base stats of the pokemon (also known as species strengths)
# * type1: the primary type of the pokemon
# * type2: the secondary type of the pokemon

# **Load the csv file and save it as a DataFrame.**

# In[ ]:


pokemon = pd.read_csv('/kaggle/input/pokemon-data-swordshield/pokemon-swordshield.csv').set_index('name01')
#set pokemon names (name01) as index 


# # 1. Display and Data Attributes

# **First to display some items from the dataFrame to check whether we have the correct data. **

# In[ ]:


pokemon.head()
#display the top 5 rows in the dataFrame


# In[ ]:


pokemon.tail(10)
#display the last 10 rows in the dataFrame


# In[ ]:


pokemon.index
#display the index values in the dataFrame


# In[ ]:


pokemon.columns
#display the names of the columns in the dataFrame


# In[ ]:


pokemon.shape
#return the # of rows and columns in the dataFrame


# From the output of the method .shape, we can know that there is a total of 408 pokemons in the video game Pokemon Sword/Shield.

# # 2. Select

# **Before going straight to the analysis, we will first try to find some of the strongest pokemons in the database.**
# **To do that, I decided to use the stat of the most well-known pokemon Charizard as the threshold to filter out "weak" pokemon**

# In[ ]:


pokemon.loc['Charizard']
#to locate the data under the index


# From the output from the .loc method, we can learn that Charizard has a species strength stat of 534.

# In[ ]:


pokemon[ pokemon['sum'] >= 534 ]    .iloc[:,-3:] 
#conditional selection: select only the data that has a sum stat that is equal otor greater than 534
#select the last three columns


# From the matrix above, we learn that almost half of the top-tier pokemons are Dragon type.
# 
# And so we have our first assumption: **Dragon type = high-stat**

# # 3. Summarize

# To prove the assumption **Dragon type = high-stat**, we need to compare the average stat of Dragon type pokemons to the average stat of all the pokemons.
# 
# Also, since some of the pokemons have evoluion stages, and a pokemon will have a drastic increase in its stat once it reached the final evolution stage. To avoid underestimation of the statistics, we will use the threshold of sum stat is equal to or greater than 450 as the filter.

# In[ ]:


pokemon[ pokemon['sum'] >= 450 ]['Attack'].sum() / pokemon[ pokemon['sum'] >= 450 ].shape[0]
#take the sum of the Attack stat and divided by the # of items in the dataFrame to get the average


# In[ ]:


pokemon[ pokemon['sum'] >= 450 ].mean()
#a faster and easier way to calculate the average


# In[ ]:


pokemon[ (pokemon['sum']  >= 450) & (pokemon['type1'] == 'Dragon')     | (pokemon['sum']  >= 450) & (pokemon['type2'] == 'Dragon') ]    .mean()


# By comparing the average of that stats, we learn that the stats of Dragon type pokemon do happen to be higher than the stat of all pokemons, which means our assumption **Dragon type = high-stat** is proved to be true. 

# In[ ]:


pokemon[ pokemon['sum'] >= 450 ].median()
#return the median value of the columns in the the dataFrame


# In[ ]:


pokemon[ pokemon['sum'] >= 450 ].std().round(2)
#return the standard deviation of the columns in the the dataFrame
#and round the output values to two decimal places


# # 4. Sort

# In[ ]:


pokemon.sort_values('sum',ascending=False)
#return a sorted dataFrame based on the values of the selected column, the default order is ascending


# **Zamazenta** and **Zacian** both have the highest sum stat of 720 in the game, but are these two really the best pokemon?
# 
# For the following, we would like to examine the question by looking into the base stats.

# In[ ]:


pokemon['HP'].max()
#return the maximum value in the column


# In[ ]:


pokemon['HP'].idxmax()
#return the index value of that has the highest value in the column


# According to the data above, we learn that even the pokemons that have the highest sum stats do not necessarily represent they have the highest base stats among others.
# 
# Players should also consider the base stats instead of only focusing on the species strengths when building a team.
# 

# Also we should take a look of the pokemon which has the lowest species strength to better understand the correlation between species strength and the base stats.

# In[ ]:


pokemon['Attack'].idxmin()
#return the index value of that has the lowest value in the column


# In[ ]:


pokemon['Attack'].min()
#return the minimum value in the column


# From the data, we can conclude that the species strength can only be a reference for the overall strength but not the base stats of a pokemon.
# 

# # 5. Split-apply-combine and Frequency Counts
For now we will try to find the correlation between the primary type (type1) and the base stats.
# In[ ]:


ptype = pokemon[ pokemon['sum'] >= 450 ]    .groupby(['type1'])    .mean().round(0)

#groupby: group the items in a dataFrame by the values of the selected column

ptype


# In[ ]:


pokemon[ pokemon['sum'] >= 450 ]    .groupby(['type1'])    .size()
#size: to count the number of apperances of the value


# In[ ]:


#an alternative method
pcount = pokemon[ pokemon['sum'] >= 450 ]    .type1.value_counts(ascending=True)
pcount


# # 6.Plot

# In[ ]:


ptype.iloc[:,:-1].plot.bar(figsize=(10,10))
#.plot.bar(): to output a bar chart of the selected dataFrame
#(figsize=( , )): to adjust the size of the chart


# According to the bar chart, we are able to conclude some obvious facts that can prove the correlation between the primary type and the base stats:
# * Steel type pokemons have the highest Defense stat and the lowest Speed stat
# * Fighting type pokemons have the highest Attack stat and the lowest Speical Attack stat
# * Normal type pokemons have the highest HP stat
# * Grass type pokemons have a relatively average base stats comparing to other types

# In[ ]:


ptype.iloc[:,-1:].plot.bar()


# From the chart, we concluded that Dragon type pokemons do have the highest species strength among other types.

# In[ ]:




