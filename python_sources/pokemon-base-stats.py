#!/usr/bin/env python
# coding: utf-8

# Testing with Pokemon Stats:
# 
# 
# 
# 1. What  is the mean of all Pokemon's base stat totals?
# 
# 
# 
# 2. What is the average speed for all Pokemon?
# 
# 
# 
# 3.  Which Pokemon has / have the highest base stats?

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pylab
from pandas import *
from pylab import *
from subprocess import check_output

PokemonTable = pd.read_csv('../input/Pokemon.csv')

print("Base stat total average: " + str(PokemonTable['Total'].mean()))
print("Speed average: " + str(PokemonTable['Speed'].mean()))
print("\nPokemon with the highest BST are as follows:")
PokemonTable[PokemonTable.Total == PokemonTable['Total'].max()]


# For competitive players in the iconic Pokemon series, stats and numbers play a vital part when it comes
# 
# 
# down to strategizing and forming teams.  While types, abilities, effort values, individual values and held
# 
# 
# items play their own roles, it is important to look at base stats as well.  There are tournaments
# 
# 
# where teams are limited based on the base stat totals of its members, so as to promote fair play.
# 
# 
# After looking into various pieces of data, it is now known that the average base stat total of one such
# 
# 
# Pokemon is 435.1025.  For individual stats, speed is often considered one of the most, if not the most 
# 
# 
# important; as going first can mean winning or losing in games.  The average speed of all Pokemon falls
# 
# 
# at an average of 68.2775.  While in competitive battles, the speed stat of most Pokemon fall above 90.
# 
# 
# With legendary Pokemon serving as the main draws and mascots of games, they unexpectedly
# 
# 
# carry the stongest of base stats.  Both mega evolutions of Mewtwo and mega Rayquaza share this throne,
# 
# 
# with their base stats being a whopping 680.     
# 
