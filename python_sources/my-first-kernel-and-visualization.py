#!/usr/bin/env python
# coding: utf-8

# # About the dataset
# This data set includes 721 Pokemon, including their number, name, first and second type, and basic stats: 
# HP, Attack, Defense, Special Attack, Special Defense, and Speed. It has been of great use when teaching statistics to kids. With certain types you can also give a geeky introduction to machine learning.
# 
# This are the raw attributes that are used for calculating how much damage an attack will do in the games. This dataset is about the pokemon games (NOT pokemon cards or Pokemon Go).
# 
# The data as described by Myles O'Neill is:
# 
# # ID for each pokemon
# Name: Name of each pokemon
# Type 1: Each pokemon has a type, this determines weakness/resistance to attacks
# Type 2: Some pokemon are dual type and have 2
# Total: sum of all stats that come after this, a general guide to how strong a pokemon is
# HP: hit points, or health, defines how much damage a pokemon can withstand before fainting
# Attack: the base modifier for normal attacks (eg. Scratch, Punch)
# Defense: the base damage resistance against normal attacks
# SP Atk: special attack, the base modifier for special attacks (e.g. fire blast, bubble beam)
# SP Def: the base damage resistance against special attacks
# Speed: determines which pokemon attacks first each round
# The data for this table has been acquired from several different sites, including:
# 
# pokemon.com
# pokemondb
# bulbapeida
# One question has been answered with this database: The type of a pokemon cannot be inferred only by it's Attack and Deffence. It would be worthy to find which two variables can define the type of a pokemon, if any. Two variables can be plotted in a 2D space, and used as an example for machine learning. This could mean the creation of a visual example any geeky Machine Learning class would love.
# 
# # Columns;
# * Name
# * Type 1
# * Type 2
# * Total
# * HP
# * Attack
# * Defense
# * Sp. Atk
# * Sp. Def
# * Speed
# * Generation
# * Legendary
# 

# # Import to Packages

# In[ ]:


import numpy as np
import pandas as pd 
import seaborn as sbr
import matplotlib.pyplot as plt
import warnings
from mpl_toolkits.mplot3d import Axes3D
import os
print(os.listdir("../input"))


# # Data Import 

# In[ ]:


data = pd.read_csv('../input/Pokemon.csv')
data.head()


# In[ ]:


data.info()


# # I think, the columns are more useful in this way

# In[ ]:


data.columns = data.columns.str.upper().str.replace(' ', '')
data.head()


# In[ ]:


data.describe()


# # I used jointplot in seaborn.

# In[ ]:


sbr.jointplot(x="ATTACK",y="DEFENSE",data=data,kind="hex",color="green");
warnings.filterwarnings("ignore")


# # Correlation in this data.

# In[ ]:


data.corr()


# # Correlation Visualization using Heatmap

# In[ ]:


f,ax=plt.subplots(figsize=(18,18))
sbr.heatmap(data.corr(),annot=True,fmt=".1f",linewidths=.5,cmap="YlGnBu",ax=ax,linecolor="black")
plt.show()


# # Violinplot

# In[ ]:


f,ax=plt.subplots(figsize=(15,15))
sbr.violinplot(x=data.GENERATION,y=data.ATTACK,inner=None)
sbr.swarmplot(x=data.GENERATION,y=data.ATTACK,size=5,color="black")
warnings.filterwarnings("ignore")


# # Boxplot

# In[ ]:


f,ax=plt.subplots(figsize=(20,20))
sbr.boxplot(x=data.GENERATION,y=data.TOTAL,palette="Set3")
sbr.stripplot(x=data.GENERATION,y=data.TOTAL,hue=data.LEGENDARY,linewidth=1,size=12,jitter=True,edgecolor="black")
warnings.filterwarnings("ignore")


# # Barplot

# In[ ]:


sbr.set(rc={'figure.figsize':(10,10)})
sbr.barplot(x=data.TOTAL,y=data.TYPE1,data=data,palette="Set2",linewidth=1)
warnings.filterwarnings("ignore")


# # Residplot

# In[ ]:


sbr.set(style="whitegrid")
sbr.residplot(x=data.GENERATION,y=data.TOTAL,lowess=True,color="r",order=0.01,robust=True)
warnings.filterwarnings("ignore")


# # 3D Scatterplot

# In[ ]:


fig=plt.figure()
ax=fig.add_subplot(111,projection="3d")
ax.scatter(data["ATTACK"],data["TOTAL"],data["GENERATION"],c="orange",s=30)
ax.view_init(20,250)
plt.show()


# In[ ]:


sbr.boxenplot(x="GENERATION",y="ATTACK",hue="LEGENDARY",data=data)
warnings.filterwarnings("ignore")


# In[ ]:


plt.figure(figsize=(17,10))
sbr.regplot(x="ATTACK",y="TOTAL",data=data)
plt.xlabel("HP",size=20)
plt.ylabel("ATTACK",size=20)
plt.show()


# In[ ]:


data.head()


# In[ ]:


plt.figure(figsize=(12,12))
sbr.scatterplot(x="SPEED",y="TOTAL",data=data,hue="GENERATION")
plt.show()

