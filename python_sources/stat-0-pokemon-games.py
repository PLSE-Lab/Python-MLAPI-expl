#!/usr/bin/env python
# coding: utf-8

# **Overview**
# 
# 
# 
# This data set includes 721 Pokemon, including their number, name, first and second type, and basic stats: HP, Attack, Defense, Special Attack, Special Defense, and Speed. It has been of great use when teaching statistics to kids. With certain types you can also give a geeky introduction to machine learning.
# 
# This are the raw attributes that are used for calculating how much damage an attack will do in the games. This dataset is about the pokemon games (NOT pokemon cards or Pokemon Go).
# 
# The data as described by Myles O'Neill is:
# 
#  ID for each pokemon
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
# 
# One question has been answered with this database: The type of a pokemon cannot be inferred only by it's Attack and Deffence. It would be worthy to find which two variables can define the type of a pokemon, if any. 
# 
# Two variables can be plotted in a 2D space, and used as an example for machine learning. This could mean the creation of a visual example any geeky Machine Learning class would love.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.stats import skew
from scipy.stats import kurtosis

from sklearn import preprocessing 
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


game = pd.read_csv("../input/Pokemon.csv")


# In[ ]:


game.head()


# In[ ]:


game.describe()


# Before proceeding to the analysis part we will examine the features by using EDA.
# The data set has 800 observation of different pokemons.
# First is the ID Column which is just an Index,we will Ignore for our analysis now.
# We examine every feature and compare it with other features as well.
# 

# **Univariate Analysis of Data
# 
# I will use this kernel as a medium to emphasis the importance of statistics in analytics.
# First we will cover what is called as Uni variate data analysis.
# Uni - single,we will examine one feature at a time.By the term examine,I mean to say that we will identify few key parameters that we are already aware of some might be new.
# 
# 
# You can check this link to know more,
# 
# https://www.statisticshowto.datasciencecentral.com/univariate/
# 
# Going in order,first we will observe Type I column.
# Clearly from the data description,which says Type I captures to which object the pokemon is resistent.
# 
# We will examine what are the unique categories are and how much each category contribute to.
# 
# 

# In[ ]:


val = pd.value_counts(game['Type 1'].values, sort=True)
val
val.plot(kind = 'bar')
#We could see that resistance to water tops,followed by Normal,Bug,Grass and so on.


# We are aware of the mean and variance concepts.The skewness of the data tells us whether the data is distrbuted normally or skewed or forming lumps.
# Acceptable skewness should be between -2 & +2.
# Kurtosis is the peakedness of the distribution.Acceptable kurtosis should be in the range -3&+3.

# In[ ]:


val2 = pd.value_counts(game['Type 2'].values,sort = True)
val2
val2.plot(kind = "bar")
#We have flying toping the chart here followed by close competition between Grounf,Poison & Psychic.


# In[ ]:


#Examining distribution of total
#The idea here is to know whether there are influential observations in the variable like ones which are abnormal and how do they affect the model/analysis.
#https://www.khanacademy.org/math/cc-eighth-grade-math/cc-8th-data/cc-8th-interpreting-scatter-plots/a/outliers-in-scatter-plots
    

plt.scatter(game.Total,game.Name)
plt.subplots_adjust(bottom=0.5, right=1.5, top=40)
##Looks like we have few influential observations,


# In[ ]:


fig=plt.figure()
ax=fig.add_subplot(111)
# plot points inside distribution's width
plt.scatter(game.Total,game.Name)
plt.scatter(game.Name, game.Total<700, marker="s", color="#2e91be")
# plot points outside distribution's width

plt.subplots_adjust(bottom=0.5, right=1.8, top=10)
plt.show()


# In[ ]:


#examining health points or HP
game.drop(['#'], axis=1)


# In[ ]:


game['Id']=game.index
game.head()

HP = pd.value_counts(game['HP'].values,sort = "False")
HP.plot(kind = 'bar')
plt.subplots_adjust(bottom=0.5, right=2, top=5)
##The value of HP is skewed towards right


# In[ ]:


print("mean:",np.mean(HP))
print("var:",np.var(HP))
print("skewness:",skew(HP))
print("kurtosis:",kurtosis(HP))


# Generally those variables having high Kurtosis are heavy tailed and have outliers in it.Therefore they are suppoed to be taken care.

# In[ ]:


#Examining Attack column

Attk = pd.value_counts(game['Attack'].values,sort = True)
Attk.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)
#This one again is skewed


# In[ ]:


print("mean:",np.mean(Attk))
print("var:",np.var(Attk))
print("skewness:",skew(Attk))
print("kurtosis:",kurtosis(Attk))


# Attack column is also skewed.

# In[ ]:


Defnc = pd.value_counts(game['Defense'].values,sort = True)
Defnc.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)


# In[ ]:


print("mean:",np.mean(Defnc))
print("var:",np.var(Defnc))
print("skewness:",skew(Defnc))
print("kurtosis:",kurtosis(Defnc))


# Skewness and Kurtosis observed.

# In[ ]:


Splatk = pd.value_counts(game['Sp. Atk'].values,sort = True)
Splatk.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)


# In[ ]:


print("mean:",np.mean(Splatk))
print("var:",np.var(Splatk))
print("skewness:",skew(Splatk))
print("kurtosis:",kurtosis(Splatk))


# In[ ]:


Spldef = pd.value_counts(game['Sp. Def'].values,sort = True)
Spldef.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)


# In[ ]:


print("mean:",np.mean(Spldef))
print("var:",np.var(Spldef))
print("skewness:",skew(Spldef))
print("kurtosis:",kurtosis(Spldef))


# In[ ]:


spd = pd.value_counts(game['Speed'].values,sort = True)
spd.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 5)


# In[ ]:


print("mean:",np.mean(spd))
print("var:",np.var(spd))
print("skewness:",skew(spd))
print("kurtosis:",kurtosis(spd))


# In[ ]:


gen = pd.value_counts(game['Generation'].values,sort = True)
gen.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 2)


# Since generation is categorical,we are not going to examine summary statistics for it,it would be insignificant.
# Category 1 has highest counts and category 6 being the least.
# We will examine each of the category.
# 

# In[ ]:


legend = pd.value_counts(game['Legendary'].values,sort = True)
legend.plot(kind = 'bar')
plt.subplots_adjust(bottom = 0.5,right = 2,top = 2)


# Number of false is more than number of True values.

# We will also look into some statistical concepts ,such as framing hypothesis soon.
# We will have to visualise the combined effects of each of the variable provided here.Therefore we will consider factors one at a time and represent it graphically.This will help us understand better.
# 

# *Standardising data
# *By standardising the data we will have a mean of 0,Standard Deviation of 1.The aforementioned conditions are specific to a normal distribution.
# Standardising is nothing but shifting the data population to form a lump in the middle and a spread on both the sides.
# For any analysis to be reliable we require the distribution of the variables to be normally distributed.
# We have seen from our EDA that most of the numerical variables are skewed towards right.This means there is a requirement for us to standardise the data before applying any algorithms.

# 
