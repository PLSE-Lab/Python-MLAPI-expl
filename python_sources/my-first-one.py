#!/usr/bin/env python
# coding: utf-8

# ****# My First Kernel. Don't Judge!****
# ## Into
# I'm new to Jupiter's Notebook and to Data Science. The best way to start is just jump in!
# You will learn nothing from this notebook. nothing at all. well maybe about carage of a nooby.
# enjoy!
# 
# ## What are we going to do?
# Well when I started this i didn't know what i wanted to do.  I didn't really thought it through. we'll just see as we go...

# ------------------------------------------------------------------------
# every new code needs a hello world! let's just get it over with

# In[ ]:


print("hello world!")


# lets just play with numpy arrays, maybe even pandas. But first let import them and go to input folder to load our csv

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns  # visualization tool

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))


# 

# well well well.... we have a lot of data about NBA players of the week, lets load and see some statistics

# In[ ]:


data = pd.read_csv("../input/NBA_player_of_the_week.csv")
data.info()


# lets compute pairwise correlation of columns, excluding NA/null values

# In[ ]:


data.corr()


# lets visualize that:

# In[ ]:


#correlation map
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# lets see the first rows of the data

# In[ ]:


data.head(10)


# Matplot is a python library that help us to plot data. The easiest and basic plots are line, scatter and histogram plots.
# 
# * Line plot is better when x axis is time.
# * Scatter is better when there is correlation between two variables
# * Histogram is better when we need to see distribution of numerical data.
# * Customization: Colors,labels,thickness of line, title, opacity, grid, figsize, ticks of axis and linestyle
# 
# 
# lets plot the  draft year and winning season. why not

# In[ ]:


data['Draft Year'].plot(kind = 'line', color = 'g',label = 'Draft Year',linewidth=1,alpha = 0.5,grid = True,linestyle = ':')
data['Season short'].plot(color = 'r',label = 'Season',linewidth=1, alpha = 0.5,grid = True,linestyle = '-.')
plt.legend(loc='upper right')     # legend = puts label into plot
plt.xlabel('samples')              # label = name of label
plt.ylabel('year')
plt.title('Line Plot')            # title = title of plot
plt.show()


# lets try to display some scatter graphs to see who is supirier. by value and age!

# In[ ]:


#first lets lonvert the string height to float
try:
    data['Height']=pd.to_numeric(data['Height'].str.replace('-','.'))
except AttributeError:
    pass


# In[ ]:



data.plot(kind='scatter', x='Height', y='Weight',alpha = 0.5,color = 'red')
plt.xlabel('Height(foot)')              # label = name of label
plt.ylabel('Weight(pounds)')
plt.title('Hieght and Weight') 


# well, we can see most basketball player's hieights is around 6.0-6.5 foot
# but lets see it more clearly

# In[ ]:


data.Height.plot(kind = 'hist',bins = 50,figsize = (12,12))
plt.show()


# Lets find the abnormal height (>7.5 or <5.5)

# In[ ]:


data[(data['Height']<5.5) | (data['Height']>7.5)]


# Lets see the age distribution and create a column with boolean if a player is old

# In[ ]:


mean = sum(data.Age)/len(data.Age)
data['isOld'] = [True if age > mean else False for age in data.Age]
data.loc[:10,['Age','isOld']]


# Lets see who is the best team!

# In[ ]:


print(data['Team'].value_counts(dropna =False))  # if there are nan values that also be counted


# And some more statistic data about the players

# In[ ]:


data.describe()


# lets visualize the above

# In[ ]:


# For example: compare attack of pokemons that are legendary  or not
# Black line at top is max
# Blue line at top is 75%
# Red line is median (50%)
# Blue line at bottom is 25%
# Black line at bottom is min
# There are no outliers
data.boxplot(column='Age',by = 'isOld')


# let's tidy out data

# In[ ]:


data_new = data.head()    # I only take 5 rows into new data
# lets melt
# id_vars = what we do not wish to melt
# value_vars = what we want to melt
melted = pd.melt(frame=data_new,id_vars = 'Player', value_vars= ['Team','Date'])
# Index is name
# I want to make that columns are variable
# Finally values in columns are value
melted.pivot(index = 'Player', columns = 'variable',values='value')


# what about subplots? lets see

# In[ ]:


data.loc[:,['Age','Height','Weight']].plot(subplots=True)


# In[ ]:


# histogram subplot with non cumulative and cumulative
fig, axes = plt.subplots(nrows=2,ncols=1)
data.plot(kind = "hist",y = "Age",bins = 50,range= (0,250),normed = True,ax = axes[0])
#computive means that how much in % of the data we histogramed so far
data.plot(kind = "hist",y = "Age",bins = 50,range= (0,250),normed = True,ax = axes[1],cumulative = True)
plt


# > I'm not from the USA so i measure weight like a normal person. in KG lets transform by 1 lbs = 0.45359237 kg

# In[ ]:


data['Weight'] = data.Weight.apply(lambda n : n*0.45359237)
data.head()


# In[ ]:




