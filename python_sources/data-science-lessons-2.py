#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt 
import seaborn as sns #Visulization library
import os

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # ****Import and First Look Data****

# In this section, we are going to learn some useful basic functions and usages to procces the data. And we are going to use Pandas library to do those things. 

# In[ ]:


data = pd.read_csv("/kaggle/input/pokemon-challenge/pokemon.csv") #Claming data from the dataset


# In[ ]:


data.info() #Showing some informations about the claimed dataset


# In[ ]:


data.corr() #Displaying a table about the relation between our variables 


# This part is, creating a heatmap based on the corrolation between the variables. And it helps us to see and understand easily.

# In[ ]:


f, ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr(), annot = True, linewidths = .5, fmt = '.4f', ax = ax)
plt.show()


# In[ ]:


data.head(10) #Gives us the first ten pokemons' information.


# In[ ]:


data.columns #Gives us the feauters(name of each cloumn)


# # Matplotlib

# Matplotlib is a visualization library like Seaborn library. It displays graphics and tables to simplify things. There are few plotting techniques for different situations:
# * Line Plot: If we want to see multiple features for each x value
# * Scatter Plot: If there is correlation between two variables
# * Histogram: If we want to see distribution of numerical data

# **Line Plotting** 

# In[ ]:


#Parameters: kind = plot type, color = color, label = label, linewidth = width of each line, alpha = opacity, grid = grid, linestyle = style of line.
data.Speed.plot(kind = 'line', color = 'g', label = 'Speed', linewidth = 1, alpha = 0.5, grid = True, linestyle = ':') #Graphic for speed feature
data.Defense.plot(kind = 'line', color = 'r', label = 'Defense', linewidth = 1, alpha = 0.5, grid = True, linestyle = '-.') #Graphic for defense feature
plt.legend(loc = 'upper right') #Defining the place of line guide
plt.xlabel('x axis') #Labeling x axis
plt.ylabel('y axis') #Labeling y axis
plt.title('Line Plot') #Naming title of graphic
plt.show()


# As we can see here, x axis is equal to pokemons' index numbers. Y axis is the normal numerical values. 
# The green lines show the value of speed for each pokemons. And the red lines show the defense values of pokemons'.

# **Scatter**

# In[ ]:


#Parameters: kind = plot type, x = the column name in dataset, y = same like x, alpha = opacity, color = color.
data.plot(kind = 'scatter', x = "Attack", y = "Defense", alpha = 0.5, color = 'r')
plt.xlabel('Attack') #Labeling x axis
plt.ylabel('Defense') #Labeling y axis
plt.title('Scatter Plot') #Naming title of graphic
plt.show()


# Here, in scatter plot, we can visualize the correlation between two features, like "Attack" and "Defense". We can clearly understand that there is a direct proportion between these two features. If pokemons' attack value increases, defense value will be increased at the same time.
# 
# PS. For sure there are some exeptions but all we care about is the avarage result.

# **Histogram**

# In[ ]:


#Parameters: kind = type of plot, bins = number of bars, figsize = sizes of graphic.
data.Speed.plot(kind = 'hist', bins = 50, figsize = (12,12))
plt.xlabel('Speed') #Labeling x axis
plt.ylabel('Frequency') #Labeling y axis
plt.title('Histogram') #Naming title of graphic
plt.show()


# Histogram shows us how many pokemons have the same speed value are there. Frequency is equal to the number of pokemons, and Speed is equal to speed values. For example, there are approximately 40 pokemons which have 100 speed value.

# In[ ]:


plt.clf() #This function clears all of the plots and tables.

