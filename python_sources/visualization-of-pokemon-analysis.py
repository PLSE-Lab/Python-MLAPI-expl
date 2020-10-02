#!/usr/bin/env python
# coding: utf-8

# <img src="https://wallpapercave.com/wp/upmtCfm.jpg" style="width:800px;height:400px;">

# # Libraries

# In[ ]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# # Information About Data

# In[ ]:


poke = pd.read_csv("/kaggle/input/pokemon/Pokemon.csv")
poke.head()


# In[ ]:


# Show some statistics about dataset
poke.describe()


# In[ ]:


poke.info()
#It gives infortmation about data.


# In[ ]:


# I search NaN values
poke.isnull().any()


# In[ ]:


poke.columns  # Shows that we have which columns


# In[ ]:


# Correlation map
plt.rcParams['figure.figsize']=(15,8)
hm=sns.heatmap(poke[[ 'Name', 'Type 1', 'Type 2', 'Total', 'HP', 'Attack', 'Defense',
       'Sp. Atk', 'Sp. Def', 'Speed', 'Generation', 'Legendary']].corr(), annot = True, linewidths=1, cmap='Blues')
hm.set_title(label='Correlation Map', fontsize=20)
hm;


# # Graphics

# In[ ]:


sns.jointplot(x="Sp. Atk", y="Speed", data=poke);
#This visual explains relationship between Special Attack and Speed


# In[ ]:


poke.Attack.plot(kind='hist',bins = 15, figsize=(5,5))
plt.show()
#Here i want to show how many pokemons that have Attack point. 


# In[ ]:


sns.boxplot(x="HP", data=poke);
# Box plot important for me so it gives HP distribution and demonstrate some outlier


# In[ ]:


poke['Generation'].value_counts() # It gives how many are there generation numbers


# In[ ]:


threshold = sum(poke.Attack)/len(poke.Attack)
print(threshold)
poke["attack_level"] = ["powerful" if i > threshold else "weakness" for i in poke.Attack]
poke.loc[:10,["attack_level","Attack"]]
#Thresold tells calculation ratio of Attack power


# In[ ]:


fig, axs = plt.subplots(2, 2, figsize = (12,12)) #plt.subplots() first two arguements are the number
# of rows and then the number of columns. The [figsize =] adjusts the size of the final output of graphs.
# See point and link 2 

ax1 = plt.subplot2grid((8,8), (0,0), rowspan=3, colspan=3) 
ax2 = plt.subplot2grid((8,8), (4,0), rowspan=3, colspan=3)
ax3 = plt.subplot2grid((8,8), (0, 4), rowspan=3, colspan=3)
ax4 = plt.subplot2grid((8,8), (4, 4), rowspan=3, colspan=3)

# ^Each one of the above ax commands positions each graph spot within a grid. 
# For a better understanding see point and link 4


fig.tight_layout() # To understand how this works see point and link 3

ax1.set_title("Plot1: HP and Attack", fontsize =18)
ax2.set_title("Plot2: HP and Attack", fontsize =18)
ax3.set_title("Plot3: HP and Attack", fontsize =18)
ax4.set_title("Plot4: HP and Attack", fontsize =18)

# ^The above code purely sets the title of each graph and the fontsize



# Plot 1
sns.regplot(x='HP', y='Attack', 
              data=poke, ax=ax1) #x_bins = 12, fit_reg = True, ci = 95, 
              #color = 'red', marker ="^", ax=ax1) 
# Notice the x and y are set columns of the poke dataset. The [ax =]
# is added because we have subplots and Python needs to know where to put this graph.
# But this graph has no customization, just the bare bones. 


# Plot 2 
sns.regplot(x='HP', y='Attack', 
              data=poke, fit_reg = False, color = 'green', marker ="^", ax=ax2)
# We're going to add some parameters. We'll add a [color =], [fit_reg =],
# [marker =] command to our function. The [color =] command let's us control the color 
# of the graph and the points. The [fit_reg =] command allows use to turn on/off the linear
# regression and just see the points. The default is True, unless we change it to False.
# The final addition is the [marker =] command, this changes the marker used on the graph to 
# mark the points.


# Plot 3
sns.regplot(x='HP', y='Attack', 
              data=poke, fit_reg = True, x_bins = 6, color = 'orange', ax=ax3)
# We're going to add some parameters still. Now we're adding the [x_bins =] command, and changing
# [fit_reg =] to True. The [x_bins =] commands seperates our data into bins, the number given is the number
# of bins the data is sepperated into. The [x_bins =] command also gives a confidence interval
# to the bins. This confidence interval is the verticle line running through the point. 
# The default is confidence interval is 95%, but that can be changed if needed. So in this plot
# we have 6 points each with a confidence interval, and linear regression running through
# our data. 


# Plot 4
sns.regplot(x='HP', y='Attack', 
              data=poke, fit_reg = False, x_bins = 12, ci = 99, color = 'red', ax=ax4)
# I like what we did with the last graph, so I'm going to add to that. But I don't like
# the line running through the data, I want the graph to be red, I want more bins, and I 
# want the confidence on each point to be 99% percent instead of 95%. To do this I turned
# [fit_reg =] to False, [color =] to 'red', [x_bins =] to 12, and introduced a new command, 
# the [ci =] command. This command sets the confidence interval of both the bins AND the line.
# In this exampe we don't have a linear fit to the data, so the [ci =] will only effect the bins.



plt.show()


# Visualize distribution of Attack variable with Seaborn distplot() function
# Seaborn distplot() function flexibly plots a univariate distribution of observations.

# In[ ]:


f, ax = plt.subplots(figsize=(10,8))
x = poke['Attack']
ax = sns.distplot(x, bins=10)
plt.show()
# This is some kind of histogram grap about Attack


# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
sns.stripplot(x="Sp. Def", y="Defense", data=poke)
plt.show()
#That's some kind of point graph and shows ratio of Sp.Defense-Defense 


# In[ ]:


sns.jointplot(x='Defense', y='Speed', data=poke, color ='yellow', kind ='hex', 
              size = 8.0)
plt.show()
#Some kind of Hexagon Chart


# In[ ]:


poke['Type 1'].unique()


# In[ ]:


fig = plt.figure(figsize=(15,15))

fig.add_subplot(211)
poke['Type 1'].value_counts().plot(kind='pie', 
                                       autopct='%1.1f%%',
                                       pctdistance=1.0)

fig.add_subplot(212)
poke['Type 2'].value_counts().plot(kind='pie', 
                                       autopct='%1.1f%%',
                                       pctdistance=1.0)

plt.show()
#Some kind of Pie Chart


# **Some kind of Violin Plot**

# In[ ]:


f, ax = plt.subplots(figsize=(8, 6))
sns.violinplot(x=poke["Sp. Def"])
plt.show()
#Finally i want to show Sp Defense point in violin graph 


# # WHAT WE DID 

# Briefly, We defined our data. Data is examined by us.We did import the libraries we will use so that we checked which have columns.We reached some kind of statistical graphics and including information in that graph.We compared each feature and we analyzed them. 
# 
# 
# THANKS FOR YOUR REVIEW. HAVE A GOOD DAY
# 
