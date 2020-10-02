#!/usr/bin/env python
# coding: utf-8

# ## Cihan Yatbaz
# 
# ### 20 / 10 / 2018
# 
# 
# 
# 1.  [Introduction:](#0)
# 2.  [Ages of the players in the draft season:](#1)
# 3. [The age density of the player ages in draft seasons :](#2)
# 4. [Were week's players chosen the most of from which teams? :](#3)
# 5. [Most selected players in 'Player of the Week' ( TOP 10):](#4)
# 6. [Which position chosen the more  'Player of the Week':](#5)

# <a id="0"></a> <br>
# ## Introduction
# The NBA contains the selected Players of the Week. <br>
# You can see the data analyses I have done with this Kaggle. If you have any suggestions or my shortcomings, please let me know. I'll be happy about that.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
get_ipython().run_line_magic('matplotlib', 'inline')

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls","../input"]).decode("utf8"))

# Any results you write to the current directory are saved as output.


# In[ ]:


# Read data
nba_data = pd.read_csv('../input/NBA_player_of_the_week.csv')


# In[ ]:


# The first 5 lines of our data
nba_data.head()


# In[ ]:


#General information of our data
nba_data.info()


# <a id="1"></a> <br>
# ## 1. Ages of the players in the draft season
# <br>

# In[ ]:


#Number of values in our 'Age' column
nba_data.Age.value_counts()


# In[ ]:


#Now we are changing 'Draft Year' with 'draft_year'.
#The reason is to be able to prevent problems that we may experience in the future.
nba_data.rename(columns={"Draft Year" : "draft_year"}, inplace=True)


# In[ ]:


#Values in our 'draft_year' column
nba_data['draft_year'].unique()


# In[ ]:


#Let's see the age of the players in the draft season with Bar Plot
nba_data.Age= nba_data.Age.astype(float)
draft_year_list= list(nba_data['draft_year'].unique())
age_values= []

for i in draft_year_list:
    x= nba_data[nba_data['draft_year']==i]
    age_rate = sum(x.Age)/len(x)
    age_values.append(age_rate)

data = pd.DataFrame({'draft_year_list': draft_year_list, 'age_values':age_values})
new_index = (data['age_values'].sort_values(ascending=False)).index.values
sorted_data = data.reindex(new_index)

#Visualization
plt.figure(figsize=(15,10))
sns.barplot(x=sorted_data['draft_year_list'], y=sorted_data['age_values'])
plt.xticks(rotation= 90)
plt.xlabel('Player"s Age', size=15)
plt.ylabel('Draft Year', size=15)
plt.title('Ages of the players in the draft season', size=15)
plt.show()


# <a id="2"></a> <br>
# ## 2. The age density of the player ages in draft seasons 
# <br>

# In[ ]:


#Let's see the age density of the player ages in draft seasons with the Joint Plot.
g = sns.jointplot(nba_data.Age, nba_data.draft_year, kind='kde', size=10)
plt.savefig('graph1.png')
plt.show()


# <a id="3"></a> <br>
# ## 3. Were week's players chosen the most of from which teams ?
# <br>
# <br>To do learn this, let's do a few things first.

# In[ ]:


#Values in our 'Team' column
nba_data['Team'].unique()


# In[ ]:


#'index' state of the teams
nba_data.Team.value_counts().index


# In[ ]:


#The number of times the teams are selected.
#To see this, we use the 'values' variable.
nba_data.Team.value_counts().values


# In[ ]:


# Now let's see the percentage of players selected from the teams with Pie Charts
nba_data.dropna(inplace=True)
labels = nba_data.Team.value_counts().index
sizes = nba_data.Team.value_counts().values

#Visualization
plt.figure(figsize=(15,15))
plt.pie(sizes, labels=labels, autopct='%1.1f%%' )
plt.title('Selection Ratio of Teams', color='Blue', fontsize=20)
plt.show()


# <a id="4"></a> <br>
# ## 4. Most selected players in 'Player of the Week' ( TOP 10)
# <br>

# In[ ]:


#The number of times the selected players are selected
#We will evaluate the players in TOP 10.
nba_data.Player.value_counts()


# In[ ]:


#Now let's see the most chosen players in "Player of the Week".

top10 = nba_data.Player.value_counts()
plt.figure(figsize=(14,10))
sns.barplot(x= top10[:10].index, y= top10[:10].values) #[:10]-> We use it to show the top10 players.
plt.xticks(rotation=45)
plt.title('Most selected players in Player of the Week (TOP 10)', color='blue', fontsize=15)
plt.show()


# <a id="5"></a> <br>
# ##  5. Which position chosen the more  'Player of the Week' 
# <BR>

# In[ ]:


# Number of players selected from positions
nba_data.Position.value_counts()


# In[ ]:


# Now  let's see which position chosen the more  'Player of the Week' with 'Count Plot'

# SF -> Small Forward            ,  FC -> Forward Center          ,  F -> Forward   
# PG -> Point Guard              ,  GF -> Guard  Forward          ,  C -> Center
# G  -> Guard                    ,  PF -> Power Forward           ,  SG -> Shotting Guard
# G-F -> Guard - Forward         ,  F-C -> Forward - Center

sns.countplot(nba_data.Position)
plt.title('Which position chosen the more Player of the Week ', color='blue', fontsize=15)
plt.show()


# > # CONCLUSION
# Thank you for your votes and comments
# <br>**If you have any suggest, May you write for me, I will be happy to hear it.**

# > # REFERENCES
# <br> 1. KAAN CAN ( DATAI ) -  Seaborn Tutorial for Beginners - https://www.kaggle.com/kanncaa1/seaborn-tutorial-for-beginners 
# 
