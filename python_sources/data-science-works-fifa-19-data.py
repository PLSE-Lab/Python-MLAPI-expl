#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # IN THIS KERNEL YOU CAN SEE CORRELATION BETWEEN PLAYER'S ATTRIBUTES,BEST PLAYERS,MOST VALUABLE PLAYERS ETC.,ALSO YOU CAN SEE EXPLORATARY DATA ANALYSIS(EDA) OF PLAYER'S DATA.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


data = pd.read_csv("../input/fifa19/data.csv")


# In[ ]:


data


# In[ ]:


data.columns


# # Correlation between player's attributes.

# In[ ]:


data.corr()


# In[ ]:


# Heatmap of player's attributes correlation
fig,axes = plt.subplots(figsize = (30,30))
sns.heatmap(data.corr(),annot = True,linewidth = 0.5,axes=axes)
plt.show()


# In[ ]:


data_best = data.head(n=21)
data_best


# In[ ]:


data_best["Age"].mean()


# In[ ]:


data_best["Overall"].mean()


# # First of all,we can see tha the best 20 players in world play in Europe,especially they play in 4 countries(France,Germany,Spain and UK)
# 
# # Also we can see age average of player is 28 years old.
# 
# # The oldest player is Diego Godin and the youngest players are Paulo Dybala and Harry Kane
# 
# # The average of overall in game is 90 points

# In[ ]:


data


# # Correlation between overall and value with scatter plot...

# In[ ]:


# Scatter Plot
# x = value  y = overall
# kind = kind of plot,x = given value of x axis,y = given value of y axis,alpha = opacity,color = color

data.plot(kind = "scatter",x = "Value",y = "Overall",alpha = 0.5,color = "g",figsize = (20,20))
plt.xlabel("Value")
plt.ylabel("Overall")
plt.title("Value - Overall Scatter Plot")
plt.show()


# # Players between 18 and 20 years old

# In[ ]:


data_young = data[(data["Age"] >= 18) & (data["Age"] <= 20)]


# In[ ]:


data_young


# # Also we are filtering players attributes that their overall is above 75.I've chosen 75 for overall because many FIFA 19 gamers think that players above this value of overall is good,actually you can include me this idea.

# In[ ]:


data_young1 = data_young[(data_young["Overall"] >= 75)]


# In[ ]:


data_young1


# In[ ]:


player_french = data_young1[data_young1["Nationality"] == "France"]
player_french


# In[ ]:


player_german = data_young1[data_young1["Nationality"] == "Germany"]
player_german


# In[ ]:


player_spanish = data_young1[data_young1["Nationality"] == "Spain"]
player_spanish


# In[ ]:


player_english = data_young1[data_young1["Nationality"] == "England"]
player_english


# # As you can see in the dataframes, France has 13 young(18-20 years old) player as professional player in clubs.
# 
# # Also results indicates that youth setup is very important for French clubs.

# # Now you can see cleaning data of players and EDA work with player's attributes. 

# In[ ]:


# Before the cleaning data,first I want to delete unnecessary columns from DataFrame
# Let's see the columns of DataFrame
data.columns


# In[ ]:


data = data.drop(["Photo","Flag","Club Logo","Real Face","Joined","Loaned From",'LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB'],axis = 1)


# In[ ]:


data


# In[ ]:


data.info()


# In[ ]:


data_eda = data.describe()
data_eda


# # First,you can see only 20 best player's name,overall,nationality,team,age with tidying data.

# In[ ]:


data20 = data.head(20)
data20


# In[ ]:


# Melting data

data_melted1 = pd.melt(frame = data20,id_vars = "Name",value_vars = ["Club","Overall","Nationality","Age","Value"])
data_melted1


# # As you can see, after melting th data,we can see player's name and some info about them but we can't visualize the data good. Because of that,if we pivot data,we can show the data with better visual

# In[ ]:


# Pivoting data

data_pivot1 = data_melted1.pivot(index = "Name",columns = "variable",values = "value")
data_pivot1


# # Also we can melt and pivot the data with different columns.
# # After that, we can concatenate datas and we can see more info about top 20 players.

# In[ ]:


# This time,we melt some other columns : Preferred Foot,Position,Jersey Number,Height,Weight,Release Clause

data_melted2 = pd.melt(frame = data20,id_vars = "Name",value_vars = ["Preferred Foot","Position","Jersey Number","Height","Weight","Release Clause"])
data_melted2


# # Now we will pivot the melted data like data_melted1

# In[ ]:


data_pivot2 = data_melted2.pivot(index = "Name",columns = "variable",values = "value")
data_pivot2


# # After melting and pivoting data with different columns,now,we can concatenate dataframes
# 
# # You can say "Bro, You can do this with one melting and one pivoting" but I did this to concatenate dataframes. 

# In[ ]:


# Concatenating dataframes

data_new20 = pd.concat([data_pivot1,data_pivot2],axis = 1)
data_new20


# # As you can see,we have a better dataframe with important info of top 20 players in the world
# 
# # Also,we can say that defenders has the lowest number as a player in top 20 players 
# 

# 
# 
# 
# 
# 
# # You can see in the boxplot,the best players choose right foot more than left foot

# In[ ]:


data_new20.boxplot(column = "Overall", by = "Preferred Foot",figsize = (15,15))
plt.show()

