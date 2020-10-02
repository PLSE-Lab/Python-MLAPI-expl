#!/usr/bin/env python
# coding: utf-8

# **This kernel provides an overview of the world cup history.(1930-2014)

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv("../input/WorldCups.csv")


# In[ ]:


data.info()


# In[ ]:


data.columns


# In[ ]:


data.corr()


# In[ ]:


f,ax = plt.subplots(figsize=(13, 13))
sns.heatmap(data.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# In[ ]:


data.head(10)


# In[ ]:


#World cup countries
data.Country.unique()


# In[ ]:


#How many times the countries has the World Cup been organized?
country_counts = Counter(data.Country)
cdf = pd.DataFrame.from_dict(country_counts,orient="index")
cdf.plot(kind="bar",figsize=(11,11),color="black")
plt.title("World Cup Homeowners")
plt.ylabel("Times")
plt.show()


# In[ ]:


#Number of matches per world cup.
data.plot(kind="line",x="Year",y="MatchesPlayed",color="blue",linewidth=1.5,grid=True,label="Matches Played",figsize=(10,10))
plt.xlabel("WC Years")
plt.ylabel("Numbers")
plt.title("Number of Matches per World Cup")
plt.legend()
plt.show()


# In[ ]:


#The correlation between teams number and matches number.
data.plot(kind="scatter",x="QualifiedTeams",y="MatchesPlayed",color="red")
plt.xlabel("Qualified Teams")
plt.ylabel("Matches Played")
plt.title("Correlation Between Teams Number & Matches Number")
plt.show()


# In[ ]:


#Correlation between goals,team,matches
plt.scatter(data.Year,data.GoalsScored,color="red",alpha=0.5,label="GoalsScored")
plt.scatter(data.Year,data.MatchesPlayed,color="green",alpha=0.5,label="MatchesPlayed")
plt.scatter(data.Year,data.QualifiedTeams,color="blue",alpha=0.5,label="QualifiedTeams")
plt.xlabel("WC Years")
plt.ylabel("Numbers")
plt.title("Correlation Between Goals-Team-Matches")
plt.legend(loc="upper left")
plt.show()


# In[ ]:


#Average of goals scored
goals_average = data.GoalsScored.mean()
goals_average


# In[ ]:


#Goal ratio
data["GoalRatio"] = ["High" if i > goals_average else "Low" for i in data.GoalsScored]
data.loc[:,["Year","GoalsScored","GoalRatio"]]


# In[ ]:


#Attendance per world cup
plt.bar(data.Year,data.Attendance,color="purple")
plt.xlabel("WC Years")
plt.ylabel("Attendance")
plt.title("Attendance per World Cup")
plt.show()


# **There was no World Cup between 1938 and 1950.Attendance rate increased in every world cup.

# In[ ]:


#Teams won
data.Winner.unique()


# In[ ]:


#the teams which won the trophy in its country
home_winner = data[data.Country == data.Winner] 
home_winner


# In[ ]:


away_winner = data[data.Country != data.Winner] 
away_winner


# In[ ]:


#How many times the countries won the trophy
win_counts = Counter(data.Winner)
windf = pd.DataFrame.from_dict(win_counts,orient="index")
windf.plot(kind="bar",figsize=(10,10),color="cyan")
plt.xlabel("Countries")
plt.ylabel("Times")
plt.title("Winning Number of Countries")
plt.show()


# In[ ]:


#second number of teams
second_counts = Counter(data.iloc[:,3])
secondf = pd.DataFrame.from_dict(second_counts,orient="index")
secondf.plot(kind="bar",figsize=(10,10),color="green")
plt.xlabel("Countries")
plt.ylabel("Times")
plt.title("Second Number of Countries")
plt.show()


# In[ ]:


#third number of teams
third_counts = Counter(data.Third)
thirdf = pd.DataFrame.from_dict(third_counts,orient="index")
thirdf.plot(kind="bar",figsize=(10,10),color="red")
plt.xlabel("Countries")
plt.ylabel("Times")
plt.title("Third Number of Countries")
plt.show()


# Brazil is the most successful team in the world cup.
