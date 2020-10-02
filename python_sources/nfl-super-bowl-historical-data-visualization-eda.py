#!/usr/bin/env python
# coding: utf-8

# # NFL Super Bowl Historical Data Visualization & Explorations

# ![](https://storage.googleapis.com/kaggle-competitions/kaggle/12125/logos/header.png?t=2018-11-30-18-08-32")

# This kernel is made for the historical winings and losing of the NFL teams based on points by winners and losers and hosted by the different stadiums in different cities and states of America

# ## About NFL
# The National Football League is America's most popular sports league, comprised of 32 franchises that compete each year to win the Super Bowl, the world's biggest annual sporting event. Founded in 1920, the NFL developed the model for the successful modern sports league, including national and international distribution, extensive revenue sharing, competitive excellence, and strong franchises across the country.
# 
# The NFL is committed to advancing progress in the diagnosis, prevention and treatment of sports-related injuries. The NFL's ongoing health and safety efforts include support for independent medical research and engineering advancements and a commitment to work to better protect players and make the game safer, including enhancements to medical protocols and improvements to how our game is taught and played.

# ### Basic Libraries Import

# In[ ]:


# for basic operations
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# for visualizations
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# for defining path
import os


# ### Data Loading

# In[ ]:


# read the dataset
nfl_data = pd.read_csv("/kaggle/input/superbowl-history-1967-2020/superbowl.csv",index_col="Date")

# let's check the shape of the dataset

nfl_data.shape # dimensions of the data


# In[ ]:


# first look of the data
nfl_data.head(10)


# ### Data Information:
# * **Date** - Month and Year of the Game
# 
# * **SB** - SB is the Super Bowl Number
# 
# * **Winner** - Winning Team of the Year
# * **Winner Pts** - Total Points of the winning team
# * **Loser** - Loser Team of the Year
# * **Loser Pts** - Total Points of the losing team
# * **MVP** - MVPis most valuable player of the match
# * **Stadium** - Stadium who host the game
# * **City** - City of the stadium
# * **State** - state of the city

# In[ ]:


# Statistical description of NFL points
round(nfl_data.describe())


# # Data Visulizations & Explorations

# In[ ]:


sns.set(font_scale=1.4)


# ### Team VS Team Faced Eachother in History
# How many times they faced each other?

# In[ ]:


nfl_data['Win_vs_lost'] = nfl_data['Winner'] +"  VS  " + nfl_data['Loser'] 
plt.figure(figsize=(20,20))
plt.xticks( rotation=90)

sns.countplot(y="Win_vs_lost",data=nfl_data,orient='h',order=nfl_data['Win_vs_lost'].value_counts().sort_values(ascending=False).index)
sns.set_style("whitegrid")


plt.title("Faced Eaachother") 
plt.xlabel("# of Time Teams Faced Eachother") 
plt.ylabel("NFL Teams") 
plt.show()


# In above plot you can see that just two times happened when common teams faced each other such as.
# 
# New York Giants VS New England Patriots
# 
# Dallas Cowboys VS Buffalo Bills
# 
# Pittsburgh Steelers VS Dallas Cowboys
# 
# San Francisco 49ers VS Cincinnati Bengals

# ### Team VS Team Faced Eachother in History by Years
# How many times and What Happened when they faced each other?

# In[ ]:



plt.figure(figsize=(20,20))
plt.xticks( rotation=75)


sns.scatterplot(x=nfl_data.index,y=nfl_data['Win_vs_lost'],hue=nfl_data['Winner'],s=150)
sns.set_style("whitegrid")


plt.title("Number of Time Faced and Winers By Year") 
plt.ylabel("NFL Teams") 
plt.xlabel("Game Years") 


# Both time match won by New York Giants
# 
# Both time match won by Dallas Cowboys
# 
# But against Pittsburgh Steelers both time match won by Pittsburgh Steelers
# 
# Both time match won by San Francisco 49ers

# ### Winers Vs Losers Colored by Winers
# What happened when each team faced other teams

# In[ ]:



plt.figure(figsize=(20,20))
plt.xticks( rotation=90)

sns.scatterplot(x=nfl_data['Winner'],y=nfl_data['Loser'],hue=nfl_data['Winner'],s=200)
sns.set_style("whitegrid")


plt.title("Winners vs Losers Teams") 
plt.ylabel("Losers") 
plt.xlabel("Winners") 


# ### Number of Winnings by Each Team of NFL Sper Bowl

# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks( rotation=75)

sns.countplot(x="Winner",data=nfl_data,order=nfl_data['Winner'].value_counts().sort_values(ascending=False).index)
sns.set_style("whitegrid")


plt.title("Wins by each Team") 
plt.ylabel("# of Wins") 
plt.xlabel("NFL Teams") 
plt.show()


# ### Winnings of Each Team of NFL Sper Bowl by Year

# In[ ]:



plt.figure(figsize=(20,20))
plt.xticks( rotation=75)


sns.scatterplot(x=nfl_data.index,y=nfl_data['Winner'],hue=nfl_data['Winner'],s=150)
sns.set_style("whitegrid")


plt.title("Wins of each Team By Year") 
plt.ylabel("NFL Teams") 
plt.xlabel("Game Years") 


# ### Number of Losts by Each Team of NFL Sper Bowl

# In[ ]:



plt.figure(figsize=(20,10))
plt.xticks( rotation=75)

sns.countplot(x="Loser",data=nfl_data,order=nfl_data['Loser'].value_counts().sort_values(ascending=False).index)
sns.set_style("white")


plt.title("Loses by each Team") 
plt.ylabel("# of Loses") 
plt.xlabel("NFL Teams") 


# ### Loses of Each Team of NFL Sper Bowl by Year

# In[ ]:



plt.figure(figsize=(20,20))
plt.xticks( rotation=75)

sns.scatterplot(x=nfl_data.index,y=nfl_data['Loser'],hue=nfl_data['Loser'],s=150)
sns.set_style("whitegrid")


plt.title("Loss of each Team By Year") 
plt.ylabel("NFL Teams") 
plt.xlabel("Game Years") 


# ### Counts of Each Super Bowls in NFL

# In[ ]:



plt.figure(figsize=(20,10))
plt.xticks( rotation=75)

sns.countplot(x="SB",data=nfl_data,order=nfl_data['SB'].value_counts().sort_values(ascending=False).index)
sns.set_style("white")


plt.title("Super Bowls Played Counts") 
plt.ylabel("# of SB's") 
plt.xlabel("Super Bowls") 


# ### Winnings of Each Team of NFL Sper Bowl by Mean Points

# In[ ]:



plt.figure(figsize=(20,10))
plt.xticks( rotation=75)

sns.barplot(x=nfl_data['Winner'],y=nfl_data['Winner Pts'])
sns.set_style("whitegrid")


plt.title("Wins NFL Teams by Points") 
plt.ylabel("Points") 
plt.xlabel("NFL Teams") 


# ### Losts of Each Team of NFL Sper Bowl by Mean Points

# In[ ]:



plt.figure(figsize=(20,10))
plt.xticks( rotation=75)

sns.barplot(x=nfl_data['Loser'],y=nfl_data['Loser Pts'])
sns.set_style("whitegrid")


plt.title("Loses NFL Teams by Points") 
plt.ylabel("Points") 
plt.xlabel("NFL Teams") 


# ### Density of Winning and Losing Teams by Points

# In[ ]:



plt.figure(figsize=(20,10))
plt.xticks( rotation=75)
sns.set_style("white")

sns.distplot(a=nfl_data['Winner Pts'], label="Winners")
sns.distplot(a=nfl_data['Loser Pts'], label="Loser")

plt.title("Histogram of Petal Lengths, by Species")

# Force legend to appear
plt.legend()


# ### Most Valuable Player of the Match by Counts of Nominations

# In[ ]:



plt.figure(figsize=(20,10))
plt.xticks( rotation=75)

sns.countplot(x="MVP",data=nfl_data,order=nfl_data['MVP'].value_counts().sort_values(ascending=False).index,palette="rocket")
sns.set_style("whitegrid")


plt.title("MVP of Game Nominated") 
plt.ylabel("# of Nominations") 
plt.xlabel("NFL Players") 


# ### Most Valuable Player of the Match by Counts of Nominations and Years

# In[ ]:



plt.figure(figsize=(20,20))
plt.xticks( rotation=75)

sns.scatterplot(x=nfl_data.index,y=nfl_data['MVP'],s=150)
sns.set_style("whitegrid")


plt.title("MVP of Game Nominated by Year") 
plt.ylabel("Players") 
plt.xlabel("Game Years") 


# ###  Matches hosted by the Stadiums Counts

# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks( rotation=85)

sns.countplot(x="Stadium",data=nfl_data,order=nfl_data['Stadium'].value_counts().sort_values(ascending=False).index,palette="rocket")

sns.set_style("whitegrid")


plt.title("Game played in most stadiums") 
plt.ylabel("# of Time Playes in Stadiums") 
plt.xlabel("Stadiums") 
plt.show()


# ###  Matches hosted by the Stadiums by Years

# In[ ]:



plt.figure(figsize=(20,20))
plt.xticks( rotation=75)

sns.scatterplot(x=nfl_data.index,y=nfl_data['Stadium'],hue=nfl_data['City'],s=200)
sns.set_style("whitegrid")



plt.title("Game Hosted by each Stadium and City") 
plt.ylabel("Stadiums") 
plt.xlabel("Year") 


# In[ ]:


plt.figure(figsize=(20,10))
plt.xticks( rotation=90)

sns.countplot(x="City",data=nfl_data,order=nfl_data['City'].value_counts().sort_values(ascending=False).index,palette="rocket")

sns.set_style("whitegrid")


plt.title("Game played in most cities") 
plt.ylabel("# of Time Playes in City") 
plt.xlabel("US Cities") 
plt.show()


# In[ ]:



plt.figure(figsize=(20,20))
plt.xticks( rotation=75)

sns.scatterplot(x=nfl_data.index,y=nfl_data['City'],hue=nfl_data['State'],s=150)
sns.set_style("whitegrid")



plt.title("Game in the city in states by year") 
plt.ylabel("Cities") 
plt.xlabel("Year") 


# # Conclusion:

# ### Common teams faced each other only two times.
# 
# * New York Giants VS New England Patriots
# * Dallas Cowboys VS Buffalo Bills
# * Pittsburgh Steelers VS Dallas Cowboys
# * San Francisco 49ers VS Cincinnati Bengals
# 
# ### Macth won by above teams.
# 
# * Both time match won by New York Giants
# * Both time match won by Dallas Cowboys
# * But against Pittsburgh Steelers both time match won by Pittsburgh Steelers
# * Both time match won by San Francisco 49ers
# 
# ### Most Wins:
# * New England Patriots Won 6 times
# * Pittsburgh Steelers Won 6 times
# 
# ### Most Loses:
# * New England Patriots Won 5 times
# * Denver Broncos Won 5 times
# 
# ### Most Valuable Player:
# * Tom Brady nominated 4 times.
# * Joe Montana nominated 3 times.
# ### Intresting Insight
# Kansas City Chiefs won the game 2 times by with the diffeerence of **50 years**
# 
