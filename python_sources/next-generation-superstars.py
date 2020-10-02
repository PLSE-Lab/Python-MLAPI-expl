#!/usr/bin/env python
# coding: utf-8

# **W**ith the modernisation and globalisation of football and the football club economy, scouting has grown in stature and importance. Competition to search for young talents is extremely keen. Although it is difficult to quantify the prevalence of scouting in modern football, circumstantial evidence of its magnitude is readily available.
# 
# The importance of scouting offers football clubs with several distinct advantages:
# 
# * **Global reach**: Scouting allows clubs to cast the largest possible net to find players from all around the world.
# * **Cheap players**: Players from lower leagues can be available at cheaper transfer prices, and command smaller wages.
# 
# Source : [Wiki](https://en.wikipedia.org/wiki/Scout_(association_football)

# In this, We will identify the players who are below 25(including 25) and could be next generation superstars.
# The idea is to look at attributes like Marking, Passing, Shot power etc. of all the players. Then we will look at top 10 players who are **similar** to Messi. 

# In[ ]:


#Loading Data
import pandas as pd
import numpy as np
fifa_data = pd.read_csv("../input/FullData.csv")


# > Looking at Age distribution of players.

# In[ ]:



bin_values = np.arange(start=5, stop=55, step=5)
fifa_data['Age'].plot(kind='hist', bins=bin_values, figsize=[12, 6], alpha=.4, legend=True)  # alpha for transparency


# Maximum players are in the age group 20-30 which is intuitive.

# In[ ]:


age_25 = fifa_data['Age'] <=25
fifa_data_25 = fifa_data[age_25]
print("There are %d players who are equal to or under 25 " %fifa_data_25.shape[0])  # Number of players who are under 25


# Selecting all the attributes from the original data and creating separate dataset. We will be needing row number of the player who is acting as reference. For example Cristiano Ronaldo is Row 0, Lionel Messi is Row 1 and so on. 

# In[ ]:


x_cols = ['Name','Weak_foot', 'Skill_Moves','Ball_Control', 'Dribbling', 'Marking', 'Sliding_Tackle','Standing_Tackle', 'Aggression', 'Reactions', 'Attacking_Position',
       'Interceptions', 'Vision', 'Composure', 'Crossing', 'Short_Pass','Long_Pass', 'Acceleration', 'Speed', 'Stamina', 'Strength', 'Balance',
       'Agility', 'Jumping', 'Heading', 'Shot_Power', 'Finishing','Long_Shots', 'Curve', 'Freekick_Accuracy', 'Penalties', 'Volleys']

from sklearn.metrics.pairwise import euclidean_distances
X= fifa_data_25[x_cols]
print(fifa_data[['Name','Club','Nationality']][fifa_data['Name'].str.contains("Messi")])
X2 = pd.DataFrame(fifa_data.loc[1,x_cols] .values.reshape(1,32)) 
# Substitute fifa_data.loc[row#,x_cols] 


# >  Using Euclidean distance between Messi & other players we will see the top 10 players who are similar to him. 

# In[ ]:


top_10 = euclidean_distances(X.iloc[:,1:], X2.iloc[:,1:])

top_10_df = pd.DataFrame(top_10)
top_10_df = top_10_df.sort_values([0])

print("Top 10 players who have similar attributes as Messi are below")
X.iloc[top_10_df.index.values[1:11],0]


# One of the primary tasks of a scout is to identify players who are talented and can be attained for a cheaper price. Paulo Dybala is already termed as "Next Messi".Neymar & Coutinho are quite popular, I must say. This kind of explains why Barcelona didn't want to lose Neymar and the Coutinho-saga.Yannick Carrasco and Quincy Promes are not popular. 
# I have not heard of them before this analysis. Now, if the statistics are saying they are similar to Messi, may be the clubs can take a shot or scout further before making a bid for them. Who knows they can be "Next Messi".
# **Only  time can tell.**
