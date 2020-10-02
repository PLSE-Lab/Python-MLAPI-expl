#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


df = pd.read_csv("/kaggle/input/marble-racing/marbles.csv")


# In[ ]:


df.head()


# In[ ]:


df.site.unique()


# In[ ]:


df.race.unique()


# ## Team-wise Average Performance among Qualifier and Race Rounds
# * I have kept the scoring similar in both qualifier and Race rounds as they were judged in the actual MarbulaOne Race.

# In[ ]:


df["Points"] = df.pole.map({
    "P1": 25, 
    "P2": 18,
    "P3": 15,
    "P4": 12,
    "P5": 10,
    "P6": 8,
    "P7": 6,
    "P8": 4,
    "P9": 2,
    "P10": 1,
    "P11": 0,
    "P12": 0,
    "P13": 0,
    "P14": 0,
    "P15": 0,
    "P16": 0
})
df.head()


# In[ ]:


Player_stats = df[["race", "marble_name", "Points", "avg_time_lap", "points"]]
Orangin = Player_stats[Player_stats.marble_name == "Anarchy"].reset_index()
Orangin


# In[ ]:


player_wise = Player_stats.groupby('marble_name').mean().reset_index()
# len(player_wise)
print("Qualifier Races ---\n" + "median Score Across all rounds: " + str(np.median(player_wise.Points.values)) + "\nmean Score Across all rounds: " + str(np.average(player_wise.Points.values)) + 
     "\nMarbulaOne Races ---\n"+ "median Score Across all rounds: " + str(np.median(player_wise.points.values)) + "\nmean Score Across all rounds: " + str(np.average(player_wise.points.values)))


# The difference in the mean/median of the Qualifier and Race Rounds is due to the **bonus points given in the Race Rounds for the fastest lap time**.

# ## Analysing the Qualifier Rounds
# * Each marble participated in 4 Qualifier Rounds (Out of total 8 Rounds)

# In[ ]:


x = player_wise.marble_name.values
y1 = player_wise.Points.values
y2 = player_wise.avg_time_lap.values
plt.figure(figsize=(25, 6))
plt.xlabel("Players")
plt.ylabel("Average performance among all Qualifier Rounds rounds")
plt.bar(x, y1)


# In[ ]:


plt.figure(figsize=(25, 7))
sns.set_style('whitegrid')
graph = sns.boxplot(data=Player_stats, x='marble_name', y='Points')
graph.axhline(np.median(player_wise.Points.values))
plt.xlabel("Players")
plt.ylabel("Average Performance among all Qualifier Rounds")


# * It is clear that **Mary from Team Primary** always came in the bottom 6 across all the four qualifier rounds it took part in, consistently performing poorly
# 
# ### Worst Performers (worst at top):
#    1. Mary *(Team Primary)*
#    2. Pulsar *(Team Galactic)*
#    3. Anarchy *(Balls of Chaos)*
#    4. Razzy *(Raspberry Racers)*
#    5. Wispy *(Midnight Wisps)*
# 
# ### Top Performers (Best at top, going by median Preformance):
#    1. Prim *(Team Primary)*
#    2. Orangin *(O'rangers)*
#    3. Smoggy *(Hazers)*
#    4. Momo *(Team Momo)*
#    5. Speedy *(Savage Speeders)*, Wospy *(Midnight Wisps)*, Yellup *(Mellow Yellow)*, Mimo*(Team Momo)* for Winning a Single Qualifier with decent median Performance

# ## Analysing the Race Rounds
# * Here too, each marble participated in 4 Race rounds.

# In[ ]:


plt.figure(figsize=(25, 7))
sns.set_style('whitegrid')
graph = sns.boxplot(data=Player_stats, x='marble_name', y='points')
graph.axhline(np.median(player_wise.points.values))
plt.xlabel("Players")
plt.ylabel("Average performance among all rounds")


# * It very clearly shows that **Mary from Team Primary** has majorly underperformed throughout the tournament.
# 
# ### Worst Performers (worst at top):
#    1. Mary *(Team Primary)* and Vespa *(Hornets)*
#    2. Sublime *(Limers)*
#    3. Snowflake *(Snowballs)*
#    4. Anarchy *(Balls of Chaos)* and Wispy *(Midnight Wisps)*
#    5. Razzy *(Raspberry Racers)*  and Hive *(Hornets)*
# 
# ### Top Performers (Best at top, going by median Preformance):
#    1. Snowy *(Snowballs)*
#    2. Speedy *(Savage Speeders)* and Smoggy *(Hazers)*
#    3. Orangin *(O'Rangers)* 
#    4. Prim *(Team Primary)* and Rapidly *(Savage Speeders)*
#    5. Clutter *(Balls of Chaos)*, Mimo *(Team Momo)* for Winning a Single Race with decent median Performance

# In[ ]:


player_wise_sum = Player_stats.groupby('marble_name').sum().reset_index()
qualifier_score = player_wise_sum.Points.values
race_score = player_wise_sum.points.values
players = player_wise_sum.marble_name.values
colors = ['lightskyblue', 'gold', 'lightcoral', 'gainsboro', 'royalblue', 'lightpink', 'darkseagreen', 'sienna',
          'khaki', 'gold', 'violet', 'yellowgreen']

plt.figure(figsize=(24, 10))

ax1 = plt.subplot(1, 2, 1)
ax1 = plt.pie(qualifier_score, autopct='%0.f%%', pctdistance=0.8, colors=colors, startangle=345, shadow=True, labels=players)
plt.title("Qualifier Points achieved per Player")

ax2 = plt.subplot(1, 2, 2)
ax2 = plt.pie(race_score, autopct='%0.f%%', pctdistance=0.8, colors=colors, startangle=345, shadow=True, labels=players)
plt.title("Race Points achieved per Player")

plt.subplots_adjust(left=0.1, right=0.90)


# ## A General Comment:
# * **Mary (Team Primary)** consistently under-performed through the MarbulaOne Season1
# * Anarchy (Balls of Chaos), Wispy (Midnight Wisps) and Razzy (Raspberry Racers) did not do much help to their team either, in both the qualifier or the race rounds
# * Prim (Team Primary), Orangin (O'Rangers), Smoggy (Hazers) and Speedy(Savage Speeders) did pretty well throughout the tournament.
# * In a general sense, top performers in Qualifiers did make a considerable impact on the Race Rounds too (as the top performers list in both the cases Suggest) and marbles doing poorly in Qualifiers did not have any significant improvement too. 

# ## Team-Wise Split of MarbulaOne Season1

# In[ ]:


team_wise = df[['team_name', 'Points', 'points']]
team_wise_score = team_wise.groupby('team_name').sum().reset_index()
team_wise_score.head()


# In[ ]:


plt.figure(figsize=(24, 8))
sns.set_style('whitegrid')

ax1 = plt.subplot(1, 2, 1)
ax1 = sns.barplot(data=team_wise_score, x='team_name', y='Points')
plt.xticks(rotation=75)
ax1.set_yticks(range(0, 111, 10))
plt.xlabel("Team Name")
plt.title('Qualifier Round Total Score per Team')

ax2 = plt.subplot(1, 2, 2)
ax2 = sns.barplot(data=team_wise_score, x='team_name', y='points')
plt.xticks(rotation=75)
ax2.set_yticks(range(0, 111, 10))
plt.xlabel("Team Name")
plt.title('Race Round Total Score per Team')

plt.subplots_adjust(left=0.02, right=0.98)


# ## The Savage Speeders went on to win the MarbulaOne Season1
# * They will be the favourites for the next time too!
# * And they have a higher chance to fish out more athlete marbles, followed by atletes of **Hazers, O'Rangers, Snowballs and Team Momo**

# In[ ]:




