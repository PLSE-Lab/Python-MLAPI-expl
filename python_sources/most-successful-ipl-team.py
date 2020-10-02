#!/usr/bin/env python
# coding: utf-8

# Cricket is undoubdtedly the most popular sport in India. In this kernel, I intend to find the teams with the **highest winning percentage** throughout all the seasons from 2007 - 2018. 

# In[83]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt


# In[84]:


df = pd.read_csv("../input/matches.csv")
df = df[['team1', 'team2', 'winner']]
df.head(2)


# In[85]:


teams = df.team1.unique()
teams


# In[86]:


winPercent = []

for each_team in teams:
    played_matches = np.count_nonzero(df['team1'].astype(str).str.contains(each_team)) + np.count_nonzero(df['team2'].astype(str).str.contains(each_team))
    matches_won = np.count_nonzero(df['winner'].astype(str).str.contains(each_team))   
    winPercent.append(100 * (matches_won / played_matches))
    
winPercent, teams = zip(*sorted(zip(winPercent, teams))) #Sort teams as per winning percentage (Descending order)

plt.figure(figsize=(12,6))
plt.barh(range(len(winPercent)), winPercent, align='center')
plt.yticks(range(len(winPercent)), teams)
plt.title("IPL Teams: Winning Percentage")
plt.show()


# No surprises there. **Chennai Super Kings** has the best winning percentage. Both the top teams in this list (Chennai Super Kings and Mumbai Indians) are tied when it comes to winning the most league titles: 3 IPL titles.
# 
# P.S. - I am a Mumbai Indians fan!
