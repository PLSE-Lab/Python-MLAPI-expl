#!/usr/bin/env python
# coding: utf-8

# Thank you to @NoahGift for providing this great dat source. 
# For original Kernels:
# https://www.kaggle.com/noahgift/nba-player-power-influence-and-performance
# and https://www.kaggle.com/noahgift/nba-team-valuation-exploration

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv");attendance_valuation_elo_df.head()


# In[ ]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()


# In[ ]:


mean_salary_df = salary_df.groupby(["TEAM"], as_index= False).mean().merge(attendance_valuation_elo_df, how="inner", on="TEAM")
mean_salary_df.head()


# In[ ]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()


# In[ ]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()


# In[ ]:


br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


# In[ ]:


plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus_minus_df["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)
plus_minus_df["PLAYER"] = players
plus_minus_df.head()


# In[ ]:


nba_players_df = br_stats_df.copy()
nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")
nba_players_df.head()


# In[ ]:


pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")
nba_players_df.head()


# In[ ]:


salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION", "TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()


# In[ ]:


diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))


# In[ ]:


len(diff)


# In[ ]:


nba_players_with_salary_df = nba_players_df.merge(salary_df); 


# In[ ]:


sns.lmplot(x= "WINS_RPM", y= "SALARY_MILLIONS", data=nba_players_with_salary_df)


# In[ ]:


wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv")
wiki_df.head()


# In[ ]:


wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)


# In[ ]:


median_wiki_df = wiki_df.groupby("PLAYER").median()


# In[ ]:


median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]


# In[ ]:


median_wiki_df_small = median_wiki_df_small.reset_index()


# In[ ]:


nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)


# In[ ]:


twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()


# In[ ]:


nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)


# In[ ]:


nba_players_with_salary_wiki_twitter_df.head()


# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap = "PuBu")


# In[ ]:


fig = plt.subplots(figsize= (15, 7))
ax = sns.boxplot(y= "SALARY_MILLIONS" , x="AGE", data = nba_players_with_salary_wiki_twitter_df, orient="Vertical", width= 0.9)


# In[ ]:


## However, there are few observations for some ages, so it is better to see age vs number of observations.
plt.subplots(figsize= (10, 5))
sns.countplot(x= "AGE", data =nba_players_with_salary_wiki_twitter_df)


# In[ ]:


## Let's check salary regarding to twitter favorite count, twitter retweet count and point closely.
sns.lmplot(x= "SALARY_MILLIONS", y= "TWITTER_FAVORITE_COUNT", data= nba_players_with_salary_wiki_twitter_df, size= 10)


# In[ ]:


sns.lmplot(x= "SALARY_MILLIONS", y= "TWITTER_RETWEET_COUNT", data= nba_players_with_salary_wiki_twitter_df, size = 10)


# In[ ]:


sns.lmplot(x= "POINTS", y= "SALARY_MILLIONS", data= nba_players_with_salary_wiki_twitter_df, size= 10)


# In[ ]:


sns.heatmap(nba_players_with_salary_wiki_twitter_df[["SALARY_MILLIONS","TWITTER_FAVORITE_COUNT", "TWITTER_RETWEET_COUNT", "POINTS"]].corr(), annot= True, cmap= "YlOrRd")


# In[ ]:


plt.subplots(figsize= (10, 7))
sns.boxplot(y= "SALARY_MILLIONS" , x="POSITION", data = nba_players_with_salary_wiki_twitter_df, orient="Vertical")


# In[ ]:


plt.subplots(figsize= (5, 7))
sns.boxplot(y= "VALUE_MILLIONS" , x="CONF", data = attendance_valuation_elo_df, orient="Vertical")


# In[ ]:


ax = sns.lmplot(x= "VALUE_MILLIONS", y= "SALARY", data= mean_salary_df, hue= "CONF", size = 10)
ax.set(xlabel='Mean Salary of a Team', ylabel='Team Valuation', title="Mean Salary vs Team Valuation:  2016-2017 Season")


# In[ ]:


ax = sns.lmplot(x="VALUE_MILLIONS", y="AVG", data=attendance_valuation_elo_df, hue="CONF", size = 10)
ax.set(xlabel='Team Valuation', ylabel='Average Attendence Per Game', title="NBA Team AVG Attendance vs Team Valuation:  2016-2017 Season")

