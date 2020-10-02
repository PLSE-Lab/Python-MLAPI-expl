#!/usr/bin/env python
# coding: utf-8

# # Exploration of NBA Salary, Performance, and Social Media Presence <br>
# This is my addition to the analyses conducted by Noah Gift.  Thanks for aggregating all of the data for us!  All the heavy lifting was already done. 

# ### **Contents**
# 1. [Import Data & Python Packages](#1-bullet) <br>
# 2. [Aggregate Player Stats Data](#2-bullet)<br>
# 3. [EDA](#3-bullet)<br>
#     * [3.1 EDA- Correlation Heatmap](#3.1-bullet) <br>
#     * [3.2 Salary By Position](#3.2-bullet) <br>    
#     * [3.3 Salary By Team](#3.3-bullet) <br>
#     * [3.4 Salary vs Age](#3.4-bullet) <br>
#     * [3.5 Salary vs Wins_RPM](#3.5-bullet) <br>
#     * [3.6 Salary vs Minutes Played](#3.6-bullet) <br>
#     * [3.7 Salary vs PPG](#3.7-bullet) <br>
#     * [3.8 Offensive Rebounds and Steals](#3.8-bullet) <br>
# 4. [Incorporate Wikipedia Page Views & Twitter Following](#4-bullet)
#    * [4.1 Numbers of Letters in Name](#4.1-bullet) <br>
#    * [4.2 EDA- Correlation Heatmap](#4.2-bullet) <br>

# ## 1. Import Data & Python Packages <a class="anchor" id="1-bullet"></a>

# In[ ]:


import numpy as np 
import pandas as pd 

import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt

from sklearn import preprocessing
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white") #white background style for seaborn plots
sns.set(style="whitegrid", color_codes=True)


# ## 2. Aggregate Player Stats Data <a class="anchor" id="2-bullet"></a>

# In[ ]:


attendance_valuation_elo_df = pd.read_csv("../input/nba_2017_att_val_elo.csv")
attendance_valuation_elo_df.head()


# In[ ]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv")
salary_df.head()


# In[ ]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv")
pie_df.head()


# In[ ]:


list(pie_df)


# In[ ]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv")
plus_minus_df.head()


# In[ ]:


br_stats_df = pd.read_csv("../input/nba_2017_br.csv")
br_stats_df.head()


# In[ ]:


list(br_stats_df)


# In[ ]:


### Remove Position Abbreviation from Name Field 

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
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head()


# In[ ]:


diff = list(set(nba_players_df["PLAYER"].values.tolist()) - set(salary_df["PLAYER"].values.tolist()))
len(diff)


# **111 is almost a quarter of the total values.  Imputing this many entries could skew our model results and give us inaccurate out of sample testing results.   We'll just remove the players without salary data from our final dataframe.**

# In[ ]:


nba_players_with_salary_df = nba_players_df.merge(salary_df)


# In[ ]:


nba_players_with_salary_df.isnull().sum()


# ## 3. Exploratory Data Analysis <a class="anchor" id="3-bullet"></a>

# ## 3.1 EDA: Correlation Heatmap <a class="anchor" id="3.1-bullet"></a>

# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY)")
corr = nba_players_with_salary_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Blues")


# [https://matplotlib.org/examples/color/colormaps_reference.html](http://)

# ## 3.2 Salary by Position <a class="anchor" id="3.2-bullet"></a>

# In[ ]:


print(set(nba_players_with_salary_df["POSITION"]))


# In[ ]:


sal_SF = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="SF","SALARY_MILLIONS"]
sal_SG = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="SG","SALARY_MILLIONS"]
sal_C = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="C","SALARY_MILLIONS"]
sal_PFC = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="PFC","SALARY_MILLIONS"]
sal_PF = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="PF","SALARY_MILLIONS"]
sal_PG = nba_players_with_salary_df.loc[nba_players_with_salary_df.POSITION=="PG","SALARY_MILLIONS"]


# First, how many per each position?

# In[ ]:


sns.countplot(x='POSITION',data=nba_players_with_salary_df, palette="Set2")
plt.show()


# In[ ]:


plt.figure(figsize=(15,8))
sns.kdeplot(sal_SF, color="darkturquoise", shade=False)
sns.kdeplot(sal_SG, color="lightcoral", shade=False)
sns.kdeplot(sal_C, color="forestgreen", shade=False)
sns.kdeplot(sal_PFC, color="dimgray", shade=False)
sns.kdeplot(sal_PF, color="gold", shade=False)
sns.kdeplot(sal_PF, color="darkorchid", shade=False)
sns.kdeplot(sal_PG, color="maroon", shade=False)
plt.legend(['SF', 'SG', 'C', 'PF-C', 'PF', 'PG'])
plt.title('Density Plot of Salary by Position')
plt.show()


# The distribution of salary is pretty consistent across most positions.  A few noteable differences include that Centers and Small Forwards are slightly more right skewed (a higher proportion make a high salary) than other positions.  Small Guards appear the least right skewed.  Let's look at the box-and-whisker plots to see the comparison more clearly.

# In[ ]:


plt.figure(figsize=(10,8))
sns.boxplot(x="SALARY_MILLIONS", y="POSITION",data=nba_players_with_salary_df, orient="h",palette="Set2")
plt.show()


# ## 3.3 Salary by team <a class="anchor" id="3.3-bullet"></a>

# In[ ]:


print(set(nba_players_with_salary_df["TEAM"]))


# In[ ]:


plt.figure(figsize=(8,15))
sns.boxplot(x="SALARY_MILLIONS", y="TEAM",data=nba_players_with_salary_df, orient="h")
plt.show()


# ## 3.4 Salary vs. Age <a class="anchor" id="3.4-bullet"></a>

# In[ ]:


sns.kdeplot(nba_players_with_salary_df["AGE"], color="mediumpurple", shade=True)
plt.show()


# In[ ]:


sns.lmplot(x="AGE", y="SALARY_MILLIONS", data=nba_players_with_salary_df)
plt.show()


# In[ ]:


results1 = smf.ols('SALARY_MILLIONS ~ AGE', data=nba_players_with_salary_df).fit()
print(results1.summary())


# ### Interpretation:

# ## 3.5 Salary vs. Wins RPM <a class="anchor" id="3.5-bullet"></a>

# In[ ]:


sns.kdeplot(nba_players_with_salary_df["WINS_RPM"], color="darkmagenta", shade=True)
plt.show()


# In[ ]:


sns.lmplot(x="WINS_RPM", y="SALARY_MILLIONS", data=nba_players_with_salary_df)
plt.show()


# In[ ]:


results2 = smf.ols('SALARY_MILLIONS ~ WINS_RPM', data=nba_players_with_salary_df).fit()
print(results2.summary())


# ### Interpretation:

# ## 3.6 Salary vs. Minutes Played Per Game <a class="anchor" id="3.6-bullet"></a>

# In[ ]:


sns.kdeplot(nba_players_with_salary_df["MPG"], color="dodgerblue", shade=True)
plt.show()


# In[ ]:


sns.lmplot(x="MPG", y="SALARY_MILLIONS", data=nba_players_with_salary_df)
plt.show()


# In[ ]:


results3 = smf.ols('SALARY_MILLIONS ~ MPG', data=nba_players_with_salary_df).fit()
print(results3.summary())


# ## 3.7 Salary vs. Points Per Game <a class="anchor" id="3.7-bullet"></a>

# In[ ]:


sns.kdeplot(nba_players_with_salary_df["POINTS"], color="darkturquoise", shade=True)
plt.show()


# In[ ]:


sns.lmplot(x="POINTS", y="SALARY_MILLIONS", data=nba_players_with_salary_df)
plt.show()


# In[ ]:


results4 = smf.ols('SALARY_MILLIONS ~ POINTS', data=nba_players_with_salary_df).fit()
print(results4.summary())


# ## 3.8 Offensive Rebounds and Steals <a class="anchor" id="3.8-bullet"></a>

# By looking at our heatmap, we can see that there are several fields where there is a noticeable difference in correlation for a particular variable vs. Salary as compared to that variable vs. Wins_RPM.   Just based on the visual, the two most apparent differences are for offensive rebounds and steals.

# In[ ]:


from pydoc import help
from scipy.stats.stats import pearsonr


# In[ ]:


pearsonr(nba_players_with_salary_df["ORB"], nba_players_with_salary_df["SALARY_MILLIONS"])
## Returns the Pearson's correlation coefficient and the 2-tailed p-value


# In[ ]:


pearsonr(nba_players_with_salary_df["ORB"], nba_players_with_salary_df["WINS_RPM"])


# <div>
# <div class="alert alert-block alert-success">
# ### There is a stronger correlation between Offensive Rebounds and WINS_RPM than there is between Offensive Rebounds and Salary.  <br> <br> 
# ### Offensive Rebounds vs. Salary --> 0.265 <br>
# ### Offensive Rebounds vs. Wins_RPM --> 0.376
# <br>
# ### This might mean that players who are strong on the offensive glass are more valuable to their franchise than their salary indicates.  They're helping their team win more games, as measured by WINS_RPM, than their salary reflects.

# In[ ]:


pearsonr(nba_players_with_salary_df["STL"], nba_players_with_salary_df["SALARY_MILLIONS"])


# In[ ]:


pearsonr(nba_players_with_salary_df["STL"], nba_players_with_salary_df["WINS_RPM"])


# <div>
# <div class="alert alert-block alert-success">
# ### Similarly, number of steals is a stronger predictor of a player's WINS_RPM than it is for Salary.  If franchises want to reward players for their contribution to wins, they might want to do better at rewarding players who make more steals.  <br>
# ### Steals vs. Salary --> 0.447 <br>
# ### Steals vs. Wins_RPM --> 0.672

# ## 4. Wikipedia Page Views & Twitter Following <a class="anchor" id="4-bullet"></a>

# In[ ]:


from ggplot import *


# In[ ]:


p = ggplot(nba_players_with_salary_df,aes(x="POINTS", y="WINS_RPM", color="SALARY_MILLIONS")) + geom_point(size=200)
p + xlab("POINTS/GAME") + ylab("WINS/RPM") + ggtitle("NBA Players 2016-2017:  POINTS/GAME, WINS REAL PLUS MINUS and SALARY")


# In[ ]:


wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()


# In[ ]:


wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)


# In[ ]:


median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]
median_wiki_df_small = median_wiki_df_small.reset_index()
nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)


# In[ ]:


twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head()


# In[ ]:


nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)


# ## 4.1 Name Length vs. Page Views <a class="anchor" id="4.1-bullet"></a>

# ### Adding in a "name length" variable (hypothesis- people don't want to type and search for long names)

# In[ ]:


nba_players_with_salary_wiki_twitter_df['name_length']=nba_players_with_salary_wiki_twitter_df['PLAYER'].str.len()


# In[ ]:


nba_players_with_salary_wiki_twitter_df.head()


# In[ ]:


sns.lmplot(x="name_length", y="PAGEVIEWS", data=nba_players_with_salary_wiki_twitter_df)
plt.show()


# In[ ]:


results_name = smf.ols('PAGEVIEWS ~ name_length', data=nba_players_with_salary_wiki_twitter_df).fit()
print(results_name.summary())


# ### Nope. No visible relationship.

# ## 4.2 EDA- Correlation Heatmap (Page views, Twitter, & Player Stats) <a class="anchor" id="4.2-bullet"></a>

# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap="Blues")
plt.show()


# In[ ]:




