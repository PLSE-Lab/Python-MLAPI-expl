#!/usr/bin/env python
# coding: utf-8

# Exploration of the stronger predictor for player impact estimate (PIE) between Salary and Social Media

# **Importing packagaes**

# In[ ]:


import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
color = sns.color_palette()
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


salary_df = pd.read_csv("../input/nba_2017_salary.csv");salary_df.head()


# In[ ]:


pie_df = pd.read_csv("../input/nba_2017_pie.csv");pie_df.head()


# In[ ]:


plus_minus_df = pd.read_csv("../input/nba_2017_real_plus_minus.csv");plus_minus_df.head()


# In[ ]:


br_stats_df = pd.read_csv("../input/nba_2017_br.csv");br_stats_df.head()


# **Exploratory data analysis**

# In[ ]:



plus_minus_df.rename(columns={"NAME":"PLAYER", "WINS": "WINS_RPM"}, inplace=True)
players = []
for player in plus_minus_df["PLAYER"]:
    plyr, _ = player.split(",")
    players.append(plyr)
plus_minus_df.drop(["PLAYER"], inplace=True, axis=1)
plus_minus_df["PLAYER"] = players
plus_minus_df.head(10)


# In[ ]:


nba_players_df = br_stats_df.copy()
nba_players_df.rename(columns={'Player': 'PLAYER','Pos':'POSITION', 'Tm': "TEAM", 'Age': 'AGE', "PS/G": "POINTS"}, inplace=True)
nba_players_df.drop(["G", "GS", "TEAM"], inplace=True, axis=1)
nba_players_df = nba_players_df.merge(plus_minus_df, how="inner", on="PLAYER")
nba_players_df.head(10)


# In[ ]:


pie_df_subset = pie_df[["PLAYER", "PIE", "PACE", "W"]].copy()
nba_players_df = nba_players_df.merge(pie_df_subset, how="inner", on="PLAYER")
nba_players_df.head(10)


# In[ ]:


salary_df.rename(columns={'NAME': 'PLAYER'}, inplace=True)
salary_df["SALARY_MILLIONS"] = round(salary_df["SALARY"]/1000000, 2)
salary_df.drop(["POSITION","TEAM", "SALARY"], inplace=True, axis=1)
salary_df.head(10)


# In[ ]:


nba_players_with_salary_df = nba_players_df.merge(salary_df); 
nba_players_with_salary_df.head(10)


# In[ ]:


wiki_df = pd.read_csv("../input/nba_2017_player_wikipedia.csv");wiki_df.head()


# In[ ]:


wiki_df.rename(columns={'names': 'PLAYER', "pageviews": "PAGEVIEWS"}, inplace=True)
wiki_df.head(10)


# In[ ]:


median_wiki_df = wiki_df.groupby("PLAYER").median()
median_wiki_df.head(10)


# In[ ]:


median_wiki_df_small = median_wiki_df[["PAGEVIEWS"]]


# In[ ]:


median_wiki_df_small = median_wiki_df_small.reset_index()


# In[ ]:


nba_players_with_salary_wiki_df = nba_players_with_salary_df.merge(median_wiki_df_small)
nba_players_with_salary_wiki_df.head(10)


# In[ ]:


twitter_df = pd.read_csv("../input/nba_2017_twitter_players.csv");twitter_df.head(10)


# In[ ]:


nba_players_with_salary_wiki_twitter_df = nba_players_with_salary_wiki_df.merge(twitter_df)


# In[ ]:


nba_players_with_salary_wiki_twitter_df.head(10)


# **Correlation matrix**

# In[ ]:


plt.subplots(figsize=(20,15))
ax = plt.axes()
ax.set_title("NBA Player Correlation Heatmap:  2016-2017 Season (STATS & SALARY & TWITTER & WIKIPEDIA)")
corr = nba_players_with_salary_wiki_twitter_df.corr()
sns.heatmap(corr, 
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values,
           cmap = "PuBu")


# **Linear regression**

# In[ ]:


results = smf.ols('PIE ~ PAGEVIEWS+TWITTER_FAVORITE_COUNT+TWITTER_RETWEET_COUNT', data = nba_players_with_salary_wiki_twitter_df).fit()


# In[ ]:


print(results.summary())


# In[ ]:


results = smf.ols('PIE ~ SALARY_MILLIONS', data = nba_players_with_salary_wiki_twitter_df).fit()


# In[ ]:


print(results.summary())


# In[ ]:


sns.heatmap(nba_players_with_salary_wiki_twitter_df[["SALARY_MILLIONS", "PAGEVIEWS", "TWITTER_FAVORITE_COUNT", "TWITTER_RETWEET_COUNT", "PIE"]].corr(), annot= True, cmap= "YlOrRd")


# In[ ]:


sns.lmplot(x= "SALARY_MILLIONS", y= "PIE", data= nba_players_with_salary_wiki_twitter_df, size= 10)

