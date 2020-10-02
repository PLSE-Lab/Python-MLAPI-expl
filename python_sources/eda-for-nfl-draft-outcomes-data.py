#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# In this exploratory data analysis, I will be looking at NFL Draft outcomes data from 1985 to 2015. I will be making a concerted effort to use the seaborn package because I want to learn it better, it is pretty easy to use, and, in my opinion, it makes beautiful images. I will also try to explore some different drafting strategies. Please upvote if you find this useful. I hope you enjoy!
# 
# ## Inspecting the data set

# In[ ]:


# importing necessary modules
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns

# read data
df = pd.read_csv("../input/nfl_draft.csv")

# inspect data
print(df.shape)
df.head()


# There were 8435 players drafted between 1985 and 2015 and 33 columns describing each player.

# In[ ]:


years = np.arange(1985, 2016, 1)
g = sns.factorplot(x = "Year", data = df, kind = "count",
                   palette = "YlGn", size = 6, aspect = 1.5, order = years)
g.set_xticklabels(step = 5)


# After 1992, the number of players drafted decreased from over 300 to around 250.

# In[ ]:


n_rounds = np.zeros((len(years), ))
for i, year in enumerate(years) :
    n_rounds[i] = df["Rnd"][df["Year"] == year].unique().shape[0]
rounds_per_year = pd.DataFrame({"years" : years, "n_rounds" : n_rounds})
g = sns.factorplot(x = "years", y = "n_rounds", data = rounds_per_year, kind = "bar",
                   palette = "YlGn", size = 6, aspect = 1.5, order = years)
g.set_xticklabels(step = 5)


# The number of players drafted per year decreased after 1992 because they changed the number of rounds from 12 to 7. There is one exception. In 1993, only one round was recorded. However, the preceding plot shows that 224 players were drafted this year (https://en.wikipedia.org/wiki/1993_NFL_Draft). Let's see who the 40th pick was (spoiler alert: it should be Michael Strahan).

# In[ ]:


df[df["Year"] == 1993][df["Pick"] == 40]


# Two things to note here. First, the data correctly has Michael Strahan as the 40th pick. Next, the round value is missing. Finally, his defensive statistics (Tkl, Def_Int, and Sk) are missing as well. I could easily fill in the missing rounds values but it would be difficult to fill in the missing defensive statistics. As such, I'm going to leave 1993 data out of my analyses.

# In[ ]:


print(df[df["Year"] == 1993]["Tkl"].fillna(-1).value_counts())
print(df[df["Year"] == 1993]["Def_Int"].fillna(-1).value_counts())
print(df[df["Year"] == 1993]["Sk"].fillna(-1).value_counts())


# Note, however, that it seems like only defensive statistics are missing.

# # Some initial explorations

# In[ ]:


# analyze distributions of Age and First4AV and their correlations
sns.jointplot(x = "Age", y = "First4AV", data = df, size = 5)


# In[ ]:


# analyze First4AV by Age
sns.boxplot(x = "Age", y = "First4AV", data = df)


# Draft players as young as you can.

# In[ ]:


# analyze distributions of Rnd and First4AV and their correlations
sns.jointplot(x = "Rnd", y = "First4AV", data = df, size = 5)


# In[ ]:


# analyze First4AV by Rnd
sns.boxplot(x = "Rnd", y = "First4AV", data = df)


# Early rounds tend to produce better NFL players, obviously.

# In[ ]:


# violin plot of First4Av by Rnd
sns.violinplot(x = "Rnd", y = "First4AV", data = df, size = 6)


# Early rounds tend to produce a wider range of talents.

# In[ ]:


# analyze PB by Rnd
sns.boxplot(x = "PB", y = "First4AV", data = df)


# I accidentally stumbled onto this relationship. Seems like the number of pro-bowl selections is a good predictor for First4AV.

# # Data structure

# In[ ]:


df.shape


# In[ ]:


df.columns.values


# # Feature engineering
# ## Years played (pre-2000)
# (Because most players selected pre-2000 are now out of the league)

# In[ ]:


df_before_2000 = df[df["Year"] < 2000]
df_before_2000["Years_Played"] = df_before_2000["To"] - df_before_2000["Year"]


# In[ ]:


# analyze distributions of Rnd and Years_Played and their correlations
sns.jointplot(x = "Rnd", y = "Years_Played", data = df_before_2000, size = 5)


# In[ ]:


# analyze Years_Played by Rnd
sns.boxplot(x = "Rnd", y = "Years_Played", data = df_before_2000)


# Players selected in earlier rounds stick around longer in the league.

# In[ ]:


# violin plot of Years_Played by Rnd
sns.violinplot(x = "Rnd", y = "Years_Played", data = df_before_2000, size = 6)


# Earlier picks tend to have a wider range of longetivities.

# # Case study: should you select a RB in the 1st round?

# In[ ]:


df_rb = df[df["Pos"] == "RB"]


# In[ ]:


df_rb.head()


# ## Calculate rushing stats per game

# In[ ]:


df_rb["Rush_Att_G"] = df_rb["Rush_Att"] / df_rb["G"]
df_rb["Rush_Yds_G"] = df_rb["Rush_Yds"] / df_rb["G"]
df_rb["Rush_TDs_G"] = df_rb["Rush_TDs"] / df_rb["G"]
df_rb["Rec_G"] = df_rb["Rec"] / df_rb["G"]
df_rb["Rec_Yds_G"] = df_rb["Rec_Yds"] / df_rb["G"]
df_rb["Rec_Tds_G"] = df_rb["Rec_Tds"] / df_rb["G"]


# In[ ]:


# total purpose yards per game
df_rb["TPY_G"] = df_rb["Rush_Yds_G"] + df_rb["Rec_Yds_G"]


# In[ ]:


# analyze distributions of Rnd and Rush_Yds_G and their correlations
sns.jointplot(x = "Rnd", y = "TPY_G", data = df_rb, size = 5)


# In[ ]:


# analyze TPY_G by Rnd
sns.boxplot(x = "Rnd", y = "TPY_G", data = df_rb)


# There seems to be a decent production dropoff from round 1 to 2. Let's take a look at TDs, which are directly related to team scoring and therefore more closely related to wins and losses.

# In[ ]:


# total purpose TDs per game
df_rb["TPTD_G"] = df_rb["Rush_TDs_G"] + df_rb["Rec_Tds_G"]


# In[ ]:


# analyze TPTD_G by Rnd
sns.boxplot(x = "Rnd", y = "TPTD_G", data = df_rb)


# It is clear that there is a dropoff in scoring as Rnd increases. Therefore, at first glance, it seems appropriate that GMs should consider drafting a RB in the 1st round. What's interesting here is that, historically, you get better value selecting a RB in the 4th round than doing so in the 3rd.

# # Case study: Do late-round QBs have any chance of starting in their careers?

# In[ ]:


df_qb = df[df["Pos"] == "QB"]


# In[ ]:


# analyze G by Rnd
sns.boxplot(x = "Rnd", y = "G", data = df_qb)


# A GMs best bet is to take a QB in the 1st round as the mean games played for Rnd 1 is higher than the 75% for all other rounds except for Rnd 2.
