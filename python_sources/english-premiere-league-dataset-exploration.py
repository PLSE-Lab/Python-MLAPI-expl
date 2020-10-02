#!/usr/bin/env python
# coding: utf-8

# # English Premiere League Dataset Exploration
# 
# The site - https://datahub.io/sports-data/english-premier-league#pandas has EPL data for all years begining from 2009/2010 season. For this exercise, I will be using data from 2016/2016 onwards. 
# This kernel will focus on how to integrate data, and looks at whether data speaks for the intuition we know about the game. 
# 
# Below are the descriptions of the columns used - 
# 
# * FTHG = The total number of goals scored by the home team during the match at full time.
# * FTAG = The total number of goals scored by the away team during the match at half time.
# * FTR = The full time result, denoted as 'H' for home team win, 'A' for away team win, or 'D' for draw,
# * HTHG = The total number of goals scored by the home team at half time.
# * HTAG = The total number of goals scored by the away team at half time.
# * HTR = The half time result, denoted 'H' for home team advantage, 'A' for away team advantage, or 'D' for draw.
# * HomeTeam = Home Team 
# * AwayTeam = Away Team
# * B365H = Bet365 home win odds
# * B365D = Bet365 draw odds
# * B365A = Bet365 away win odds
# * LBH = Ladbrokes home win odds
# * LBD = Ladbrokes draw odds
# * LBA = Ladbrokes away win odds
# * FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)

# ### Importing Relevant Libraries

# In[ ]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


print(os.getcwd())


# ### Data Collection and Preprocessing
# 
# #### 1) Collecting data for 4 years - 2015-2019

# In[ ]:


file1819 = ('../input/season-1819_csv.csv')
file1718 = ('../input/season-1718_csv.csv')
file1617 = ('../input/season-1617_csv.csv')
file1516 = ('../input/season-1516_csv.csv')


# In[ ]:


df_1819 = pd.read_csv(file1819, header=0)
df_1718 = pd.read_csv(file1718, header=0)
df_1617 = pd.read_csv(file1718, header=0)
df_1516 = pd.read_csv(file1516, header=0)


# #### 2) Dropping data so columns are consistent throughout the years. The columns dropped are LBH, LBD and LBA which we wont be using in our analysis. 

# In[ ]:


df_1819.shape


# In[ ]:


df_1718 = df_1718.drop(['LBH', 'LBD', 'LBA'],axis=1)
df_1617 = df_1617.drop(['LBH', 'LBD', 'LBA'],axis=1)
df_1516 = df_1516.drop(['LBH', 'LBD', 'LBA'],axis=1)


# #### 3) Creating a combined dataframe for all years 

# In[ ]:


frames = [df_1819, df_1718, df_1617,df_1516]
epl_df = pd.concat(frames)
epl_df.shape


# In[ ]:


epl_df.head(5)


# In[ ]:


epl_df.columns


# #### 4) Create HomeWin, AwayWin and Draw Columns using the FTHG, FTAG columns 

# In[ ]:


epl_df = epl_df.assign(HomeWin=lambda epl_df: epl_df.apply(lambda row: 1 if row.FTHG > row.FTAG else 0, axis='columns'),
              Draw=lambda epl_df: epl_df.apply(lambda row: 1 if row.FTHG == row.FTAG else 0, axis='columns'),
              AwayWin=lambda epl_df: epl_df.apply(lambda row: 1 if row.FTHG < row.FTAG else 0, axis='columns'))
epl_df.head(10)


# #### 5) Adding 'Year' column to see past performances

# In[ ]:


epl_df['Year'] = pd.DatetimeIndex(epl_df['Date']).year
epl_df.tail(10)


# ### Data Exploration
# 
# #### Home Vs Away
# 
# We know by intuition that home games always prove to be winning matches for the home team. Let's take a look if the data speaks for this as well.

# In[ ]:


win_rates = (epl_df.groupby('Year')
    .mean()
    .loc[:,['HomeWin', 'Draw', 'AwayWin']])

win_rates


# As seen, the Home Win % is fairly high except for 2014/2015. Let's plot this for better understanding

# In[ ]:


# Set the style
plot_cols = ['HomeWin', 'AwayWin', 'Draw']

fig, axes = plt.subplots(3,1, figsize=(20,15))
win_rates[plot_cols].plot(subplots=True, ax=axes)

for ax, col in zip(axes, plot_cols):

    # lets add horizontal zero lines
    ax.axhline(0.15, color='k', linestyle='-', linewidth=1)
    
    # add titles
    ax.set_title('Win Trend - ' + col)
    
    # add axis labels
    ax.set_ylabel('Win Probability')
    ax.set_xlabel('Year')


# #### Home Ground Advantage

# In[ ]:


home_win_df = (epl_df.groupby(['HomeTeam'])
    .HomeWin
    .mean())

away_win_df = (epl_df.groupby(['AwayTeam'])
    .AwayWin
    .mean())


# In[ ]:


home_grnd_adv = (home_win_df - away_win_df).reset_index().rename(columns={0: 'HomeGrndAdv'}).sort_values(by='HomeGrndAdv', ascending=False)
home_grnd_adv.head()


# This shows that Arsenal has had the most advantage in terms of home wins. Lets look at a few top of the table teams and their performance at home vs away 

# In[ ]:


big_clubs = ['Liverpool', 'Man City', 'Man United', 'Chelsea', 'Arsenal']
home_win_rates_5 = epl_df[epl_df.HomeTeam.isin(big_clubs)].groupby(['HomeTeam', 'Year']).HomeWin.mean()
away_win_rates_5 = epl_df[epl_df.AwayTeam.isin(big_clubs)].groupby(['AwayTeam', 'Year']).AwayWin.mean()
top_5 = home_win_rates_5 - away_win_rates_5

top_5.unstack(level=0)


# In[ ]:


fig, ax = plt.subplots(figsize=(20, 15))
sns.lineplot(x='Year', y='HomeGrndAdv', hue='Teams', data=top_5.reset_index().rename(columns={0: 'HomeGrndAdv', 'HomeTeam': 'Teams'}), ax=ax)
plt.legend(loc='lower right', ncol=6, bbox_to_anchor=(0.5, 0., 0.5, 0.5), fontsize='large')
plt.title("Home Ground Advantage for the Top 5 Clubs in the Table", fontsize=16)
plt.show()


# #### Refree Home Ground Bias
# 
# Let's see if there exists a refree home ground bias. 

# In[ ]:


print('Overall Home Win Rate: {:.4}%'.format(epl_df.HomeWin.mean() * 100))

# Get the top 10 Refs 
top_10_refs = epl_df.Referee.value_counts().head(10).index

epl_df[epl_df.Referee.isin(top_10_refs)].groupby('Referee').HomeWin.mean().sort_values(ascending=False)


# This shows that K. Friend is the most influenced by home crowd. With an Overall Win Rate of 45%, the win rate when Friend is refree is 51%.
# 
# #### This completes the intial data exploration on this English Premiere League Dataset
#                                                 * ------------- * -------------- *
# 
