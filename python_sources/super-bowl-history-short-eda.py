#!/usr/bin/env python
# coding: utf-8

# A quick exploratory data analysis of Super Bowls. 

# In[ ]:


# Import packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[ ]:


# Create dataframe
df = pd.read_csv("../input/superbowl-history-1967-2020/superbowl.csv")


# In[ ]:


# Check columns
df.columns


# In[ ]:


df.info()


# In[ ]:


df.head()


# In[ ]:


df.describe()


# The data is pretty bland as is. Let's spice it up a bit.

# In[ ]:


# Add point differential, total points, year
df['Pt_Diff'] = df['Winner Pts'] - df['Loser Pts']
df['Total_Pts'] = df['Winner Pts'] + df['Loser Pts']
df['Year'] = df['Date'].str[-4:]
df['Game'] = df['Winner'] + [' v '] + df['Loser'] + [' '] + df['Year']


# In[ ]:


df.head()


# Continue with data aggregated by team.

# In[ ]:


# Record data
team_records = pd.concat([df['Winner'].value_counts(), df['Loser'].value_counts()], axis=1)
team_records = team_records.fillna(0)
team_records = team_records.rename(columns={'Winner':'Wins', 'Loser':'Losses'})
team_records['Games'] = team_records['Wins'] + team_records['Losses']
team_records['Pct'] = team_records['Wins'] / team_records['Games']
team_records.sort_values(by=['Games', 'Wins'], ascending=False)


# At the time of writing, only 4 teams have never made it to the Super Bowl:
# * Cleveland Browns
# * Detriot Lions
# * Houston Texans
# * Jacksonville Jaguars

# In[ ]:


# Teams who have been in a Super Bowl but never won
team_records[team_records['Wins'] == 0]


# The results above show 9 teams in addition to the 4 who have never made it to the Super Bowl. However, the Rams franchise won a Super Bowl in 2000 when the team was franchised in St. Louis. Consequently, there are only 12 franchises who have never won a Super Bowl.

# In[ ]:


# Add to team data
team_pts = pd.concat([df['Winner'], df['Winner Pts'], df['Loser Pts']], axis=1)
team_pts = team_pts.rename(columns={'Winner':'Team', 'Winner Pts':'Avg_Pts_For', 'Loser Pts':'Avg_Pts_Against'})
team_pts2 = pd.concat([df['Loser'], df['Loser Pts'], df['Winner Pts']], axis=1)
team_pts2 = team_pts2.rename(columns={'Loser':'Team', 'Loser Pts':'Avg_Pts_For', 'Winner Pts':'Avg_Pts_Against'})
team_pts = team_pts.append(team_pts2, ignore_index=True)
team_pt_avgs = team_pts.groupby(team_pts['Team']).mean()
team_pt_avgs['Avg_Diff'] = team_pt_avgs['Avg_Pts_For'] - team_pt_avgs['Avg_Pts_Against']
team_stats = pd.concat([team_records['Wins']
          ,team_records['Losses']
          ,team_records['Games']
          ,team_records['Pct']
          ,team_pt_avgs['Avg_Pts_For']
          ,team_pt_avgs['Avg_Pts_Against']
          ,team_pt_avgs['Avg_Diff']], axis=1)
team_stats.sort_values(by=['Games', 'Wins'], ascending=False)


# Interestingly, the New England Patriots, one of the most dominant teams in recent memory under quarterback (QB) Tom Brady and head coach (HC) Bill Belichick, have an average point differential of -3.27. This is largely due to a 10 to 46 loss against the Chicago Bears in 1986. 
# 
# Some impressive point differentials include those of the Dallas Cowboys, San Francisco 49ers, and Green Bay Packers. The Cowboys won two Super Bowls in the 70s and three more in the 90s. The 49ers' success throughout the 80s and 90s came mostly from legendary QB Joe Montana, HC Bill Walsh, and later wide receiver (WR) Jerry Rice. The Packers won the first two SBs ever under two-time SB MVP and QB Bart Starr. 

# In[ ]:


# Winningest SB teams
plt_data = team_stats.sort_values(by='Wins', ascending=False)
plt.bar(plt_data['Wins'].head(10).index, plt_data['Wins'].head(10));
plt.xticks(rotation=75);


# In[ ]:


# Line plot of winning and losing points
line_data = pd.concat([df['Winner Pts'], df['Loser Pts']], axis=1)
fig, ax = plt.subplots(figsize=(14, 6))
sns.lineplot(data=line_data, linewidth=2.5);

