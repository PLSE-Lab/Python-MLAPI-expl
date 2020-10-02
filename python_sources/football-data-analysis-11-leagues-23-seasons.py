#!/usr/bin/env python
# coding: utf-8

# # Football Data Analitics
# 
# **Introduction**
# 
# This kernel provide some basic analysis of football dataset for  **11 european leagues from 23 seasons**:
# 
# * **Premiership** seasons 1995-2018
# * **Championship** seasons 1997-2018
# * **Bundesliga 1** seasons 1995-2018
# * **Bundesliga 2** seasons 1996-2018
# * **Eredivisie** seasons 1997-2018
# * **Jupiler League** seasons 1995-2018
# * **League Un** seasons 1995-2018
# * **Portugal 1** seasons 1998-2018
# * **Primera Division** seasons 1995-2018
# * **Segunda Division** seasons 1997-2018
# * **Serie A** seasons 1995-2018

# In[ ]:


#imports
import numpy as np
import pandas as pd
import os

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt  # Matlab-style plotting
import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)
import matplotlib as mpl

mpl.rcParams['xtick.labelsize'] = 18
mpl.rcParams['ytick.labelsize'] = 18

print(os.listdir("../input"))


# In[ ]:


#What do we have here
data = pd.read_csv('../input/complete_football_dataset.csv')

data.head()


# In[ ]:


data.shape


# Okay, where to start?
# Let's check in which league we have the biggest chance to see goals?

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(24,10))

data1 = data.groupby(['Div'])['GS_full'].mean().sort_values(ascending=False).reset_index()
sns.barplot("Div", y='GS_full', data=data1, ax=ax[0])
ax[0].set_title('Goals Scored Full Time All Seasons', fontsize=20)
ax[0].set_xlabel("Division",fontsize=18)
ax[0].set_ylabel("Mean",fontsize=18)
ax[0].set_xticklabels(data1['Div'], rotation=90,fontsize=18)

data2 = data.groupby(['Div'])['GS_half'].mean().sort_values(ascending=False).reset_index()
sns.barplot("Div", y='GS_half', data=data2, ax=ax[1])
ax[1].set_title('Goals Scored Half Time All Seasons', fontsize=20)
ax[1].set_xlabel("Division",fontsize=18)
ax[1].set_ylabel("Mean",fontsize=18)
ax[1].set_xticklabels(data2['Div'], rotation=90,fontsize=18)


# Looks like the leading league in goals scoring is Eredivise
# In which season we had biggest chance to see goals?

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(24,10))

data1 = data.groupby(['season'])['GS_full'].mean().sort_values(ascending=False).reset_index()
sns.barplot('season', y='GS_full', data=data1, ax=ax[0])
ax[0].set_title('Goals Scored Full Time All Leagues', fontsize=20)
ax[0].set_xlabel("season",fontsize=18)
ax[0].set_ylabel("Mean",fontsize=18)
ax[0].set_xticklabels(data1['season'], rotation=90,fontsize=18)

data2 = data.groupby(['season'])['GS_half'].mean().sort_values(ascending=False).reset_index()
sns.barplot('season', y='GS_half', data=data2, ax=ax[1])
ax[1].set_title('Goals Scored Half Time All Leagues', fontsize=20)
ax[1].set_xlabel("season",fontsize=18)
ax[1].set_ylabel("Mean",fontsize=18)
ax[1].set_xticklabels(data2['season'], rotation=90,fontsize=18)


# Season 2012/2013 leading before season 2011/2012. Let's check those

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(24,10))

data1 = data[data.season == 'season-12-13'].groupby(['Div'])['GS_full'].mean().sort_values(ascending=False).reset_index()
sns.barplot('Div', y='GS_full', data=data1, ax=ax[0])
ax[0].set_title('Goals Scored for season 2012/2013', fontsize=20)
ax[0].set_xlabel("Division",fontsize=18)
ax[0].set_ylabel("Mean",fontsize=18)
ax[0].set_xticklabels(data1['Div'], rotation=90,fontsize=18)

data2 = data[data.season == 'season-11-12'].groupby(['Div'])['GS_full'].mean().sort_values(ascending=False).reset_index()
sns.barplot('Div', y='GS_full', data=data2, ax=ax[1])
ax[1].set_title('Goals Scored for season 2011/2012', fontsize=20)
ax[1].set_xlabel("Division",fontsize=18)
ax[1].set_ylabel("Mean",fontsize=18)
ax[1].set_xticklabels(data2['Div'], rotation=90,fontsize=18)


# Look's like Eredivise is still heading. What about last season 2017/2018?

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12,10))
data1 = data[data.season == 'season-17-18'].groupby(['Div'])['GS_full'].mean().sort_values(ascending=False).reset_index()
sns.barplot('Div', y='GS_full', data=data1, ax=ax)
ax.set_title('Goals Scored for season 2017/2018', fontsize=20)
ax.set_xlabel("Division",fontsize=18)
ax.set_ylabel("Mean",fontsize=18)
ax.set_xticklabels(data1['Div'], rotation=90,fontsize=18)


# Okay, let's check first five leagues. for this calcutation I assumed year as as season period eg. year 2005 is season 2005/2006, year 1997 is season 1997/1998 ect.

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12,10))
data1 = data[data.Div.isin(['Eredivise', 'Jupiler League', 'Bundesliga 1', 'Bundesliga 2', 'Primera Division'])].groupby(['Div', 'year'])['GS_full'].mean().reset_index()
sns.lineplot(x="year", y="GS_full", hue="Div", data=data1)
ax.set_title('Goals Scored Full Time In One Season', fontsize=20)
ax.set_xlabel("year",fontsize=18)
ax.set_ylabel("Mean",fontsize=18)
ax.legend(prop={'size': 15})


# Maybe we want to see also how single teams score

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(24,10))

palette = {"Eredivise":"#f7a431","Bundesliga 1":"#159e09","Bundesliga 2":"#43ff32", "Premiership":"#006a91", "Jupiler League":'#af0000',
          "Primera Division":"#ff5959", "Portugal 1":"#f8fc0a", "Championship":'#49ceff', "Segunda Division": "#ff5b5b", 
           "Serie A": "#38bc6d", "Ligue Un": "#e930f7"}

data1 = data.groupby(['Div', 'Team'])['GS_full'].mean().sort_values(ascending=False).reset_index().head(15)
sns.barplot(x="Team", y='GS_full', data=data1, hue='Div', dodge=False, palette=palette, ax=ax[0])
ax[0].set_title('Goals Scored Full Time', fontsize=20)
ax[0].set_xlabel("Team",fontsize=18)
ax[0].set_ylabel("Mean",fontsize=18)
ax[0].set_xticklabels(data1['Team'], rotation=90,fontsize=18)
ax[0].legend(prop={'size': 15})

data2 = data.groupby(['Div', 'Team'])['GS_half'].mean().sort_values(ascending=False).reset_index().head(15)
sns.barplot("Team", y='GS_half', data=data2, hue='Div', dodge=False, palette=palette, ax=ax[1])
ax[1].set_title('Goals Scored Half Time', fontsize=20)
ax[1].set_xlabel("Team",fontsize=18)
ax[1].set_ylabel("Mean",fontsize=18)
ax[1].set_xticklabels(data2['Team'], rotation=90,fontsize=18)
ax[1].legend(prop={'size': 15})


# Does anyone expected to see here "Holstain Kiel"?! Heading are of course Netherlands teams PSV and Ajax

# In[ ]:


#Lost?
fig, ax = plt.subplots(1, 2, figsize=(24,10))

data1 = data.groupby(['Div', 'Team'])['GL_full'].mean().sort_values(ascending=False).reset_index().head(15)
sns.barplot("Team", y='GL_full', data=data1, hue='Div', dodge=False, palette=palette, ax=ax[0])
ax[0].set_title('Goals Lost Full Time', fontsize=20)
ax[0].set_xlabel("Team",fontsize=18)
ax[0].set_ylabel("Mean",fontsize=18)
ax[0].set_xticklabels(data1['Team'], rotation=90,fontsize=18)
ax[0].legend(prop={'size': 15})

data2 = data.groupby(['Div', 'Team'])['GL_half'].mean().sort_values(ascending=False).reset_index().head(15)
sns.barplot("Team", y='GL_half', data=data2, hue='Div', dodge=False, palette=palette, ax=ax[1])
ax[1].set_title('Goals Lost Half Time', fontsize=20)
ax[1].set_xlabel("Team",fontsize=18)
ax[1].set_ylabel("Mean",fontsize=18)
ax[1].set_xticklabels(data2['Team'], rotation=90,fontsize=18)
ax[1].legend(prop={'size': 15})


# Starts to be a little exotic.... Ok, but what abou goals difference ratio? Let's check it also. 

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(24,10))

data1 = data.groupby(['Div', 'Team'])['SL_full'].mean().sort_values(ascending=False).reset_index().head(15)
sns.barplot("Team", y='SL_full', data=data1, hue='Div', dodge=False, palette=palette, ax=ax[0])
ax[0].set_title('Goals Score/Lost Ratio Full Time', fontsize=20)
ax[0].set_xlabel("Team",fontsize=18)
ax[0].set_ylabel("Mean",fontsize=18)
ax[0].set_xticklabels(data1['Team'], rotation=90,fontsize=18)
ax[0].legend(prop={'size': 15})

data2 = data.groupby(['Div', 'Team'])['SL_half'].mean().sort_values(ascending=False).reset_index().head(15)
sns.barplot("Team", y='SL_half', data=data2, hue='Div', dodge=False, palette=palette, ax=ax[1])
ax[1].set_title('Goals Score/Lost Ratio Half Time', fontsize=20)
ax[1].set_xlabel("Team",fontsize=18)
ax[1].set_ylabel("Mean",fontsize=18)
ax[1].set_xticklabels(data2['Team'], rotation=90,fontsize=18)
ax[1].legend(prop={'size': 15})


# Only Porto looks like improving. Barcelona nd PSV for Half Time have similar ratio. 
# We see so far that PSV scored well but it was the case all time? Let's chcek it with correlation with another heading clubs: Ajax, Porto, Barcelona, Bayern Munich

# In[ ]:


fig, ax = plt.subplots(1, 1, figsize=(12,10))
data1 = data[data.Team.isin(['PSV Eindhoven', 'Ajax', 'Porto', 'Barcelona', 'Bayern Munich'])].groupby(['Team', 'year'])['SL_full'].mean().reset_index()
sns.lineplot(x="year", y="SL_full", hue="Team", data=data1)
ax.set_title('Goals Scored/Lost Ratio Full Time', fontsize=20)
ax.set_xlabel("year",fontsize=18)
ax.set_ylabel("Mean",fontsize=18)
ax.legend(prop={'size': 15})


# It's not! Looks like PSV used to have good Goals Scored/Lost Ratio in the 1998-2005. We can see also strong improvement of Barcelona and Bayern form 2007. Oh BTW. how it looks if we compare long time rival of Barcelona: Real Madrid?

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(24,20))
palette={'Barcelona':'#9803b7', 'Real Madrid':'#050000'}

data0 = data[data.Team.isin(['Barcelona', 'Real Madrid'])].groupby(['Team', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y="PT", hue="Team", data=data0, palette=palette, ax=ax[0][0])
ax[0][0].set_title('Points for the end of the season', fontsize=20)
ax[0][0].set_xlabel("year",fontsize=18)
ax[0][0].set_ylabel("Total",fontsize=18)
ax[0][0].legend(prop={'size': 15})

data1 = data[data.Team.isin(['Barcelona', 'Real Madrid'])].groupby(['Team', 'year'])['SL_full'].mean().reset_index()
sns.lineplot(x="year", y="SL_full", hue="Team", data=data1, palette=palette, ax=ax[1][1])
ax[1][1].set_title('Goals Scored/Lost Ratio Full Time', fontsize=20)
ax[1][1].set_xlabel("year",fontsize=18)
ax[1][1].set_ylabel("Mean",fontsize=18)
ax[1][1].legend(prop={'size': 15})

data2 = data[data.Team.isin(['Barcelona', 'Real Madrid'])].groupby(['Team', 'year'])['GS_full'].mean().reset_index()
sns.lineplot(x="year", y='GS_full', hue='Team', data=data2, palette=palette, ax=ax[0][1])
ax[0][1].set_title('Goals Score Full Time', fontsize=20)
ax[0][1].set_xlabel("year",fontsize=18)
ax[0][1].set_ylabel("Mean",fontsize=18)
ax[0][1].legend(prop={'size': 15})

data3 = data[data.Team.isin(['Barcelona', 'Real Madrid'])].groupby(['Team', 'year'])['GL_full'].mean().reset_index()
sns.lineplot(x="year", y='GL_full', hue='Team', data=data3, palette=palette, ax=ax[1][0])
ax[1][0].set_title('Goals Lost Full Time', fontsize=20)
ax[1][0].set_xlabel("year",fontsize=18)
ax[1][0].set_ylabel("Mean",fontsize=18)
ax[1][0].legend(prop={'size': 15})


# Well, looks like slightly but... Barcelona. Really bad time for Real deffensive: 1998, 2003, 2008. Really bad time for Barcelona defensive: 1997, 2000, but fantastic for 2010 and 2014. I'm courius also about Manchesters...   

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(24,20))
palette={'Man United':'#b70303', 'Man City':'#0303b7'}

data0 = data[data.Team.isin(['Man City', 'Man United'])].groupby(['Team', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y="PT", hue="Team", data=data0, palette=palette, ax=ax[0][0])
ax[0][0].set_title('Points for the end of the season', fontsize=20)
ax[0][0].set_xlabel("year",fontsize=18)
ax[0][0].set_ylabel("Total",fontsize=18)
ax[0][0].legend(prop={'size': 15})

data1 = data[data.Team.isin(['Man City', 'Man United'])].groupby(['Team', 'year'])['SL_full'].mean().reset_index()
sns.lineplot(x="year", y="SL_full", hue="Team", data=data1, palette=palette, ax=ax[1][1])
ax[1][1].set_title('Goals Scored/Lost Ratio Full Time', fontsize=20)
ax[1][1].set_xlabel("year",fontsize=18)
ax[1][1].set_ylabel("Mean",fontsize=18)
ax[1][1].legend(prop={'size': 15})

data2 = data[data.Team.isin(['Man City', 'Man United'])].groupby(['Team', 'year'])['GS_full'].mean().reset_index()
sns.lineplot(x="year", y='GS_full', hue='Team', data=data2, palette=palette, ax=ax[0][1])
ax[0][1].set_title('Goals Score Full Time', fontsize=20)
ax[0][1].set_xlabel("year",fontsize=18)
ax[0][1].set_ylabel("Mean",fontsize=18)
ax[0][1].legend(prop={'size': 15})

data3 = data[data.Team.isin(['Man City', 'Man United'])].groupby(['Team', 'year'])['GL_full'].mean().reset_index()
sns.lineplot(x="year", y='GL_full', hue='Team', data=data3, palette=palette, ax=ax[1][0])
ax[1][0].set_title('Goals Lost Full Time', fontsize=20)
ax[1][0].set_xlabel("year",fontsize=18)
ax[1][0].set_ylabel("Mean",fontsize=18)
ax[1][0].legend(prop={'size': 15})


# Change of the lider in 2012! London?

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(24,20))
palette={'Arsenal': '#f94b0c', 'Chelsea':'#0c4bf9', 'Tottenham':'#17cfd6', 'West Ham':'#5e09a3'}

data0 = data[data.Team.isin(['Chelsea', 'Arsenal', 'Tottenham', 'West Ham'])].groupby(['Team', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y="PT", hue="Team", data=data0, palette=palette, ax=ax[0][0])
ax[0][0].set_title('Points for the end of the season', fontsize=20)
ax[0][0].set_xlabel("year",fontsize=18)
ax[0][0].set_ylabel("Total",fontsize=18)
ax[0][0].legend(prop={'size': 15})

data1 = data[data.Team.isin(['Chelsea', 'Arsenal', 'Tottenham', 'West Ham'])].groupby(['Team', 'year'])['SL_full'].mean().reset_index()
sns.lineplot(x="year", y="SL_full", hue="Team", data=data1, palette=palette, ax=ax[1][1])
ax[1][1].set_title('Goals Scored/Lost Ratio Full Time', fontsize=20)
ax[1][1].set_xlabel("year",fontsize=18)
ax[1][1].set_ylabel("Mean",fontsize=18)
ax[1][1].legend(prop={'size': 15})

data2 = data[data.Team.isin(['Chelsea', 'Arsenal', 'Tottenham', 'West Ham'])].groupby(['Team', 'year'])['GS_full'].mean().reset_index()
sns.lineplot(x="year", y='GS_full', hue='Team', data=data2, palette=palette, ax=ax[0][1])
ax[0][1].set_title('Goals Score Full Time', fontsize=20)
ax[0][1].set_xlabel("year",fontsize=18)
ax[0][1].set_ylabel("Mean",fontsize=18)
ax[0][1].legend(prop={'size': 15})

data3 = data[data.Team.isin(['Chelsea', 'Arsenal', 'Tottenham', 'West Ham'])].groupby(['Team', 'year'])['GL_full'].mean().reset_index()
sns.lineplot(x="year", y='GL_full', hue='Team', data=data3, palette=palette, ax=ax[1][0])
ax[1][0].set_title('Goals Lost Full Time', fontsize=20)
ax[1][0].set_xlabel("year",fontsize=18)
ax[1][0].set_ylabel("Mean",fontsize=18)
ax[1][0].legend(prop={'size': 15})


# Tottenham play so well last time?! Well I need to update my footbal news... 

# Ok, time to see teams position in the end of the season. Start from London?

# In[ ]:


fig, ax = plt.subplots(1, 2, figsize=(24,10))

data2 = data[(data.Team.isin(['Chelsea', 'Arsenal', 'Tottenham', 'West Ham']))  & (data.HA == 'Home')].groupby(['Team', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y='PT', hue='Team', data=data2, palette=palette, ax=ax[0])
ax[0].set_title('Home Points for Each Year', fontsize=20)
ax[0].set_xlabel("year",fontsize=18)
ax[0].set_ylabel("Total",fontsize=18)
ax[0].legend(prop={'size': 15})

data3 = data[(data.Team.isin(['Chelsea', 'Arsenal', 'Tottenham', 'West Ham']))  & (data.HA == 'Away')].groupby(['Team', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y='PT', hue='Team', data=data3, palette=palette, ax=ax[1])
ax[1].set_title('Away Points for Each Year', fontsize=20)
ax[1].set_xlabel("year",fontsize=18)
ax[1].set_ylabel("Total",fontsize=18)
ax[1].legend(prop={'size': 15})


# Let's make it more clear for Home and Away  

# In[ ]:


fig, ax = plt.subplots(2, 2, figsize=(24,20))
palette = {'Home':'#0c67f9', 'Away':'#f90c0c'}

data1 = data[data.Team.isin(['Chelsea'])].groupby(['HA', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y="PT", hue="HA", data=data1, palette=palette, ax=ax[0][0])
ax[0][0].set_title('Chelsea', fontsize=20)
ax[0][0].set_xlabel("year",fontsize=18)
ax[0][0].set_ylabel("Total",fontsize=18)
ax[0][0].legend(prop={'size': 15})

data2 = data[data.Team.isin(['Arsenal'])].groupby(['HA', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y='PT', hue='HA', data=data2, palette=palette, ax=ax[0][1])
ax[0][1].set_title('Arsenal', fontsize=20)
ax[0][1].set_xlabel("year",fontsize=18)
ax[0][1].set_ylabel("Total",fontsize=18)
ax[0][1].legend(prop={'size': 15})

data3 = data[data.Team.isin(['Tottenham'])].groupby(['HA', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y='PT', hue='HA', data=data3, palette=palette, ax=ax[1][0])
ax[1][0].set_title('Tottenham', fontsize=20)
ax[1][0].set_xlabel("year",fontsize=18)
ax[1][0].set_ylabel("Total",fontsize=18)
ax[1][0].legend(prop={'size': 15})

data3 = data[data.Team.isin(['West Ham'])].groupby(['HA', 'year'])['PT'].max().reset_index()
sns.lineplot(x="year", y='PT', hue='HA', data=data3, palette=palette, ax=ax[1][1])
ax[1][1].set_title('West Ham', fontsize=20)
ax[1][1].set_xlabel("year",fontsize=18)
ax[1][1].set_ylabel("Total",fontsize=18)
ax[1][1].legend(prop={'size': 15})


# Okay, as we are here. It's time to see some correlations between features. What about if we ask about correlation between differences in points before match and it;s impact on differences in score. Check!

# In[ ]:


sns.jointplot(x='PTBG_difer', y='SL_full', data=data, kind='hex', gridsize=20, xlim=[30, -30], ylim=[5, -5])


# Well, as you can see there is a lot to analyse here! I'll leave you here guys, hava a good analysis!

# In[ ]:
























