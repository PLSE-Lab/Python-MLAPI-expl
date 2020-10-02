#!/usr/bin/env python
# coding: utf-8

# ***Glossary of the dataset:*** 
# 
# """Home - 1 is Home matches, 0 is Away matches
# MP Minutes Played, 
# FG Field Goals Made,
# FGA Field Goals Attempted, 
# FGP Field Goal Percentage, 
# TP  3 Point Field Goals Made, 
# TPA 3 Point Field Goals Attempted, 
# TPP 3 Point Field Goals Percentage,
# FT Free Throws Made, 
# FTA Free Throws Attempted, 
# FTP Free Throw Percentage, 
# ORB Offensive Rebounds,
# DRB Defensive Rebounds, 
# TRB Total Rebounds, 
# AST Assists, 
# TOV Turnovers, 
# STL Steals, 
# BLK Blocks, 
# PF Personal Fouls, 
# PTS Points """

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))


# In[ ]:


champion = pd.read_csv('../input/champs_and_runner_ups_series_averages.csv')


# In[ ]:


champion.head()


# In[ ]:


champion.shape


# In[ ]:


len(champion)


# In[ ]:


champion.columns


# In[ ]:


champion.size


# In[ ]:


champion[['FGP','TPP','FTP']].round(2).head()


# In[ ]:


champion.isnull().sum().sum()


# In[ ]:


champion.isnull().sum().idxmax()


# In[ ]:


champion['Team'].unique().tolist()


# In[ ]:


champion.groupby(['Year','Team','Game', 'Home'], as_index=True).agg({"FG": "sum", "FGA": "sum","Win": "max", "PTS": "sum"})


# In[ ]:


champion.groupby('Team')[['Game','Win','PTS']].sum()


# In[ ]:


# Getting specific rows where the values in column 'Home' = 1,
champion[(champion.Home == 1) & (champion.index.isin([0,2,4,6,8,10]))]


# In[ ]:


# inverse operator, returns all rows where values Home =1 is not present in 0,2,4,6,8,10
champion[~((champion.Home == 1) & (champion.index.isin([0,2,4,6,8,10])))].head()


# In[ ]:


# using loc and select 1:4 will get a different result than using iloc to select rows 1:4.
champion.loc[1:4]


# In[ ]:


champion.iloc[1:4]


# In[ ]:


champion.iloc[1, 3]


# In[ ]:


champion.loc[0, ['Team', 'Year', 'Game']]


# In[ ]:


champion.loc[[0, 10, 19], :] 
# returns 1st,11th & 20th row and all columns


# In[ ]:


# iloc[row slicing, column slicing]
champion.iloc[1:6, 1:4]


# In[ ]:


# idxmin() to get the index of the min of PTS
champion.loc[champion.groupby("Team")["PTS"].idxmin()]


# In[ ]:


# alternate method to idxmin()
champion.sort_values(by="PTS").groupby("Team", as_index=False).first()


# In[ ]:


#  idxmax() to get the index of the max of PTS
champion.loc[champion.groupby("Team")["PTS"].idxmax()]


# In[ ]:


# alternate method to idxmax()
champion.sort_values(by="PTS").groupby("Team", as_index=False).last()


# In[ ]:


champion.sort_values(by=(['Team', 'PTS']), ascending=False).head()


# In[ ]:


# select all rows that have a Win = 1
champion[champion.Win == 1].head()


# In[ ]:


# select all rows that do not contain Win = 1, ie. returns all rows with Win = 0. 
# Win = 0 are the losers
champion[champion.Win != 1].head()


# In[ ]:


# finds all rows with Team = 'Lakers'
champion[champion['Team'].isin(['Lakers'])].head()


# In[ ]:


champion.groupby('Year')['Win'].count().sum()


# In[ ]:


year_wins = pd.crosstab(champion.Win, champion.Year, margins=True)
year_wins.T


# In[ ]:


dfhome_1 = champion[champion['Home'] == 1 ]
dfhome_1.head()


# In[ ]:


champion.groupby('Team')['Win'].agg(np.sum).plot(kind = 'bar')


# In[ ]:


over = champion.groupby('Team', as_index=True).agg({"PTS": "sum"})
over['PTS'].plot(kind='bar')


# In[ ]:


champion[(champion['Home']>0) & (champion['Year'] == 2000) & (champion['Team'] == 'Lakers')]


# In[ ]:


champion[(champion['Home']>0) & (champion['Team'] == 'Lakers') | (champion['Team'] == 'Bulls')].head()


# In[ ]:


champion[champion["Team"] == "Lakers"]["PTS"].value_counts().plot(kind="bar")


# In[ ]:


champion.loc[10:15]


# In[ ]:


champion.iloc[5:10,0:7] 


# In[ ]:


champion.Win.nlargest(5)


# In[ ]:


# check the type of an object like this
type(champion.Win.nlargest(5))


# In[ ]:


champion['Win'].nlargest(5).dtype


# In[ ]:


champion.index


# In[ ]:


champion.loc[6]


# In[ ]:


champion_pieces = [champion[:3], champion[3:7], champion[7:5]]
champion_pieces


# In[ ]:


champion[champion['TPP'].notnull()].head()


# In[ ]:


champion[champion['TPP'].isnull()]


# In[ ]:


champion.ix[2, 'Team']


# In[ ]:


champion.Team.ix[2]


# In[ ]:


champion.describe()   


# In[ ]:


champion.info()   


# In[ ]:


champion.Team.str.len()


# In[ ]:


champion.groupby('Team').agg(['min', 'max'])


# In[ ]:


# Min PTS by all teams in Home = 0 (away matches) and 1(home matches)
table = pd.pivot_table(champion,values=['PTS'],index=['Home'],columns=['Team'],aggfunc=np.min,margins=True)
table.T


# In[ ]:


# Max PTS by all teams in Home = 0 (away matches) and 1(home matches)
table = pd.pivot_table(champion,values=['PTS'],index=['Home'],columns=['Team'],aggfunc=np.max,margins=True)
table.T


# In[ ]:


champion.groupby(['Team','Year']).sum()


# In[ ]:


champion.groupby(['Year']).groups.keys()


# In[ ]:


len(champion.groupby(['Year']).groups[1980])


# In[ ]:


champ_runnerup = pd.read_csv('../input/champs_and_runner_ups_series_averages.csv')
champ_runnerup.head()


# In[ ]:


champ_runnerup.shape


# In[ ]:


champ_runnerup.columns


# In[ ]:


champ_runnerup[['Year', 'Status','Team','PTS']]


# In[ ]:


# convert the PTS field from float to an integer 
champ_runnerup['PTS'] = champ_runnerup['PTS'].astype('int64')
champ_runnerup['PTS'].dtype


# In[ ]:


champ_runnerup[['Year', 'Status','Team','PTS']].head()


# In[ ]:


champs = champ_runnerup[champ_runnerup['Status'] == 'Champion'].groupby('Year') ['Team'].sum()
champs.head()


# In[ ]:


ch = champs.value_counts()
ch1 = ch.to_frame().reset_index()
ch1


# In[ ]:


type(ch1)


# In[ ]:


runnerup = champ_runnerup[champ_runnerup['Status'] == 'Runner Up'].groupby('Year') ['Team'].sum()
runnerup.head()


# In[ ]:


ru = runnerup.value_counts()
ru1 = ru.to_frame().reset_index()
ru1


# In[ ]:


finalteams = pd.merge(ch1,ru1, on = 'index', how = 'outer')
finalteams


# In[ ]:


# Finalists in the tournament
final_teams = pd.concat([ch1, ru1], axis=1, ignore_index=True)
final_teams


# In[ ]:


# Finding Nan values 
final_teams[final_teams.isnull().any(axis=1)]


# In[ ]:


final_teams[pd.isnull(final_teams).any(axis=1)]


# In[ ]:


gc = champ_runnerup.groupby(['Status'])
gc.get_group('Runner Up').head().round()

