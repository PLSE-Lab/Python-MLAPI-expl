#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# # Reading/preprocessing the data
# 
# Reading the data using pandas. Each .csv correspond to one season in top-flight English football. Most data was taken from this Kaggle dataset: https://www.kaggle.com/siddhrath/football-match-dataset, the ones taht were not were 2002-03 to 2004-05 seasons, since the .csv uploaded there were apparently corrupted. Also, added season 2016-17 onwards, since that dataset ends in 2015-16. 

# In[ ]:


all_raw_data_1 = pd.read_csv('../input/football-match-dataset/2000-01.csv')
all_raw_data_2 = pd.read_csv('../input/football-match-dataset/2001-02.csv')
all_raw_data_3 = pd.read_csv('../input/2003to2005data/2002-03.csv')
all_raw_data_4 = pd.read_csv('../input/2003to2005data/2003-04.csv')
all_raw_data_5 = pd.read_csv('../input/2003to2005data/2004-05.csv',encoding = "ISO-8859-1") #need that encoding line for the code to work.
all_raw_data_6 = pd.read_csv('../input/football-match-dataset/2005-06.csv')
all_raw_data_7 = pd.read_csv('../input/football-match-dataset/2006-07.csv')
all_raw_data_8 = pd.read_csv('../input/football-match-dataset/2007-08.csv')
all_raw_data_9 = pd.read_csv('../input/football-match-dataset/2008-09.csv')
all_raw_data_10 = pd.read_csv('../input/football-match-dataset/2009-10.csv')
all_raw_data_11 = pd.read_csv('../input/football-match-dataset/2010-11.csv')
all_raw_data_12 = pd.read_csv('../input/football-match-dataset/2011-12.csv')
all_raw_data_13 = pd.read_csv('../input/football-match-dataset/2012-13.csv')
all_raw_data_14 = pd.read_csv('../input/football-match-dataset/2013-14.csv')
all_raw_data_15 = pd.read_csv('../input/football-match-dataset/2014-15.csv')
all_raw_data_16 = pd.read_csv('../input/football-match-dataset/2015-16.csv')
all_raw_data_17 = pd.read_csv('../input/footballdata2016onwards/2016-17.csv')
all_raw_data_18 = pd.read_csv('../input/footballdata2016onwards/2017-18.csv')
all_raw_data_19 = pd.read_csv('../input/footballdata2016onwards/2018-19.csv')


# Now, reading the columns I only want, all the seasons should have the same shape. 

# In[ ]:


raw_data_1 = all_raw_data_1[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_1.shape)
raw_data_2 = all_raw_data_2[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_2.shape)
raw_data_3 = all_raw_data_3[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_3.shape)
raw_data_4 = all_raw_data_4[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_4.shape)
raw_data_5 = all_raw_data_5[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_5.shape)
raw_data_6 = all_raw_data_6[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_6.shape)
raw_data_7 = all_raw_data_7[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_7.shape)
raw_data_8 = all_raw_data_8[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_8.shape)
raw_data_9 = all_raw_data_9[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_9.shape)
raw_data_10 = all_raw_data_10[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_10.shape)
raw_data_11 = all_raw_data_11[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_11.shape)
raw_data_12 = all_raw_data_12[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_12.shape)
raw_data_13 = all_raw_data_13[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_13.shape)
raw_data_14 = all_raw_data_14[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_14.shape)
raw_data_15 = all_raw_data_15[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_15.shape)
raw_data_16 = all_raw_data_16[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_16.shape)
raw_data_17 = all_raw_data_17[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_17.shape)
raw_data_18 = all_raw_data_18[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_18.shape)
raw_data_19 = all_raw_data_19[['HomeTeam','AwayTeam','FTHG','FTAG','FTR','HTHG','HTAG','HTR','HS','AS','HST','AST','HF','AF','HC','AC','HY','AY','HR','AR']]
print(raw_data_19.shape)


# It seems that season 2014-15 has one match more than it should have. Let's see what's happening here.

# In[ ]:


raw_data_15.tail()


# It seems it has an invalid last row, so lets just delete it.

# In[ ]:


raw_data_15 = raw_data_15.iloc[0:380,]
raw_data_15.shape


# Let's put all matches across all seasons in the same table. 

# In[ ]:


playing_stat = pd.concat([raw_data_1,raw_data_2,raw_data_3,raw_data_4,raw_data_5,raw_data_6,raw_data_7,raw_data_8,raw_data_9,raw_data_10,raw_data_11,raw_data_12,raw_data_13,raw_data_14,raw_data_15,raw_data_16, raw_data_17, raw_data_18, raw_data_19],ignore_index=True)
seasons = [raw_data_1,raw_data_2,raw_data_3,raw_data_4,raw_data_5,raw_data_6,raw_data_7,raw_data_8,raw_data_9,raw_data_10,raw_data_11,raw_data_12,raw_data_13,raw_data_14,raw_data_15,raw_data_16, raw_data_17, raw_data_18, raw_data_19]


# In[ ]:


playing_stat.head()


# In[ ]:


playing_stat.shape


# The columns mean the following:
# 
# FTHG - Full Time Home Goal
# 
# FTAG - Half Time Away Goal
# 
# FTR - Full Time Result
# 
# HTHG - Half Time Home Goal
# 
# HTAG - Half Time Away Goal
# 
# HTR - Half Time Result
# 
# HS - Home Shots
# 
# AS - Away Shots
# 
# HST - Home Shots on Target
# 
# AST - Away Shots on Target
# 
# HF - Home Team Foul
# 
# AF - Away Team Foul
# 
# HC - Home Team Corner
# 
# AC - Away Team Corner
# 
# HY - Home Team Yellow Card
# 
# Ay - Away Team Yellow Card
# 
# HR - Home Team Red Card
# 
# Ar - Away Team Red Card
# 

# # Feature Extraction

# Now, the table in itself is not very informative, but we can extract some features from it, relating to the offensive/defensive capabilities of the home/away team. 

# In[ ]:


table = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS'))

avg_home_scored = playing_stat.FTHG.sum() / 7220.0
avg_away_scored = playing_stat.FTAG.sum() / 7220.0
avg_home_conceded = avg_away_scored
avg_away_conceded = avg_home_scored
#print
res_home = playing_stat.groupby('HomeTeam')
#print()
res_away = playing_stat.groupby('AwayTeam')
all_teams_list = list(res_home.groups.keys())
print(all_teams_list)
table.Team = list(res_home.groups.keys())
table.HGS = res_home.FTHG.sum().values
table.HGC = res_home.FTAG.sum().values
table.AGS = res_away.FTAG.sum().values
table.AGC = res_away.FTHG.sum().values
#19 Home matches for each team each season and 16 seasons therefore 361 home matches and 361 away matches
table.HAS = (table.HGS / 361.0) / avg_home_scored
table.AAS = (table.AGS / 361.0) / avg_away_scored
table.HDS = (table.HGC / 361.0) / avg_home_conceded
table.ADS = (table.AGC / 361.0) / avg_away_conceded
#table


# In[ ]:


#Extract necessary features from the data file
feature_table = playing_stat.iloc[:,:23]

#Full Time Result(FTR), Home Shots on Target(HST), Away Shots on Target(AST), Home Corners(HC), Away Corners(AC)
feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST','HC','AC']]

#Home Attacking Strength(HAS), Home Defensive Strength(HDS), Away Attacking Strength(AAS), Away Defensive Strength(ADS)
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index,row in feature_table.iterrows():
    f_HAS.append(table[table['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table[table['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table[table['Team'] == row['AwayTeam']]['AAS'].values[0])
    f_ADS.append(table[table['Team'] == row['AwayTeam']]['ADS'].values[0])

feature_table['HAS'] = f_HAS
feature_table['HDS'] = f_HDS
feature_table['AAS'] = f_AAS
feature_table['ADS'] = f_ADS
feature_table


# Now we need to know who won each match, that will be our classification target (our $y$). Lets just transform a home win in a $1$, a home defeat in a $-1$ and a draw in a $0$. 

# In[ ]:


#Function to transform FTR into numeric data type
def transformResult(row):
    if(row.FTR == 'H'):
        return 1
    elif(row.FTR == 'A'):
        return -1
    else:
        return 0


# In[ ]:


feature_table["Result"] = feature_table.apply(lambda row: transformResult(row),axis=1)
y_pd = feature_table["Result"]
y = y_pd.values #Implementation Issue
feature_table.tail()


# But this features are not very good: we assume that a team has the same offensive/defensive capabilities across all seasons, and that is simply not true. Lets get the same features then but lets compute them per season, not across all seasons. 

# In[ ]:


def get_features_per_season(data_now):
    table_new = pd.DataFrame(columns=('Team','HGS','AGS','HAS','AAS','HGC','AGC','HDS','ADS'))
    avg_home_scored = data_now.FTHG.sum() / 380.0
    avg_away_scored = data_now.FTAG.sum() / 380.0
    avg_home_conceded = avg_away_scored
    avg_away_conceded = avg_home_scored
    res_home = data_now.groupby('HomeTeam')
    res_away = data_now.groupby('AwayTeam')
    table_new.Team = list(res_home.groups.keys())
    table_new.HGS = res_home.FTHG.sum().values
    table_new.HGC = res_home.FTAG.sum().values
    table_new.AGS = res_away.FTAG.sum().values
    table_new.AGC = res_away.FTHG.sum().values
    #19 Home matches for each team each season
    table_new.HAS = (table_new.HGS / 19.0) / avg_home_scored
    table_new.AAS = (table_new.AGS / 19.0) / avg_away_scored
    table_new.HDS = (table_new.HGC / 19.0) / avg_home_conceded
    table_new.ADS = (table_new.AGC / 19.0) / avg_away_conceded
    return table_new

get_features_per_season(raw_data_15)


# In[ ]:


map_features = map(get_features_per_season, seasons) 
list_tables = list(map_features)


# In[ ]:


import math

feature_table = playing_stat.iloc[:,:23]
feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST','HC','AC']]

def construct_X_per_match(table, match):
        HAS = table[table['Team'] == match['HomeTeam']]['HAS'].values[0]
        HDS = table[table['Team'] == match['HomeTeam']]['HDS'].values[0]
        AAS = table[table['Team'] == match['AwayTeam']]['AAS'].values[0]
        ADS = table[table['Team'] == match['AwayTeam']]['AAS'].values[0]
        return HAS, HDS, AAS, ADS


# In[ ]:


#Full Time Result(FTR), Home Shots on Target(HST), Away Shots on Target(AST), Home Corners(HC), Away Corners(AC)
feature_table = feature_table[['HomeTeam','AwayTeam','FTR','HST','AST','HC','AC']]

#Home Attacking Strength(HAS), Home Defensive Strength(HDS), Away Attacking Strength(AAS), Away Defensive Strength(ADS)
f_HAS = []
f_HDS = []
f_AAS = []
f_ADS = []
for index,row in feature_table.iterrows():
    table_ix = math.floor(index/380)
    table = list_tables[table_ix]
    f_HAS.append(table[table['Team'] == row['HomeTeam']]['HAS'].values[0])
    f_HDS.append(table[table['Team'] == row['HomeTeam']]['HDS'].values[0])
    f_AAS.append(table[table['Team'] == row['AwayTeam']]['AAS'].values[0])
    f_ADS.append(table[table['Team'] == row['AwayTeam']]['ADS'].values[0])

feature_table['HAS'] = f_HAS
feature_table['HDS'] = f_HDS
feature_table['AAS'] = f_AAS
feature_table['ADS'] = f_ADS
feature_table


# But these features take into account to long of a time period (season long), and do not take into account the head to head results between both teams. I think that some match-ups favor some team over another, and I will be looking at results between individual teams. Form across all season is important, but I think taking into account indiviual match-ups is also important (for example, Sunderland used to beat Manchester City when they were in the premier league, altough Manchester was a much better team). 
# 
# The features we are going to implement are the following: Average goals of the home team in the past $n$ matches against the away team, average goals of the away team in the past $n$ matches against the home team and average points of the home team in the past $n$ matches against the away team (we do not include the average points of the away team in the past $n$ matches against the home team, this feature is 'contained' in the past one, since points scored are entriely determined by points scored by the rival team). But there is a problem: What if there are not $n$ matches between the teams prior to a given match? Lets split our problem in two cases:
# 
# Case 1: There are no prior matches between the two teams.
# 
# This happens frequently for two reasons: 1) teams are relegated/promoted each season, so maybe a team which has never been promoted can take on a match between a premier league staple team, and no past matches will be availabe; 2) the first matches of the season we include in this project (season 2000-01) will have no prior encounters between teams.
# 
# To solve this problem, we will therefore fill each of the features with some mean values: i.e., instead of using average goals of the home team in the past $n$ matches against the away team, we will use the average goals of home teams across all seasons. The same with average away goals and average home points.
# 
# Case 2: There are $m$ prior matches between the two teams but $m < n$.
# 
# In this case, we simply use the average across those $m$ matches. 

# In[ ]:


n_matches = len(playing_stat)
average_home_goals = sum(playing_stat['FTHG'])/n_matches
average_away_goals = sum(playing_stat['FTAG'])/n_matches
average_home_points = (3*sum(playing_stat['FTR'] == 'H') + sum(playing_stat['FTR'] == 'D'))/n_matches
average_away_points = (3*sum(playing_stat['FTR'] == 'A') + sum(playing_stat['FTR'] == 'D'))/n_matches
print(average_home_goals)
print(average_away_goals)
print(average_home_points)
print(average_away_points)


# In[ ]:


def get_features_match(match, n=5):
    team1 = match['HomeTeam']
    team2 = match['AwayTeam']
    # Constructing a table when all the matches between the two teams are shown.
    res = playing_stat[((playing_stat['HomeTeam']==team1) & (playing_stat['AwayTeam']==team2)) | ((playing_stat['AwayTeam']==team1) & (playing_stat['HomeTeam']==team2))]
    name = match.name
    idx = res.index.get_loc(name)
    if idx >= n: #If there is at least n matches between the teams
        matches = res.iloc[idx-n:idx]
        pts_home = 0
        goals_home = 0
        goals_away = 0
        for index, row in matches.iterrows():
            #This ifs control if the current home team was home team or away in the past matches between the team in order
            #to sum correctly home/away goals. 
            if row['HomeTeam'] == team1:
                goals_home += row['FTHG']
                goals_away += row['FTAG']
                if row['FTR'] == 'H':
                    pts_home += 3
                elif row['FTR'] == 'D':
                    pts_home += 1
            if row['AwayTeam'] == team1:
                goals_home += row['FTAG'] 
                goals_away += row['FTHG']
                if row['FTR'] == 'A':
                    pts_home += 3
                elif row['FTR'] == 'D':
                    pts_home += 1
        pts_avg = pts_home/n
        goals_home_avg = goals_home/n
        goals_away_avg = goals_away/n
    elif idx == 0: # If there is 0 matches between the teams
        pts_avg = 1.6450138504155125
        goals_home_avg = 1.5336565096952908
        goals_away_avg = 1.1425207756232687
    else: #Some games between the teams but not n
        matches = res.iloc[0:idx]
        m = len(matches)
        pts_home = 0
        goals_home = 0
        goals_away = 0
        for index, row in matches.iterrows():
            #This ifs control if the current home team was home team or away in the past matches between the team in order
            #to sum correctly home/away goals. 
            if row['HomeTeam'] == team1:
                goals_home += row['FTHG']
                goals_away += row['FTAG']
                if row['FTR'] == 'H':
                    pts_home += 3
                elif row['FTR'] == 'D':
                    pts_home += 1
            if row['AwayTeam'] == team1:
                goals_home += row['FTAG'] 
                goals_away += row['FTHG']
                if row['FTR'] == 'A':
                    pts_home += 3
                elif row['FTR'] == 'D':
                    pts_home += 1
        pts_avg = pts_home/m
        goals_home_avg = goals_home/m
        goals_away_avg = goals_away/m
    return pts_avg, goals_home_avg, goals_away_avg


# In[ ]:


pts_avgs = []
goals_home_avgs = []
goals_away_avgs = []
for index, row in playing_stat.iterrows():
    pts_avg, goals_home_avg, goals_away_avg = get_features_match(row, n=5)
    pts_avgs.append(pts_avg)
    goals_home_avgs.append(goals_home_avg)
    goals_away_avgs.append(goals_away_avg)
len(pts_avgs)


# In[ ]:


feature_table['FFPTSH'] = pts_avgs
feature_table['FFHG'] = goals_home_avgs
feature_table['FFAG'] = goals_away_avgs 
feature_table.tail()


# Now we're going to extract other features relating to form. Form is a term often used in football to refer to a team recent performance, i.e. good form is that they have been winning many matches lately. If teams have been getting good results as of late, there are many chances that the results will be good in the match that we are considering.  The features will be related to home team form and away team form: average points and goals of home team in the last $n$ matches and average points and goals of away team in last $n$ matches. We will handle missing/incomplete data in the same way we handled in the past features.
# 
# I will use a different function to get streak at home and to get streak away: they are very similar, there was a minor implementation issue so I had to separate them in order for them to work okay. Maybe they can be merged into one function but with a lot of ifs in order to control for which teams have enough matches and will be very tedious to program, I prefer it this way. 

# In[ ]:


def get_features_streak_home(match, n=10):
    team1 = match['HomeTeam']
    team1_stats = playing_stat[((playing_stat['HomeTeam']==team1) | (playing_stat['AwayTeam']==team1))]
    name = match.name
    idx = team1_stats.index.get_loc(name)
    if idx == 0:
        pts_avg = 1.6450138504155125
        goals_scored_avg = 1.5336565096952908
        goals_conceded_avg = 1.1425207756232687
    else:
        if idx-n < 0:
            newidx = 0
        else:
            newidx = idx - n
        matches_team1 = team1_stats.iloc[newidx:idx]
        m = len(matches_team1)
        pts = 0
        goals_scored = 0
        goals_conceded = 0
        for index, row in matches_team1.iterrows():
            if row['HomeTeam'] == team1:
                goals_scored += row['FTHG']
                goals_conceded += row['FTAG']
                if row['FTR'] == 'H':
                    pts += 3
                elif row['FTR'] == 'D':
                    pts += 1
            if row['AwayTeam'] == team1:
                goals_scored += row['FTAG'] 
                goals_conceded += row['FTHG']
                if row['FTR'] == 'A':
                    pts += 3
                elif row['FTR'] == 'D':
                    pts += 1
        pts_avg = pts/m
        goals_scored_avg = goals_scored/m
        goals_conceded_avg = goals_conceded/m
    return pts_avg, goals_scored_avg, goals_conceded_avg


# In[ ]:


#Seeing if it works for a given match
match = playing_stat.iloc[1234,:]
get_features_streak_home(match, n=10)


# In[ ]:


pts_streak_home = []
goals_scored_streak_home = []
goals_conceded_streak_home = []
for index, row in playing_stat.iterrows():
    pt_streak_home, goal_scored_streak_home, goal_conceded_streak_home = get_features_streak_home(row, n=15)
    pts_streak_home.append(pt_streak_home)
    goals_scored_streak_home.append(goal_scored_streak_home)
    goals_conceded_streak_home.append(goal_conceded_streak_home)
len(pts_streak_home)


# In[ ]:


feature_table['PSH'] = pts_streak_home
feature_table['SSH'] = goals_home_avgs
feature_table['CSH'] = goals_away_avgs 
feature_table.tail()


# The head of the table (first matches) should have the average placeholder values, see that it worked.

# In[ ]:


feature_table.head()


# In[ ]:


def get_features_streak_away(match, n=10):
    team1 = match['AwayTeam']
    team1_stats = playing_stat[((playing_stat['HomeTeam']==team1) | (playing_stat['AwayTeam']==team1))]
    name = match.name
    idx = team1_stats.index.get_loc(name)
    if idx == 0:
        pts_avg = 1.1023545706371192
        goals_scored_avg = 1.1425207756232687
        goals_conceded_avg = 1.5336565096952908
    else:
        if idx-n < 0:
            newidx = 0
        else:
            newidx = idx - n
        matches_team1 = team1_stats.iloc[newidx:idx]
        m = len(matches_team1)
        pts = 0
        goals_scored = 0
        goals_conceded = 0
        for index, row in matches_team1.iterrows():
            if row['HomeTeam'] == team1:
                goals_scored += row['FTHG']
                goals_conceded += row['FTAG']
                if row['FTR'] == 'H':
                    pts += 3
                elif row['FTR'] == 'D':
                    pts += 1
            if row['AwayTeam'] == team1:
                goals_scored += row['FTAG'] 
                goals_conceded += row['FTHG']
                if row['FTR'] == 'A':
                    pts += 3
                elif row['FTR'] == 'D':
                    pts += 1
        pts_avg = pts/m
        goals_scored_avg = goals_scored/m
        goals_conceded_avg = goals_conceded/m
    return pts_avg, goals_scored_avg, goals_conceded_avg


# In[ ]:


pts_streak_away = []
goals_scored_streak_away = []
goals_conceded_streak_away = []
for index, row in playing_stat.iterrows():
    pt_streak_away, goal_scored_streak_away, goal_conceded_streak_away = get_features_streak_away(row, n=15)
    pts_streak_away.append(pt_streak_away)
    goals_scored_streak_away.append(goal_scored_streak_away)
    goals_conceded_streak_away.append(goal_conceded_streak_away)
len(pts_streak_home)


# In[ ]:


feature_table['PSA'] = pts_streak_away
feature_table['SSA'] = goals_scored_streak_away
feature_table['CSA'] = goals_conceded_streak_away
feature_table.tail()


# # Model Selection and Betting Strategy
# 
# Now we have to select our classification model and its hyperparameters. Also, we need to select our betting strategy (naive betting strategy is just betting all on the classifier output). Basing our performance in classification score does not make much sense, and I will base our performance on something more akin to the problem: Percentage gains on the initial capital. Lets say we will bet on all matches across 2018-19 season the same amount. Lets call $Capital_{start}$ our starting capital. Every time we place a bet, our capital decreases by $1$ and if we do not classify correctly, this money will be gone. If we classify correctly, our capital will increase by the corresponding betting odd (for example lets say that a betting odd for the home team winning is $1.1$. We lose $1$ placing the bet, but if we classify correctly, we get back $1.1$). Then, a good metric for performance will be percentual gains, that is $\frac{Capital_{start} - Capital_{end}}{Capital_{start}}*100$.
# 
# Another thing to take into account is that it may not be always advisable to bet on a team that our classifier tells us is going to win (because of bad odds, for example). I will consider other betting strategies also in this work. 
# 
# I want to fine-tune the model, and for that we will need validation data in order to have generalizable results. This will be season 2017-18. Our process is resumed like this:
# 
# 1) Train using data from seasons 2000-01 through 2016-17.
# 
# 2) Test a bunch of classifying models with a bunch of betting strategies, using 2017-18 season data as testing data.
# 
# 3) Select the best model in terms of percentual gains in captital.
# 
# 4) Retrain the model selected used in numeral 3 but this time using 2000-01 through 2017-18 seasons.
# 
# 5) See how well the model does for the 2018-19 season data.
# 

# First, before fine tuning, I will show an example of how I measure performance. Lets try a kNN classifier with $45$ neighbors. Our first betting strategy will bet on the best odd across houses we can get for the predicted output of the classifier. 

# In[ ]:


raw_data_val = pd.read_csv('../input/footballdata2016onwards/2017-18.csv')
betting_odds_val = raw_data_val[['B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD', 'IWA', 'PSH', 'PSD','PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']]
betting_odds_val


# In[ ]:


#Choosing the data for training and validation.

X = feature_table.iloc[:,7:]
X_train_val = X[0:6460]
X_val = X[6460:6840]
y_train_val = y[0:6460]
y_test_val = y[6460:6840]
print(X_train_val.shape)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

def performance(betting_odds, y_t, predicted):
    m, n = betting_odds.shape
    wins = [i % 3 == 0 for i in range(n)]
    draw = [i % 3 == 1 for i in range(n)]
    lost = [i % 3 == 2 for i in range(n)]
    capital = m
    capital_start = capital
    for i, row in enumerate(betting_odds.values):
        pred = predicted[i]
        if pred == 1:
            rownew = row[wins]
            bestodd = max(rownew)
        elif pred == 0:
            rownew = row[draw]
            bestodd = max(rownew)
        else:
            rownew = row[lost]
            bestodd = max(rownew)
        capital -= 1
        if pred == y_t[i]:
            capital += bestodd

    #print('Capital Starting:')
    #print(capital_start)
    #print('Capital at end:')
    #print(capital)
    #print('Comparative winnings (percentage):')
    #print((capital-capital_start)*100/capital_start)
    return (capital-capital_start)*100/capital_start

print('Betting randomly')
unifs = np.random.randint(low=-1.0, high=2, size=(380,))
p = performance(betting_odds_val, y_test_val, unifs)
print(p)
clfknn = KNeighborsClassifier(n_neighbors = 15)
clfknn.fit(X_train_val, y_train_val)
y_pred_knn = clfknn.predict(X_val)
print('Betting using kNN with 45 neighbors')
p = performance(betting_odds_val, y_test_val, y_pred_knn)
print(p)


# You can see my initial model was not very good (I actually end up loosing money), that is why we would want to use fine-tuning in order to get the best model we can get, with different betting strategies. First, lets define the other betting strategies.

# New betting strategy: Pick the best odds for each of the outcomes across all betting houses. Lets call this odds $o_w$, $o_d$, $o_l$, corresponding to the odds for home winning, drawing and loosing. Also, lets call the predicted probabilities for each of our classes $p_w, p_d$ and $p_l$. Now, for each bet, lets consider the objective function $f = \alpha p_w o_w + \beta p_d o_d + \gamma p_l o_l$. Our best betting strategy is therefore betting $\alpha, \beta, \gamma$ such that they maximize $f$ and subject to $\alpha + \beta + \gamma = 1$.

# In[ ]:


from scipy.optimize import linprog

def strategize(betting_odds, clf, X_test):
    m, n = betting_odds.shape
    A_eq = np.array([[1, 1, 1]])
    b_eq = [1]
    xs = []
    odds_new = []
    funs = []
    for i in range(m):
        predicted_probs = clf.predict_proba(X_test)
        prob = predicted_probs[i]
        test = betting_odds.iloc[i,:]
        wins = [i % 3 == 0 for i in range(n)]
        draw = [i % 3 == 1 for i in range(n)]
        lost = [i % 3 == 2 for i in range(n)]
        o_w = max(test[wins])
        o_d = max(test[draw])
        o_l = max(test[lost])
        odds_new.append([o_w, o_d, o_l])
        c = np.array([-o_w*prob[0], -o_d*prob[1], -o_l*prob[2]])
        res  = linprog(c, A_eq, b_eq, bounds=(0, 1))
        xs.append(res.x)
        fo = -res.fun
        funs.append(fo)
    return np.asarray(xs), np.asarray(odds_new), funs


# This new betting strategy has an assumption: the classifiers are good at predicting each class probability. However, this is not generally true: classifiers are usually designed with classifiying power in mind, predicting probabilities is not always taken into account (see https://stats.stackexchange.com/questions/76693/machine-learning-to-predict-class-probabilities for a discussion). However, all classifiers can be calibrated using isotonic regression so it gives better probability predictions. Scikitlearn has a package to do exactly this.

# In[ ]:


from sklearn.calibration import CalibratedClassifierCV

clfknn = KNeighborsClassifier(n_neighbors = 15)
cccv = CalibratedClassifierCV(clfknn , method='isotonic', cv=5)
cccv.fit(X_train_val, y_train_val)
xs, odds, funs = strategize(betting_odds_val, cccv, X_val)
print(xs)


# The strategy ALWAYS select betting all into a single outcome (believe me, i've tried with many classifiers). Now lets see how we would perform using this classifier. Therefore, we will need to turn these strategies into actual 'predictions' (i.e. if the first row tells you to bet all on away team I need to convert that into a -1, so we have a vector instead of a matrix). Also, since I extracted best odds into a separate matrix, I will need to construct a new performance function with those odds in mind. 

# In[ ]:


def bet_strat_to_predictions(bet_strat):
    bets = []
    m, n = bet_strat.shape
    for strat in bet_strat:
        if strat[0] == 1:
            bets.append(1)
        elif strat[1] == 1:
            bets.append(0)
        elif strat[2] == 1:
            bets.append(-1)
    return bets

def performance_new(odds, bet_strat, y_t):
    m, n = odds.shape
    capital = m
    capital_start = capital
    for i, row in enumerate(odds):
        strat = bet_strat[i]
        capital -= 1
        if strat == 1:
            odd = row[0]
        elif strat == 0:
            odd = row[1]
        elif strat == -1:
            odd = row[2]
        if strat == y_t[i]:
            capital += odd

    #print('Capital Starting:')
    #print(capital_start)
    #print('Capital at end:')
    #print(capital)
    #print('Comparative winnings (percentage):')
    #print((capital-capital_start)*100/capital_start)
    return (capital-capital_start)*100/capital_start

xs, odds, fun = strategize(betting_odds_val, cccv, X_val)
bets = bet_strat_to_predictions(np.asarray(xs))
p = performance_new(odds, bets, y_test_val)
print(p)


# Still, quite poor performance. But let us not panic yet: we have not fine tuned. Finally, lets try one last betting strategy before fine tuning: What if we don't bet every time? What if some matches are deemed not good enough to bet on? I implement this using the value of objective function of the linear programming problem I proposed above. The strategy is like this: If we the objective function is above a certain theshold, then I will bet, if it is not, I will not bet. How to choose the treshold? This is not trivial. Lets see how much is the mean and the standard deviation of the objective function (I actually saved the values in the variable fun).

# In[ ]:


print(np.mean(fun))
print(np.std(fun))


# The standard deviation is high compared to the mean (almost the same magnitude). Lets choose $3$ different thresholds and bet according to these: The first one is the median. The other two are the $25\%$ and $75\%$ cuantiles. 

# In[ ]:


t1 = np.quantile(fun, 0.5)
t2 = np.quantile(fun, 0.25)
t3 = np.quantile(fun, 0.75)
print(t1)
print(t2)
print(t3)


# In[ ]:


def performance_new_2(odds, bet_strat, y_t, fun, threshold):
    m, n = odds.shape
    capital = m
    capital_start = capital
    for i, row in enumerate(odds):
        strat = bet_strat[i]
        if fun[i] >= threshold:
            capital -= 1
            if strat == 1:
                odd = row[0]
            elif strat == 0:
                odd = row[1]
            elif strat == -1:
                odd = row[2]
            if strat == y_t[i]:
                capital += odd

    #print('Capital Starting:')
    #print(capital_start)
    #print('Capital at end:')
    #print(capital)
    #print('Comparative winnings (percentage):')
    #print((capital-capital_start)*100/capital_start)
    return (capital-capital_start)*100/capital_start


# In[ ]:


print('First Threshold (median)')
p = performance_new_2(odds, bets, y_test_val, fun, t1)
print(p)
print('Second threshold (25%)')
print(p)
p = performance_new_2(odds, bets, y_test_val, fun, t2)
print('Third Threshold (75%)')
print(p)
p = performance_new_2(odds, bets, y_test_val, fun, t3)


# Now, finally, fine tuning. We will use only one type of model: kNN classifiers (later I will fine tune other models, don't worry). So, we need to fine tune on the following parameters: Number of neighbors, lets call this $k$. Number of matches in the face to face matchups to take into account, lets call it $n$. Also, number of matches to take into account for form. Lets call this $m$. For each combination of these parameters we will have to construct a model... It will be quite costly computationally. For $k$ lets consider $k = [ 5, 11, \dots, 101]$. For $n$ lets consider $n = [1, 2, 3, 4, 5]$. For m lets consider $m = [5, 10, 15, 20]$. Then our search space of parameters is $20 \times 5 \times 4=400$. This is quite big (we would have to train $400$ models plus evaluation with the $5$ different betting strategies. Also, for each combination of $n$ and $m$ we have to construct a brand new feature table, which is also costly computationally. I propose something a little better, albeit a little less robust: just choose parameters randomly $50$ times and compute the model performance for those parameters across all betting strategies. Then the parameters that score better will be the ones that we choose.

# This is actually to much computation time for kaggle to handle. Running the search offline, we get the following hyperparameters (with the performance): Startegy = 1, k = 31, n = 5, m = 20. 

# In[ ]:


pts_avgs = []
goals_home_avgs = []
goals_away_avgs = []
pts_streak_home = []
goals_scored_streak_home = []
goals_conceded_streak_home = []
pts_streak_away = []
goals_scored_streak_away = []
goals_conceded_streak_away = []
print('Feature extraction: begin')
for index, row in playing_stat.iterrows():
    pts_avg, goals_home_avg, goals_away_avg = get_features_match(row, n=5)
    pts_avgs.append(pts_avg)
    goals_home_avgs.append(goals_home_avg)
    goals_away_avgs.append(goals_away_avg)
    pt_streak_home, goal_scored_streak_home, goal_conceded_streak_home = get_features_streak_home(row, n=20)
    pts_streak_home.append(pt_streak_home)
    goals_scored_streak_home.append(goal_scored_streak_home)
    goals_conceded_streak_home.append(goal_conceded_streak_home)
    pt_streak_away, goal_scored_streak_away, goal_conceded_streak_away = get_features_streak_away(row, n=20)
    pts_streak_away.append(pt_streak_away)
    goals_scored_streak_away.append(goal_scored_streak_away)
    goals_conceded_streak_away.append(goal_conceded_streak_away)
print('Feature Extraction: done')
feature_table['FFPTSH'] = pts_avgs
feature_table['FFHG'] = goals_home_avgs
feature_table['FFAG'] = goals_away_avgs
feature_table['PSH'] = pts_streak_home
feature_table['SSH'] = goals_home_avgs
feature_table['CSH'] = goals_away_avgs
feature_table['PSA'] = pts_streak_away
feature_table['SSA'] = goals_scored_streak_away
feature_table['CSA'] = goals_conceded_streak_away
X = feature_table.iloc[:,7:]
X_train_val = X[0:6460]
X_val = X[6460:6840]
y_train_val = y[0:6460]
y_test_val = y[6460:6840]


# In[ ]:


raw_data_test = pd.read_csv('../input/footballdata2016onwards/2018-19.csv')
betting_odds = raw_data_test[['B365H','B365D','B365A','BWH','BWD','BWA','IWH','IWD', 'IWA', 'PSH', 'PSD','PSA', 'WHH', 'WHD', 'WHA', 'VCH', 'VCD', 'VCA']]
betting_odds


# In[ ]:


X = feature_table.iloc[:,7:]
X_train = X[0:-380]
X_test = X[-380:]
y_train = y[0:-380]
y_test = y[-380:]
clfknn = KNeighborsClassifier(n_neighbors = 31)
clfknn.fit(X_train, y_train)
y_pred_knn = clfknn.predict(X_train)
p1 = performance(betting_odds, y_test, y_pred_knn)
print(p1)


# Our performance was then $6.88 \%$ increase in the initial bet.

# In[ ]:





# I also tried using SVM, and got the following performance (I searched the best C parameter offline also), because of computing times.

# In[ ]:


from sklearn.svm import SVC
clfsvm = SVC(C=0.001, gamma='scale')
clfsvm.fit(X_train, y_train)
y_pred_svm = clfsvm.predict(X_test)
p1 = performance(betting_odds, y_test, y_pred_knn)
print(p1)


# # Theoretical Bounds
# 
# For the kNN algorithm, lets call $L_{kNN}$ the loss function of the classifier, and $L^* = \lim_{n \to \infty} E[L_n]$. We know (see Theorem 5.6 in DeVroye's book), that $L_{kNN} - L^* \leq \frac{1}{\sqrt{ke}}$, meaning that the difference between our classifier and the best empirical classifier is at must

# In[ ]:


print(1/np.sqrt(30*np.exp(1)))


# # Dimensionality reduction
# 
# I will not be performing dimentionality reduction: the whole point of the work was to try and construct features for a classification problem, since those features were not explicitly there and I had to create certain functions in order to 'extract' them from the original data. 

# # Difficulties with defining a parametric model
# 
# In this problem we have $13$ features: the first $4$, which are season long features for each team. The next $3$ that relate to face-to-face results between the teams, and the last $6$ which relate to the form of both teams.
# 
# Note that these features have dependence on each other: form is, of course, related to season long performance, for example. Or goals scored and points scored are also dependent on each other. So, in order to use the features we chose, we would have to model a $13$-dimensional distribution. That would not be ideal, of course, since we could not visualize if our estimated $13$ dimensional density fits the data correctly.
# 
# So, if I wanted to do better parametric models I would have to reduce dimension, and that is not desirable either (beats the purpose of the exervise, see Dimensionality Reduction subsection): That is why it is better to just use nonparametric classification techniques and do not worry to much about the distribution of the data. 

# # ROC Curve
# 
# Lastly, we will implement ROC curves for this problem. ROC Curves were originally invented for binary classification, but we can still use ROC curves for multiclass problems with some modifications, and we will obtain a ROC Curve for each of the classes, and with that we can obtain a 'mean' ROC curve.

# In[ ]:


from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC

from sklearn.svm import SVC
clfsvm = SVC(C=0.001, gamma='scale')
clfsvm.fit(X_train, y_train)
y_pred_svm = clfsvm.predict(X_test)

y_binary = label_binarize(y, classes=[-1, 0, 1])
y_binary_train = y_binary[0:-380]
y_binary_test = y_binary[-380:]
n_classes = y_binary.shape[1]
classifier = OneVsRestClassifier(SVC(probability=True, gamma='scale', C=0.001))
y_score = classifier.fit(X_train, y_binary_train).decision_function(X_test)


# In[ ]:


from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_binary_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_binary_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])


# In[ ]:


from scipy import interp
import matplotlib.pyplot as plt
from itertools import cycle

# Compute macro-average ROC curve and ROC area

lw = 2

# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(-(i-1), roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Some extension of Receiver operating characteristic to multi-class')
plt.legend(loc="lower right")
plt.show()


# 

# # Conclusions
# 
# We can see in the ROC curves that the more difficult class to classify was 'Draw'. We achieves significant classifying score in each class, but, despite of that, we couldn't get a truly great augment in the initial bet (our best was an increase of $7.155263157894765 \%$. This can be because of various reasons:
# 
# -Betting houses are too good at estimating probabilities, very hard to beat them. We didn't use the numbers they have as features, because the point of this exercise was to see that if by selecting some features one might think are important, one might estimate good probabilities for matches outcomes.
# 
# -Our features are not enough: We may want to extract more features from the data. For examples, teams are differently motivated for different matches and that may change match outcome, for example, if a team can become champions by winning the next game then they probably will give $100\%$ in that game and win. A feature might be constructed then that relates to motivation.
# 
# -Our betting strategies were poor: The strategy we ended up choosing was the naive one: just bet on what the classifier tells you to. This might not be the best strategy, maybe some others could be proposed, which uses the probability of each class to try and make a better bet. 

# In[ ]:




