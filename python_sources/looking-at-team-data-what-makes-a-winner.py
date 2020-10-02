#!/usr/bin/env python
# coding: utf-8

# First time submitting to Kaggle so feedback appreciated. 
# 
# My analysis will focus on 
# 
#  1. Creating a season total, by team, data frame
#  2. Exploring this team data - what does it look like? is it complete?
#  3. Plotting team data
#  4. Use linear regression to see how certain data correlates to wins
# 
# Thanks to @epattaro for his submission, his code helped shape sections of this submission

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn import linear_model


# In[ ]:


#load data & check row/column size
data = pd.read_csv('../input/shot_logs.csv', header=0)

data.head()


# In[ ]:


#split the matchup into date, team & opponenet
def matchup_split(x):
    (a,b) = x.split('-')
    a = a.strip()
    return(a)

def team_split(x):
    (a,b) = x.split('-')
    if '@' in b:
        (b1, b2) = b.split('@')
    if 'vs.' in b:
        (b1, b2) = b.split('vs.')
    b1 = b1.strip()
    
    return(b1)
    
def opponent_split(x):
    (a,b) = x.split('-')
    if '@' in b:
        (b2, b1) = b.split('@')
    if 'vs.' in b:
        (b2, b1) = b.split('vs.')
    b1 = b1.strip()
    return(b1)

data['DATE'] = data['MATCHUP'].apply(matchup_split)
data['DATE'] = data['DATE'].apply(pd.to_datetime)
data['TEAM'] = data['MATCHUP'].apply(team_split)
data['OPPONENT'] = data['MATCHUP'].apply(opponent_split)

#check to make sure it worked
data[['DATE','TEAM','OPPONENT']].head()


# Now let's manipulate this data into team season totals

# In[ ]:


#first step: roll-up select data to team level; # of shots, shots made, & pts for team/opponent
df1 = data.groupby(['TEAM','GAME_ID','player_name'])[['FINAL_MARGIN','SHOT_NUMBER']].max()
df1[['FGM','PTS']] = data.groupby(['TEAM','GAME_ID','player_name'])[['FGM','PTS']].sum()
df1 = df1.reset_index(['TEAM','GAME_ID','player_name'])
df2 = df1.groupby('TEAM')[['SHOT_NUMBER','FGM','PTS']].sum()

df1 = data.groupby(['OPPONENT','GAME_ID','player_name'])[['FINAL_MARGIN','SHOT_NUMBER']].max()
df1[['FGM','PTS']] = data.groupby(['OPPONENT','GAME_ID','player_name'])[['FGM','PTS']].sum()
df1 = df1.reset_index(['OPPONENT','GAME_ID','player_name'])
df2[['OP_SHOT_NUMBER','OP_FGM','OP_PTS']] = df1.groupby('OPPONENT')[['SHOT_NUMBER','FGM','PTS']].sum()

df2['FINAL_MARGIN'] = df2.PTS - df2.OP_PTS

df2.head()


# In[ ]:


#next up let's add games played & games won
team_list = data['TEAM'].unique().tolist()
games_played = pd.DataFrame({'GAMES_PLAYED':0}, index=team_list)

temp_data = data.groupby(['TEAM','GAME_ID'])['FINAL_MARGIN'].sum()
temp_data = temp_data.reset_index(['TEAM','GAME_ID'])
temp_data = temp_data[temp_data['FINAL_MARGIN'] > 0]

gp = []
for i in team_list:
   gp.append(len(data[data.TEAM == i]['GAME_ID'].unique()))
   
gw = []
for i in team_list:
   gw.append(len(temp_data[temp_data.TEAM == i]))

gp = pd.DataFrame(gp, index=team_list)
gw = pd.DataFrame(gw, index=team_list)

df2['GAMES_PLAYED'] = gp
df2['GAMES_WON'] = gw


# In[ ]:


#next steps are to do some calculations
df2['SHOTS_PER_GAME'] = df2.SHOT_NUMBER / df2.GAMES_PLAYED
df2['OP_SHOTS_PER_GAME'] = df2.OP_SHOT_NUMBER / df2.GAMES_PLAYED
df2['DIFF_SHOTS_PER_GAME'] = df2.SHOTS_PER_GAME - df2.OP_SHOTS_PER_GAME
df2['PTS_PER_SHOT'] = df2.PTS / df2.SHOT_NUMBER
df2['OP_PTS_PER_SHOT'] = df2.OP_PTS / df2.OP_SHOT_NUMBER
df2['DIFF_PTS_PER_SHOT'] = df2.PTS_PER_SHOT - df2.OP_PTS_PER_SHOT
df2['PTS_PER_GAME'] = df2.PTS / df2.GAMES_PLAYED
df2['OP_PTS_PER_GAME'] = df2.OP_PTS / df2.GAMES_PLAYED

#For the upcoming steps, I only want per game data & games won
df3 = pd.DataFrame(df2[['SHOTS_PER_GAME','OP_SHOTS_PER_GAME','PTS_PER_SHOT','OP_PTS_PER_SHOT','DIFF_PTS_PER_SHOT',
                        'PTS_PER_GAME','OP_PTS_PER_GAME','DIFF_SHOTS_PER_GAME','GAMES_WON']])
df3.head()


# Okay, now we have our team data. Quick overview of what's going on
# 
# Appears we have incomplete data
# 
#  - In a standard NBA season, each team plays 82 games. Assuming I didn't make mistakes (big assumption), this data set is incomplete given the GAMES_PLAYED column is ~60 per team.
#  - It also looks like we don't have complete data by game either. Typically NBA teams score closer to 100 pts/game, not ~75
# 
# Why use points per shot (PTS_PER_SHOT), instead of shots made/shots taken (FG%)?
# 
#  - Points per Shot is a more complete metric to gauge a team's (or player's) efficiency as it accounts for 2-pt or 3-pt shots. A general rule is a points per shot value over 1 is efficient - a value of 1 is equal to either 50% FG% on 2-pt shots or 33.3% FG% on 3-pt shots.
# 
# Let's see what we can discover about wins given the rest of the data.

# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2,sharex=True,figsize=(8,5))
df3.plot('PTS_PER_SHOT', 'GAMES_WON', kind= 'scatter', ax=axis1)
df3.plot('OP_PTS_PER_SHOT', 'GAMES_WON', kind= 'scatter', ax=axis2)


# In[ ]:


fig, (axis1,axis2) = plt.subplots(1,2, sharex=True,figsize=(8,5))
df3.plot('SHOTS_PER_GAME', 'GAMES_WON', kind= 'scatter', ax=axis1)
df3.plot('OP_SHOTS_PER_GAME', 'GAMES_WON', kind= 'scatter', ax=axis2)


# In[ ]:


df3.plot('DIFF_PTS_PER_SHOT', 'GAMES_WON', kind= 'scatter')
df3.plot('DIFF_SHOTS_PER_GAME', 'GAMES_WON', kind= 'scatter')


# While not the only path forward, I want to how well 'DIFF_PTS_PER_SHOT' & 'DIFF_SHOTS_PER_GAME' indicate how many games a team has won.
# 
# Intuitively, it would make sense that being more efficient and shooting more would lead to more wins but there are more factors at play in a game (turnovers, rebounds, free throws, etc).
# 
# Let's see if this intuition is valid

# In[ ]:


X = pd.DataFrame({'x1': df3['DIFF_PTS_PER_SHOT'], 'x2': df2['DIFF_SHOTS_PER_GAME']}).values
y = pd.DataFrame({'y': df2['GAMES_WON']}).values

degrees = 4

lr = linear_model.LinearRegression()
poly = PolynomialFeatures(degree=degrees)
X_poly = poly.fit_transform(X)


model = Pipeline([('poly', PolynomialFeatures(degree=degrees)),
                  ('linear',linear_model.LinearRegression(fit_intercept=False))])

model = model.fit(X,y)
model.named_steps['linear'].coef_

prediction = model.named_steps['linear'].predict(X_poly)

print('Correlation of: ', model.named_steps['linear'].score(X_poly,y))


# Without taking proper steps to check over-fitting (I plan on getting to that), the two 'DIFF' variables have a high correlation with wins.
# 
# Definitely didn't learn anything ground-breaking - teams that take more shots, more efficiently win more games. 
# 
# Will be working on
# 
#  - Looping through degrees and comparing the 'scores'
# 
#  - Plotting 3D (need to do some learning here) 
# 
# I should have done a train/cv/test on for diagnostic purposes. Once I get the polynomial degree looping logic & 3D plot added, I might do it right w/ the data split. Maybe give it a test on predicting game by game...
# 
# Like mentioned above - first time posting on Kaggle. Would love to hear your feedback.
# 
# 
