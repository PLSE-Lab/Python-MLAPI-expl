#!/usr/bin/env python
# coding: utf-8

# **Thanks for viewing my Kernel! If you like my work and find it useful, please leave an upvote! :)**
# 
# Player Unknown's BattleGrounds aka PUBG is an online multiplayer game where up to 100 players parachute onto an island where they scavange for weapons and equipment to kill others in a last man standing battle royale. Players can choose to enter the match solo, or with a small team of up to four people. In either case, the last person or team left alive wins the match. 
# 
# **Key insights:**
# * Highly correlated variables with winPlacePerc: walkDistance (0.8), killPlace (-0.7), boosts (0.6) and weaponsAcquired (0.6)
# * 57% of players didn't have a single kill. 88% of players have only less than 3 kills.
# * If a player gets 3 or more health boosting boosts and heals, the chances are high that he/she gets more than 0.5 as winPlacePerc
# * Less than 5 players in a team is ideal for an average winPlacePerc of more than 0.5

# In[ ]:


from IPython.display import Image
Image(filename="../input/pubg-overview/overview.jpg")


# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly import tools
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)

import os
print(os.listdir("../input/pubg-finish-placement-prediction"))


# In[ ]:


train = pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')
test = pd.read_csv('../input/pubg-finish-placement-prediction/test_V2.csv')
sub = pd.read_csv('../input/pubg-finish-placement-prediction/sample_submission_V2.csv')

print('Train data: \nRows: {}\nCols: {}'.format(train.shape[0],train.shape[1]))
print(train.columns)

print('\nTest data: \nRows: {}\nCols: {}'.format(test.shape[0],test.shape[1]))
print(test.columns)

print('\nSubmission data: \nRows: {}\nCols: {}'.format(sub.shape[0],sub.shape[1]))
print(sub.columns)


# There is only one missing value in the Train data. Target value is not available for that record. The match ID corresponding to that record doesn't have any other record. Hence, dropping that record. 

# In[ ]:


print("Missing values in Train data")
for x in train.columns:
    if train[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,train[x].isnull().values.ravel().sum()))

print("Missing values in Test data")
for x in test.columns:
    if test[x].isnull().values.ravel().sum() > 0:
        print('{} - {}'.format(x,test[x].isnull().values.ravel().sum()))
        
train.dropna(inplace=True)


# **Very High Negative correlation**
# - killPoints and rankPoints
# 
# **Very High Positive correlation**
# - winPoints and killPoints
# - maxPlace and numGroups
# - kills and damageDealt

# In[ ]:


f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax, cmap=sns.color_palette("RdBu", 20))

ax.set_xlabel('Numeric variables', size=14, color="#3498DB")
ax.set_ylabel('Numeric variables', size=14, color="#3498DB")
ax.set_title('[Heatmap] Correlation Matrix', size=18, color="#3498DB")

plt.show()


# **57% of players didn't have a single kill. 88% of players have only less than 3 kills.**

# In[ ]:


data = train[['kills']].copy()

data.loc[data['kills'] > data['kills'].quantile(0.99)] = 8
temp1 = data['kills'].value_counts(sort=False).reset_index()
temp2 = data['kills'].value_counts(sort=False, normalize=True).reset_index()
temp2['kills'] = temp2['kills'] * 100

temp = pd.merge(temp1,temp2,how='inner',on='index')
temp['cum'] = temp['kills_y'].cumsum()
temp.loc[temp['index']==8, 'index'] = '8+'
temp['index'] = temp['index'].map(str) + " kills"

trace1 = dict(type='bar',
    x=temp['index'],
    y=temp['kills_x'],
    marker=dict(
        color='#2196F3'
    ),
    name='Number of kills',
    opacity=0.8
)

trace2 = dict(type='scatter',
    x=temp['index'],
    y=temp['cum'],
    marker=dict(
        color='#263238'
    ),
    line=dict(
        color= '#263238', 
        width= 1.5),
    name='Cumulative % of Players',
    xaxis='x1', 
    yaxis='y2' 
)

data = [trace1, trace2]

layout = go.Layout(
    title='[Pareto Analysis] Kills vs % of Players',
    legend= dict(orientation="h"),
    yaxis=dict(
        range=[0,2625000],
        title='Number of Kills',
        titlefont=dict(
            color="#2196F3"
        )
    ),
    yaxis2=dict(
        title='Cumulative % of Players',
        titlefont=dict(
            color='#263238'
        ),
        range=[0,105],
        overlaying='y',
        anchor='x',
        side='right'
        )
    )

fig = go.Figure(data=data, layout=layout)
iplot(fig, filename="pareto")


# **Average of winPlacePerc increases with the number of kills**

# In[ ]:


data = train[['kills','winPlacePerc']].copy()
data.loc[data['kills'] > data['kills'].quantile(0.99), 'kills'] = '8+'
data['kills'] = data['kills'].map(str) + " kills"
x_order = data.groupby('kills').mean().reset_index()['kills']

fig, ax = plt.subplots(figsize=(20,8))
a = sns.boxplot(x='kills', y='winPlacePerc', data=data, ax=ax, color="#2196F3", order=x_order)
ax.set_xlabel('Kills', size=14, color="#263238")
ax.set_ylabel('winPlacePerc', size=14, color="#263238")
ax.set_title('[Box Plot] Average winPlacePerc of Players with specific kills', size=18, color="#263238")
plt.show()


# **walkDistance has a 0.8 correlation with winPlacePerc. 99th percentile of walk Distance is 4396m. Grouping the walkDistance and creating a boxplot beautifully illustrates the correlation. **

# In[ ]:


kills = train[['walkDistance','winPlacePerc']].copy()

print("99th percentile of walk distance is {}m".format(kills['walkDistance'].quantile(0.99)))

cut_range = np.linspace(0,4400,23)
cut_range = np.append(cut_range, 26000)
kills['walkDistanceGrouping'] = pd.cut(kills['walkDistance'],
                                 cut_range,
                                 labels=["{}-{}".format(a_, b_) for a_, b_ in zip(cut_range[:23], cut_range[1:])],
                                 include_lowest=True
                                )

fig, ax = plt.subplots(figsize=(15,10))
sns.boxplot(x="winPlacePerc", y="walkDistanceGrouping", data=kills, ax=ax, color="#2196F3")
ax.set_xlabel('winPlacePerc', size=14, color="#263238")
ax.set_ylabel('walkDistance Groups (m)', size=14, color="#263238")
ax.set_title('[Horizontal Box Plot] Win Place Percentile vs Walk Distance', size=18, color="#263238")
plt.gca().xaxis.grid(True)
plt.show()


# **Abnormality observed in killPlace 30-40. Why is that so? What is a killPlace statistic exactly? Is it a sorted value of kills and winPlacePerc?**

# In[ ]:


kill_place = train[['killPlace','winPlacePerc']].copy()

cut_range = np.linspace(0,100,11)

kill_place['killPlaceGroups'] = pd.cut(kill_place['killPlace'],
                                 cut_range,
                                 labels=["{0:.0f}-{1:.0f}".format(a_, b_) for a_, b_ in zip(cut_range[:11], cut_range[1:])],
                                 include_lowest=True
                                )

fig, ax = plt.subplots(figsize=(15,6))
sns.boxplot(x="winPlacePerc", y="killPlaceGroups", data=kill_place, ax=ax, color="#2196F3")
ax.set_xlabel('winPlacePerc', size=14, color="#263238")
ax.set_ylabel('killPlace Groups', size=14, color="#263238")
ax.set_title('[Horizontal Box Plot] Win Place Percentile vs Kill Place', size=18, color="#263238")
plt.gca().xaxis.grid(True)
plt.show()


# **Weapons are key to win this game. More the weapons a player has, the higher is the winPlacePerc till 10 weapons after which there is a slight dip.**

# In[ ]:


weapons = train[['weaponsAcquired','winPlacePerc']].copy()
weapons.loc[weapons['weaponsAcquired'] > weapons['weaponsAcquired'].quantile(0.99), 'weaponsAcquired'] = '11+'
weapons['weaponsAcquired'] = weapons.weaponsAcquired.apply(lambda x: '0' + str(x) if isinstance(x,int) and x<10 else x) 
weapons['weaponsAcquired'] = weapons['weaponsAcquired'].map(str) + " weapons"
x_order = weapons.groupby('weaponsAcquired').mean().reset_index()['weaponsAcquired']

fig, ax = plt.subplots(figsize=(20,8))
a = sns.boxenplot(x='weaponsAcquired', y='winPlacePerc', data=weapons, ax=ax, color="#2196F3", order=x_order)
ax.set_xlabel('Weapons', size=14, color="#263238")
ax.set_ylabel('Mean winPlacePerc', size=14, color="#263238")
ax.set_title('[Box Plot] Average winPlacePerc vs weapons acquired', size=18, color="#263238")
plt.show()


# In[ ]:


# Generate Features
def generate_features(df):
    # All boosters
    df['boosters'] = df['heals'] + df['boosts']
    
    # All kills
    df['allKills'] = df['headshotKills']+df['kills']+df['roadKills']+df['teamKills']+df['assists']
    
    # All distance
    df['allDistance'] = df['rideDistance']+df['swimDistance']+df['walkDistance']
    
    # Players in team
    agg = df.groupby(['groupId']).size().to_frame('players_in_team')
    df = df.merge(agg, how='left', on=['groupId'])
    
    # Players in match
    agg = df.groupby(['matchId']).size().to_frame('players_in_match')
    df = df.merge(agg, how='left', on=['matchId'])
    
    return df

train = generate_features(train)
test = generate_features(test)


# **If a player gets 3 or more health boosting boosts and heals, the chances are high that he/she gets more than 0.5 as winPlacePerc**

# In[ ]:


data = train[['boosters','winPlacePerc']].copy()
data.loc[data['boosters'] > data['boosters'].quantile(0.99), 'boosters'] = '18+'
data['boosters'] = data.boosters.apply(lambda x: '0' + str(x) if isinstance(x,int) and x<10 else x) 
data['boosters'] = data['boosters'].map(str) + " boosters"
x_order = data.groupby('boosters').mean().reset_index()['boosters']

fig, ax = plt.subplots(figsize=(20,12))
a = sns.boxplot(x='winPlacePerc', y='boosters', data=data, ax=ax, color="#2196F3", order=x_order)
ax.set_ylabel('Boosters', size=14, color="#263238")
ax.set_xlabel('winPlacePerc', size=14, color="#263238")
ax.set_title('[Horizontal Box Plot] winPlacePerc of Players vs boosters', size=18, color="#263238")
plt.show()


# **Less than 5 players in a team is ideal for an average winPlacePerc of more than 0.5**

# In[ ]:


temp = train[['players_in_team','winPlacePerc']].copy()
temp = temp.groupby('players_in_team').mean().reset_index()

fig, ax = plt.subplots(figsize=(20,8))
a = sns.lineplot(x='players_in_team', y='winPlacePerc', data=temp, ax=ax, color="#2196F3")
ax.set_xlabel('Players in Team', size=14, color="#263238")
ax.set_ylabel('Mean winPlacePerc', size=14, color="#263238")
ax.set_title('[Line Plot] Average winPlacePerc vs players in team', size=18, color="#263238")
plt.show()


# **Thanks for reading my kernel! Upvote if you like it.**
