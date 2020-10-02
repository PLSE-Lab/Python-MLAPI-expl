#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# We import useful libraries, and fetch our csv file.

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

data = pd.read_csv('../input/fifa19/data.csv', encoding='utf-8')
data.shape


# In[ ]:


data.columns


# Wow that's whole lot of columns. Looks tasty. We will be first checking some of the missing values, if there's any.
# 

# In[ ]:


msno.matrix(data)


# As we see above, there are some NULL values in the dataset. However, null value does not indicate that it's an incomplete dataset, since there are columns like 'ST', which refers to a position of a player. It could be similar to one-hot encoding.

# So for now, let's drop only those without their names, overall stats and money value. Otherwise keep them for now.

# In[ ]:


data.dropna(subset=['Name', 'Overall', 'Value'], how='all', inplace=True)
data.shape


# Well the row number hasn't changed, so lucky for us, there is no un-usable rows in our dataset. Let's continue on!

# I want to see the overall picture of our data first. Let's look at their distributions on some columns.

# In[ ]:


def scatter_them(data, col1 = 'Age', col2='Overall'):
    plt.figure(figsize=(13,7))
    plt.title("{0} AND {1}".format(col1, col2))
    plt.scatter(data[col1], data[col2], color='dodgerblue', s=6)
    plt.xlabel(col1)
    plt.ylabel(col2)


# In[ ]:


df = data.copy()
scatter_them(df)


# Maybe there are too many data points, so scatterplot might not be the best option. Let's try box plot instead.

# In[ ]:


def box_them(data, col1='Age', col2='Overall', intervals=[15, 20, 25, 30, 35, 40, 50]):
    df = data.copy()
    df['for_purpose'] = pd.cut(df[col1], intervals, labels=[str(point) for point in intervals[1:]])
    plt.figure(figsize=(13,7))
    sns.boxplot(data=df, x='for_purpose', y=col2)
    plt.title("{0} AND {1}".format(col1, col2))


# In[ ]:


df = data.copy()
box_them(data=df)


# Now we see a clearer picture. The peak age for soccer player in terms of overall stat seems to be around 30.

# In[ ]:


box_them(data=df, col2='Potential')


# This one is a bit obvious. There is a column called potential, which indicates the probability of player growth. It clearly declines with time.

# I also take a look at player's value, but currently this column is in string format. So I will need a little modification before handling it.

# In[ ]:


df = data.copy()
df['Value']


# Alright, so it's pretty much human-specific notation. Let's check some units first.

# In[ ]:


def first_and_last(x):
    return (x[0], x[-1])

units = df['Value'].apply(first_and_last)
set(units)


# Okay. Now we know that all the values are in euros, and they end either in 0, K, or M.

# In[ ]:


def change_money(x):
    if x[-1] == '0':
        return int(x[1:])
    elif x[-1] == 'K':
        return 1000*float(x[1:-1])
    else:
        return 1000000*float(x[1:-1])


# In[ ]:


data['Value'] = data['Value'].apply(change_money)


# In[ ]:


data['Value'].min(), data['Value'].max()


# Now we have our values in place. Soccer fans might be already familiar, but let's check out some of the richest players of the world.

# In[ ]:


df = data.copy()

df.sort_values(by='Value', ascending=False)[['Name', 'Club', 'Nationality', 'Age', 'Value', 'Overall']].iloc[:10]


# For me personally, the fact that Christiano Ronaldo is no longer in the list was surprising. Messi shows incredible ability stats despite his age.

# In[ ]:


box_them(data=df, col1='Age', col2='Value')


# This is affected too much by the outliars. We'll have to do something to make this plot more meaningful.

# In[ ]:


by_value = df.sort_values(by='Value', ascending=False)

box_them(data=by_value[:1000], col1='Age', col2='Value')


# Better, but not yet. Let's drop out some of the highest paying outliars this time. They are simply unrealistic.

# In[ ]:


box_them(data=by_value[500:], col1='Age', col2='Value')


# I'd say it's pretty much enough for now. Again, aligned with Overall stat, Value is at peak around age 30.

# Lastly for age issue, let's check the overall tendency with line plots.

# In[ ]:


def line_them(data=df, col='Overall', groupby='Age'):
    x = df.groupby(groupby)[col].mean()
    plt.figure(figsize=(13,7))
    plt.title("{0} Over {1}".format(col, groupby))
    plt.plot(x, color='dodgerblue')
    plt.xlabel(groupby)
    plt.ylabel(col)


# In[ ]:


line_them()


# There appears to be an outliar around age 45. Let's see who is beating his time so well. 

# In[ ]:


df[df['Age']>=42].sort_values(by='Overall', ascending=False)


# A guy named O. Perez from Mexico, age 45, is doing pretty well considering his age. But since we have only a couple of players in these age groups, it will be better to leave them out for now.

# In[ ]:


df = data[data['Age']<42].copy()
line_them(data=df, col='Overall', groupby='Age')


# In[ ]:


line_them(col='Value', groupby='Age')


# Value shows a more dramatic tendency. Though our veterans showed relatively good performance, they couldn't get good valuations in the market.

# In[ ]:


line_them(col='Potential')


# Surely potential falls over time. That's what 'potential' means in the first place. At this point, I started to wonder how EAsports defined 'Potential' in game.

# In[ ]:


plt.figure(figsize=(13,7))
plt.plot(df.groupby('Age')['Overall'].mean(), color='red')
plt.plot(df.groupby('Age')['Potential'].mean(), color='darkblue')
plt.title("Overall and Potential stats over time")
plt.xlabel('Age')
plt.ylabel('Potential and Overall')


# In[ ]:


df.groupby('Age')['Overall'].mean() == df.groupby('Age')['Potential'].mean()


# So according to our game, potential only applies to players under or equal to age 30. Enough for this.

# I think it's enough about the age. I want to see how the stats are defined and allocated.

# In[ ]:


stats = df[['Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle']]

f,ax = plt.subplots(figsize=(15, 15))
sns.heatmap(stats.corr(), annot=True, linewidths=.5, fmt= '.1f',ax=ax)
plt.show()


# With this heatmap, I can see how some stats are correlated. But it's not easy to see since there are so many of them.
# Instead, I will look at which stat takes the most pivotal role in playing soccer. By this, I mean which is the stat 'hub' that affects the most overall ability.

# In[ ]:


stats.corr().sum().sort_values(ascending=False)


# In[ ]:


top5 = stats.corr().sum().sort_values(ascending=False)[:5]
bot5 = stats.corr().sum().sort_values(ascending=False)[-5:]

print("The top 5 most influencial stats were \n{0} \nand the bottom 5 were \n{1}".format(top5, bot5))


# Now this indeed is an interesting finding. Abilities like ball control, shortpassing, dribbling were the most influencial ones whereas jumping and strength had almost nothing to do with other abilities. Now, what about with Overall and Value?

# In[ ]:


df = data.copy()

stat_and_val = df[['Overall', 'Value', 'Crossing',
       'Finishing', 'HeadingAccuracy', 'ShortPassing', 'Volleys', 'Dribbling',
       'Curve', 'FKAccuracy', 'LongPassing', 'BallControl', 'Acceleration',
       'SprintSpeed', 'Agility', 'Reactions', 'Balance', 'ShotPower',
       'Jumping', 'Stamina', 'Strength', 'LongShots', 'Aggression',
       'Interceptions', 'Positioning', 'Vision', 'Penalties', 'Composure',
       'Marking', 'StandingTackle']]

stat_and_val.corr()[['Value', 'Overall']].sort_values(by='Value', ascending=False).drop(index=['Value', 'Overall'])


# The result is a little surprising, to be honest. I thought since those excellent finishers are usually the most renowned ones, the stat "Finishing" would have to do much with the value. But data doesn't lie, and as it turns out, skills like reaction, composure and vision tend to decide one's value most.

# Now let's look from the club aspect. See if we can find some interesting pattern there as well.

# In[ ]:


df = data.copy()

## TOP 10 most clubs with the best players
df.groupby("Club")['Overall'].mean().sort_values(ascending=False)[:10]


# In[ ]:


## TOP 10 clubs with the most pricy guys
df.groupby("Club")['Value'].mean().sort_values(ascending=False)[:10]


# In[ ]:


df['Wage'][:5]


# There is a column called 'Wage', and that has the same problem with our original Value column. Let's apply the same function here.

# In[ ]:


data['Wage'] = data['Wage'].apply(change_money)

df = data.copy()

df.groupby("Club")['Wage'].sum().sort_values(ascending=False)[:10]
## TOP 10 clubs with highest payrolls


# Manchester United is thought to be quite an inefficient club, paying too much in wage in comparison to their performance. Let's see if we can define something like, wage efficiency.

# In[ ]:


for_overall = df.groupby("Club")['Overall'].mean()
for_wage = df.groupby("Club")['Wage'].sum()

wage_efficiency = for_overall / for_wage
wage_efficiency.sort_values(ascending=False)[:10]


# In[ ]:


## BOTTOM 10 in terms of wage efficiency
wage_efficiency.sort_values()[:10]


# Well it turns out, all those prominent clubs recorded terrible wage efficiency. That is in fact inevitable, since player's wage rises exponentially as their ability grows. In short, their wages are just astronomical.

# Now, among the top 10 clubs, let's try to draw out their strength and weakness.

# In[ ]:


big_clubs = df.groupby("Club")['Overall'].mean().sort_values(ascending=False)[:10].index
big_clubs


# In[ ]:


df = data.copy()
df[df['Club'].isin(big_clubs)].groupby('Club')['ShortPassing'].mean().sort_values(ascending=False)


# Barca it's tikitaka.

# In[ ]:


df[df['Club'].isin(big_clubs)].groupby('Club')['SprintSpeed'].mean().sort_values(ascending=False)


# In[ ]:


df[df['Club'].isin(big_clubs)].groupby('Club')['BallControl'].mean().sort_values(ascending=False)


# In[ ]:


df[df['Club'].isin(big_clubs)].groupby('Club')['Aggression'].mean().sort_values(ascending=False)


# In[ ]:


df[df['Club'].isin(big_clubs)].groupby('Club')['Age'].mean().sort_values(ascending=False)


# Now I'll try to see with position segments in filter. In other words, I'll first divide our dataset in accordance with each player's position, whether it be FW, MF, DF and GK.

# In[ ]:


df = data.copy()
df.keys()


# In[ ]:


positions = df[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']]

positions.head()


# In[ ]:


df.iloc[3]


# Now we know that NaN values come from GKs, who belongs nowhere in position df. We exclude GKs for now, only looking at the field players.

# In[ ]:


for_position = df.dropna(subset=['LS', 'ST', 'RCB', 'CM'], how='all')


# In[ ]:


positions = for_position[['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW',
       'LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM',
       'CDM', 'RDM', 'RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']]


# In[ ]:


len(for_position), len(df)


# Since the positions values are not numbers but string, we need a little wrangling.

# In[ ]:


def position_value(x):
    return int(x[:2])+int(x[3])

positions = positions.applymap(position_value)


# Now that we can approach their position stats, we could assume the position name with highest number would be one's main position.

# In[ ]:


positions.iloc[3][:3]


# In[ ]:


main_position = positions.idxmax(axis=1)

for_position['Position'] = main_position


# In[ ]:


for_position.drop(columns=['Unnamed: 0'], inplace=True)
for_position.head()


# In[ ]:


for_position['Position']


# In[ ]:


positions.columns


# In[ ]:


for_position[for_position['Position']=='LW'].sort_values(by='Overall', ascending=False).iloc[:3]


# We know that LW, RW belong to forward mainly.

# In[ ]:


forward = ['LS', 'ST', 'RS', 'LW', 'LF', 'CF', 'RF', 'RW']
midfield = ['LAM', 'CAM', 'RAM', 'LM', 'LCM', 'CM', 'RCM', 'RM', 'LWB', 'LDM', 'CDM', 'RDM']
defense = ['RWB', 'LB', 'LCB', 'CB', 'RCB', 'RB']

fw = for_position[for_position['Position'].isin(forward)]
mf = for_position[for_position['Position'].isin(midfield)]
df = for_position[for_position['Position'].isin(defense)]

len(fw), len(mf), len(df)


# In[ ]:


## Statements on positions aggregate

print("Among pro field players, number of forwards were {0}({1:.2f}%), midfielders were {2}({3:.2f}%), and defenders were {4}({5:.2f}%).".format(len(fw), len(fw)*100/len(for_position), len(mf), len(mf)*100/len(for_position), len(df), len(df)*100/len(for_position)))


# In[ ]:




