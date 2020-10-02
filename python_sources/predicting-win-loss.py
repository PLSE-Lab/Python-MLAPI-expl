#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


# League of Legends is a multiplayer online battle arena game made by Riot Games. Players play as team of 5 and the gameplay involves destroying towers, inhibitors and ultimately the enenemy team's nexus to secure the victory. The game can also end with the enemy team's surrender should they feel victory is impossible. During the battle to destory each other's nexus, teams can acquire leads and gold advantage by killing enemy champions or securing neutral objectives such as dragons and baron nashor. 
# 
# In this analysis, I will try and predict the outcome (win/loss) of a game based on game duration, number of towers and inhibitors a team destroyed and number of neutral objectives they have secured.  

# #### Importing data

# In[ ]:


data = pd.read_csv('../input/games.csv')
CDict= pd.read_json('../input/champion_info_2.json')
SDict= pd.read_json('../input/summoner_spell_info.json')


# #### Identifying which variables correlates best with outcome

# In[ ]:


data1 = data[['winner','firstBlood','t1_towerKills','t1_inhibitorKills','t1_baronKills','t1_dragonKills','t1_riftHeraldKills']]
data1.replace({'winner': {2:0}},inplace=True)
data1['firstBlood'].replace(2, 0,inplace=True)
graph = plt.figure(figsize=(7,7))
sns.heatmap(data1.corr(), annot=True,square=True)
plt.show()

data2 = data[['winner','firstBlood','t2_towerKills','t2_inhibitorKills','t2_baronKills','t2_dragonKills','t2_riftHeraldKills']]
data2.replace({'winner': {1:0}},inplace=True)
data2.replace({'winner': {2:1}},inplace=True)
data2.replace({'firstBlood':{1:0}},inplace=True)
data2.replace({'firstBlood':{2:1}},inplace=True)
graph = plt.figure(figsize=(7,7))
sns.heatmap(data2.corr(), annot=True,square=True)
plt.show()


# Expectedly, number of towers and inhibitors have the highest correlation with victory. Simply because of the structure of the game, there are certain numbers of towers and inhibitors a team has to destroy in order to approach the enemy teams nexus. I'm surprised to see that number of dragon kills has a higher correlation to outcome than number of baron kills since securing baron gives the team a stronger buff and more gold. First blood has a very weak correlation to outcome, hence noone should give up and type ff15 after giving first blood- every game is winnable.  

# #### Applying champion & spell dictionary to dataset

# In[ ]:


champInfo = pd.read_json((CDict['data']).to_json(), orient= 'index')
Spellinfo= pd.read_json((SDict['data']).to_json(), orient='index')
champCols = ['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id',
             't2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id']
banCols = ['t1_ban1','t1_ban2','t1_ban3','t1_ban4','t1_ban5',
             't2_ban1','t2_ban2','t2_ban3','t2_ban4','t2_ban5']
sumSpellsCols = ['t1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2',
                 't1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1','t1_champ5_sum2','t2_champ1_sum1','t2_champ1_sum2',
                 't2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2',
                 't2_champ5_sum1','t2_champ5_sum2']

champs = champInfo[['id','name']]
champ_dict= dict(zip(champs['id'], champs['name']))
print(champ_dict)

for c in champCols:
    pick = data[c].replace(champ_dict, inplace=True)
for b in banCols:
    ban = data[b].replace(champ_dict, inplace=True)

spell = Spellinfo[['id', 'name']]
spell_dict= dict(zip(spell['id'],spell['name']))
for s in sumSpellsCols:
    spell = data[s].replace(spell_dict, inplace=True)


# #### Exploratory Data Analysis: Most picked and banned champions & Most picked summoner spells 

# In[ ]:


Picks = data.loc[:,['t1_champ1id','t1_champ2id','t1_champ3id','t1_champ4id','t1_champ5id',
             't2_champ1id','t2_champ2id','t2_champ3id','t2_champ4id','t2_champ5id']]
picksum = Picks.apply(pd.Series.value_counts)
Top_Pick =picksum.sum(axis=1).sort_values(ascending = False).head(10)
Top_Pick= Top_Pick.to_frame()
# Top picked champions are - Thresh, Trist, Vayne, Kayn, Lee, Twitch, Janna, Lucian, Jhin, Jinx
sns.barplot(x= Top_Pick.index, y= Top_Pick[0], palette='rocket')
plt.title('Most Picked Champions')
plt.xticks(rotation='vertical')
plt.show()

Bans = data.loc[:,['t1_ban1','t1_ban2','t1_ban3','t1_ban4','t1_ban5',
             't2_ban1','t2_ban2','t2_ban3','t2_ban4','t2_ban5']]
Bansum= Bans.apply(pd.Series.value_counts)
Top_Ban= Bansum.sum(axis=1).sort_values(ascending = False).head(10)
Top_Ban= Top_Ban.to_frame()
sns.barplot(x= Top_Ban.index, y=Top_Ban[0], palette='rocket')
plt.title('Most Banned Champions')
plt.xticks(rotation='vertical')
plt.show()

Sums = data.loc[:, ['t1_champ1_sum1','t1_champ1_sum2','t1_champ2_sum1','t1_champ2_sum2','t1_champ3_sum1','t1_champ3_sum2',
                 't1_champ4_sum1','t1_champ4_sum2','t1_champ5_sum1','t1_champ5_sum2','t2_champ1_sum1','t2_champ1_sum2',
                 't2_champ2_sum1','t2_champ2_sum2','t2_champ3_sum1','t2_champ3_sum2','t2_champ4_sum1','t2_champ4_sum2',
                 't2_champ5_sum1','t2_champ5_sum2']]
Sumsum= Sums.apply(pd.Series.value_counts)
Top_Sum= Sumsum.sum(axis=1).sort_values(ascending=False).head(10)
Top_Sum= Top_Sum.to_frame()
sns.barplot(x= Top_Sum.index, y=Top_Sum[0], palette='rocket')
plt.title('Most Used Summoner Spells')
plt.show()


# For our KNN-prediction, we will specifically look at Tristana, the notorious late game carry. Our target variable will obviously be match outcome- a win or a loss. Our feature variables will include the main objectives (Tower, Inhibitor, Dragon, Rift Herald & Baron) as well as firstblood and game duration.

# In[ ]:



Tristana= pd.DataFrame()
Tristana['gameDuration']= data['gameDuration'].astype(int)
def which_team(t):
    if (t['t1_champ1id'] == 'Tristana') or (t['t1_champ2id'] == 'Tristana') or (t['t1_champ3id'] == 'Tristana')            or(t['t1_champ4id'] == 'Tristana') or (t['t1_champ5id'] == 'Tristana'):
        return 1
    else:
        return 2
he = data.apply(which_team, axis=1)
data['Team'] = he

def victory(w):
    if w['Team'] == w['winner']:
        return '1'
    else:
        return '0'
win = data.apply(victory, axis=1)
Tristana['victory'] = win

def blood(w):
    if w['Team'] == w['firstBlood']:
        return '1'
    else:
        return '0'
fb = data.apply(blood, axis=1)
Tristana['FirstBlood'] = fb

def drag(t):
    if t['Team'] == 2:
        return t['t2_dragonKills']
    else:
        return t['t1_dragonKills']
dragon = data.apply(drag, axis=1)
Tristana['Dragon'] = dragon

def bar(t):
    if t['Team'] == 2:
        return t['t2_baronKills']
    else:
        return t['t1_baronKills']
baron= data.apply(bar,axis=1)
Tristana['Baron'] = baron

def tow(t):
    if t['Team'] == 2:
        return t['t2_towerKills']
    else:
        return t['t1_towerKills']
tower = data.apply(tow, axis=1)
Tristana['Tower'] = tower

def inhib(t):
    if t['Team'] == 2:
        return t['t2_inhibitorKills']
    else:
        return t['t1_inhibitorKills']
inhibitor = data.apply(inhib, axis=1)
Tristana['Inhibitor'] = inhibitor

from sklearn.model_selection import train_test_split
data_feature = Tristana[['gameDuration','FirstBlood','Dragon','Baron','Tower','Inhibitor']].values
data_target = Tristana[['victory']].values

X_train, X_test, Y_train, Y_test = train_test_split(
    data_feature,data_target, test_size= 0.33, random_state=21, stratify=data_target)


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
prediction = knn.predict(X_test)
print(knn.score(X_test, Y_test))


# 0.925 accuracy is not bad but also not great. I suppose there are many other factors attributed to winning a game such as number of wards, number of kills and gold difference just to name a few. And since these are all solo-q games, there will always be that one teammate that gets caught and throws the game for you.

# In[ ]:


# ['gameDuration','FirstBlood','Dragon','Baron','Tower','Inhibitor']
willwin = np.array([[1145,1,2,0,5,1],[1324,0,1,0,2,1],[2568,0,1,0,6,1]])
print(knn.predict(willwin))


# If a game ends at 19:05 (1145) with Tristana's team securing FirstBlood, 2 Dragon, 5 Towers and 1 inhibitor, our model predicts a victory for Tristana
