#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


train=pd.read_csv('../input/pubg-finish-placement-prediction/train_V2.csv')


# In[ ]:


train.describe()


# In[ ]:


train[train['winPlacePerc'].isnull()]


# In[ ]:


train['weaponsAcquired'].value_counts().tail(90)


# In[ ]:


train.info()


# In[ ]:


train['kills'].value_counts().tail(90)


# In[ ]:


#d1=(df['matchType']=='crashfpp') | (df['matchType']=='crashtpp') | (df['matchType']=='flarefpp') | (df['matchType']=='flaretpp') | (df['matchType']=='normal-duo') | (df['matchType']=='normal-duo-fpp') | (df['matchType']=='normal-solo') | (df['matchType']=='normal-solo-fpp') | (df['matchType']=='normal-squad') | (df['matchType']=='normal-squad-fpp')
#df[d1]


# In[ ]:


train['matchId'].value_counts().tail(10)


# In[ ]:


train[train['matchId'].duplicated()]


# In[ ]:


#df['kills'].quantile(0.99)


# In[ ]:


x=train.groupby('matchType')
x.size()


# # Cleaning
# 1. there are some values of the col teamKills is greater 4 which is not possible -
# 2. in some rows of weaponsAcquired col have much greater values -
# 3. one nan value in winPlacePerc col -
# 4. few col have inappropriate data type -
# 5. killPlace > 100 is not possible -
# 6. headshotKills and roadKills column is that much important as we can count the kills from kills column -
# 7. killPoints column is not accurate -
# 8. revive column is also not that much important -
# 9. there are some unsusal types of matches in matchType column -

# In[ ]:


df=train.copy()

# dropping headshotKills column
df.drop(columns=['headshotKills'],inplace=True)

# dropping revives column
df.drop(columns=['revives'],inplace=True)

# dropping roadKills column
df.drop(columns=['roadKills'],inplace=True)

# dropping killPoints column
df.drop(columns=['killPoints'],inplace=True)

# dropping killPlace column
df.drop(df[df['killPlace'] > 100].index,inplace=True)
# Killing more than 4 teammates is not possible, 
#So I am dropping the rows where the value of teamKills column is greater than 4
df.drop(df[df['teamKills'] > 4].index,inplace=True)

# Fixing the unsusal types of matches in matchType column
df.drop(df[(df['matchType']=='crashfpp') | (df['matchType']=='crashtpp') | (df['matchType']=='flarefpp') | (df['matchType']=='flaretpp') | (df['matchType']=='normal-duo') | (df['matchType']=='normal-duo-fpp') | (df['matchType']=='normal-solo') | (df['matchType']=='normal-solo-fpp') | (df['matchType']=='normal-squad') | (df['matchType']=='normal-squad-fpp')].index,inplace=True)

# Filling nan value detected in winPlacePerc column
df['winPlacePerc'].fillna(df['winPlacePerc'].mean(),inplace=True)

# Fixing inappropriate data types
df['teamKills']=df['teamKills'].astype('category')
df['vehicleDestroys']=df['vehicleDestroys'].astype('category')
#df['kills']=df['kills'].astype('category')
df['matchType']=df['matchType'].astype('category')
#df['killStreaks']=df['killStreaks'].astype('category')
#df['assists']=df['assists'].astype('category')
#df['matchId']=df['matchId'].astype('category')

# As the column weaponsAcquired have some values which are much higher than usual 
# So I droped the rows who's weaponsAcquired value is more than 99 Percentile 
a=df['weaponsAcquired'].quantile(0.99)
df.drop(df[df['weaponsAcquired'] > a].index,inplace=True)

# As the column kills have some values which are much higher than usual 
# So I droped the rows who's kills value is more than 99 Percentile
b=df['kills'].quantile(0.99)
df.drop(df[df['kills'] > b].index,inplace=True)

df.info()


# # Catagory
# - vehicleDestroys
# - teamKills
# - matchType
# # Int
# - DBNOs
# - boosts
# - heals
# - kills
# - killPlace 
# - killStreaks
# - matchDuration 
# - maxPlace 
# - assists
# - rankPoints 
# - weaponsAcquired 
# - winPoints
# - numGroups
# # Float
# - damageDealt
# - longestKill 
# - rideDistance 
# - swimDistance 
# - walkDistance 
# - winPlacePerc
# # Object
# - Id 
# - groupId 
# - matchId

# # UNIVARIATE ANALYSIS

# In[ ]:


# Analizing the column teamKills
plt.figure(figsize=(20,6))
sns.countplot(df['teamKills'])


# In[ ]:


# Analizing the column vehicleDestroys
plt.figure(figsize=(20,6))
sns.countplot(df['vehicleDestroys'])


# In[ ]:


# Analizing the column matchType
plt.figure(figsize=(20,6))
sns.countplot(df['matchType'])


# In[ ]:


# Analizing the column matchDuration
plt.figure(figsize=(20,2))
sns.boxplot(df['matchDuration'])
print(df['matchDuration'].skew())
print(df['matchDuration'].kurt())


# In[ ]:


print("players with survival time between 400s to 600 ",df[(df['matchDuration']>400) & (df['matchDuration']<600)].shape[0])
print("players with survival time between 100s to 400 ",df[(df['matchDuration']>100) & (df['matchDuration']<400)].shape[0])
print("players with survival time less than 100 ",df[(df['matchDuration']<100)].shape[0])


# **CONCLUSION**
# - players with survival time less than 100 may be consider as outliers

# In[ ]:


# Analizing the column kills
plt.figure(figsize=(20,2))
sns.boxplot(df['kills'])
print(df['kills'].skew())
print(df['kills'].kurt())


# In[ ]:


print("players with kills between 3 to 5 ",df[(df['kills']>3) & (df['kills']<=5)].shape[0])
print("players with kills between 5 to 7 ",df[(df['kills']>5) & (df['kills']<=6)].shape[0])
print("players with kills more than 30 ",df[(df['kills']>=7)].shape[0])
print("players who died without any kil ",df[(df['kills']==0)].shape[0])


# **CONCLUSION**
# - A lot of player died without any kill
# - May be Outliers are present in the data

# In[ ]:


# Analizing the column weaponsAcquired
plt.figure(figsize=(20,2))
sns.boxplot(df['weaponsAcquired'])
print(df['weaponsAcquired'].skew())
print(df['weaponsAcquired'].kurt())


# **CONCLUSION**
# - Quite good data
# - less outliers are present

# In[ ]:


# Analizing the column numGroups
plt.figure(figsize=(20,2))
sns.distplot(df['numGroups'])
print(df['numGroups'].skew())
print(df['numGroups'].kurt())


# **CONCLUSION**
# - Quite good result 
# - outliers may be present in tha data
# 

# In[ ]:


# Analizing the column killStreaks
plt.figure(figsize=(20,2))
sns.boxplot(df['killStreaks'])
print(df['killStreaks'].skew())
print(df['killStreaks'].kurt())


# **CONCLUSION**
# - As we saw earlier that there is a lot of player died withot any kill that means killStreaks for those players is also ZERO.
# - Outliers may by present

# In[ ]:


# Analizing the column maxPlace
plt.figure(figsize=(20,2))
sns.boxplot(df['maxPlace'])
print(df['maxPlace'].skew())
print(df['maxPlace'].kurt())


# **CONCLUSION**
# - Outliers might not present

# In[ ]:


# Analizing the column winPlacePerc
plt.figure(figsize=(20,2))
sns.boxplot(df['winPlacePerc'])
print(df['winPlacePerc'].skew())
print(df['winPlacePerc'].kurt())


# **CONCLUSION**
# - Consistent data 

# # MULTIVARIATE ANALYSIS

# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='winPlacePerc',y='kills',hue='teamKills',data= df)


# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='winPlacePerc',y='kills',hue='vehicleDestroys',data= df)


# In[ ]:


plt.figure(figsize=(20,10))
sns.scatterplot(x='winPlacePerc',y='kills',hue='matchType',data= df)


# In[ ]:


plt.figure(figsize=(20,10))
sns.jointplot(x='winPlacePerc',y='damageDealt',kind='reg',data=df)


# In[ ]:


sns.catplot(x='winPlacePerc',y='matchType',kind='box',data=df)


# In[ ]:


plt.figure(figsize=(20,10))
sns.distplot(df[df['kills']==2]['winPlacePerc'])
sns.distplot(df[df['kills']==3]['winPlacePerc'])
sns.distplot(df[df['kills']==4]['winPlacePerc'])
sns.distplot(df[df['kills']==5]['winPlacePerc'])
sns.distplot(df[df['kills']==6]['winPlacePerc'])


# # Feature Engineering

# In[ ]:


# We are creating a new column named distance by adding walkDistance,rideDistance and swimDistance 
df['distance'] = df['rideDistance'] + df['swimDistance'] + df['walkDistance']


# In[ ]:


df.info()


# In[ ]:


df['matchDuration'].mean()


# In[ ]:


def distance_type(y):
    if y<1500:
        return "short"
    elif y>=1500 and y<=2000:
        return "medium"
    else:
        return "long"
    


# In[ ]:


df['distance_Type']=df['distance'].apply(distance_type)


# In[ ]:


df.drop(columns=['distance','rideDistance','swimDistance','walkDistance'],inplace=True)


# In[ ]:


df['distance_Type']=df['distance_Type'].astype('category')


# In[ ]:


pd.crosstab(df['distance_Type'],df['matchType']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# In[ ]:


def winning(w):
    if w<=0.4:
        return "Low"
    elif w>0.4 and w<=0.7:
        return "Average"
    else:
        return "High"


# In[ ]:


df['winning']=df['winPlacePerc'].apply(winning)


# In[ ]:


df.info()


# In[ ]:


df.drop(columns=['winPlacePerc'],inplace=True)


# In[ ]:


pd.crosstab(df['winning'],df['matchType']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# In[ ]:


sns.catplot(x='matchType',hue='winning',kind='count',data=df)


# In[ ]:


def maxP(m):
    if m<=30:
        return "Low"
    elif m>30 and m<=70:
        return "Average"
    else:
        return "High"
df['maxP']=df['maxPlace'].apply(maxP)
df.drop(columns=['maxPlace'],inplace=True)


# In[ ]:


pd.crosstab(df['winning'],df['maxP']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# In[ ]:


def killPlacea(k):
    if k<=30:
        return "Lower"
    elif k>30 and k<=70:
        return "Average"
    else:
        return "Higher"
df['killPlacea']=df['killPlace'].apply(killPlacea)
df.drop(columns=['killPlace'],inplace=True)


# In[ ]:


pd.crosstab(df['winning'],df['killPlacea']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# In[ ]:


sns.catplot(x='killPlacea',hue='winning',kind='count',data=df)


# In[ ]:


def damage(d):
    if d<=350:
        return "Lower"
    elif d>350 and d<=700:
        return "Average"
    else:
        return "Higher"
df['damage']=df['damageDealt'].apply(damage)
df.drop(columns=['damageDealt'],inplace=True)


# In[ ]:


pd.crosstab(df['winning'],df['damage']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# In[ ]:


def Duration(d1):
    if d1<=1200:
        return "Lower"
    elif d1>1200 and d1<=1800:
        return "Average"
    else:
        return "Higher"
df['Duration']=df['matchDuration'].apply(Duration)
df.drop(columns=['matchDuration'],inplace=True)


# In[ ]:


pd.crosstab(df['winning'],df['Duration']).apply(lambda r: round((r/r.sum())*100,1),axis=1)


# # ONE HOT ENCODING

# In[ ]:


df=pd.get_dummies(data=df,columns=['distance_Type','winning','maxP','killPlacea','damage','Duration'],drop_first=True)


# In[ ]:


plt.figure(figsize=(20,7))
sns.heatmap(df.corr(),linewidth=1,annot=True)


# # CONCLUSION
# ***WHAT TO DO IN PUBG***
# * **Higher no. of kills means higher chance to win**
# * **Lower kill palce means higher chance to win**
# * **The no of weapon picked up means higher chance to win**
# * **There is a high chance of win tha game if you are playing in a squad-fpp match**
# * **There is also a higher chance to win if tarvel distance is higher**
# ***SO WE CAN SAY THAT IF ANY PLAYER PLAYING A SQUAD-FPP MATCH WITH SUFFICIENT WEAPONS WITH MORE THAN 5 KILLS IN LOWER KILL PLACE AND WITH MUDEUM TRAVEL DISTANCE CAN WIN THE MATCH***
# 
# ***WHAT NOT TO DO IN PUBG***
# * **Every time look for a kill not for the damage or assist**
# * **dont stay in the same place for the entier match**
