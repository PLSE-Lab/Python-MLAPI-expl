#!/usr/bin/env python
# coding: utf-8

# The Current Dataset has so many columns, One can analyze the player's performance based on many attributes.
# In this kernel,
# *  I would like to calculate an overall total rating which depends on few attributes
# *  I would like to calculate the best player grouped by Age
# *  Also I would like to show the Jersey Numbers of Top Players.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sn

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
df = pd.read_csv("../input/data.csv")

# Any results you write to the current directory are saved as output.


# In[ ]:


#Let's see a sample
df.sample()


# Since I'm calculating Overall Rating based on few attributes, I would like to access those columns only.

# In[ ]:


df=df[['ID','Name','Age','Nationality','Overall','Potential','Preferred Foot','International Reputation','Skill Moves','Jersey Number',]]


# In[ ]:


# Important Features
df.sample()


# In[ ]:


#let's see the Players Age Histogram PLot
plt.hist(df.Age)


# In[ ]:


#As you can see, Players above Age 40 are few and I would like to remove them.
df.Age.value_counts()


# In[ ]:


df = df[df.Age<=40]


# In[ ]:


df.Age.max()


# In[ ]:


plt.hist(df.Age)


# In[ ]:


df.sample()


# In[ ]:


#Now we can see that , Overall Rating , Potential Rating are a scale out of 100
df.describe()


# Age wise, we can see the Maximum Overall Ratings

# In[ ]:


df.groupby('Age')['Overall'].agg(np.max)


# In[ ]:


df[(df.Age == 16) & (df.Overall == 64)]


# Since there are two players who have Overall Rating same of Age = 16, 
# I would like to calculate 'total_score' which is calculated as 
# **(df.Overall/20 + df.Potential/20 + df['International Reputation'] + df['Skill Moves'])*5**( On a scale of 100)
# Based on this we can decide the best player of respective AGE.

# In[ ]:


df['total_score'] = (df.Overall/20 + df.Potential/20 + df['International Reputation'] + df['Skill Moves'])*5


# In[ ]:


df.total_score


# So, now I have scaled these attributes and given a total_score.

# In[ ]:


df.sample(2)


# In[ ]:


#Now, The Best Players are as follows
best = df.groupby('Age')['total_score'].agg(np.max).reset_index()
age_scores = list(zip(best.Age,best.total_score))
best_players = pd.DataFrame(columns=df.columns)
for index,i in enumerate(age_scores):
    best_players = best_players.append(df[(df.Age == i[0] ) & (df.total_score == i[1])],ignore_index=True )


# So, Finally based on average values of (Overall Rating, Potential Rating, International Reputation, Skill Moves).....
# Below Players are the best among the Players of their Age.
# 

# In[ ]:


best_players


# As we can see above,
# Some big names are visble...
# Ronaldo,Zlatan,Messi,Neymar etc...
# 
# Lets plot based on their Nations..
# France has best Players of different Age Groups.
# No wonder how it won FWC 2018.
# I don't know how many of these players were in 2018 squad.
# 

# In[ ]:


sss = best_players.Nationality.value_counts().reset_index(name='counts')
sn.barplot(x="index", y="counts", data=sss)
plt.xticks(rotation=60)


# And here comes, the best 5 players based on rating,
# * Ronaldo
# * Neymar
# * Zlatan
# * Messi
# * Pogba

# In[ ]:


#Top 5 players with best total_score
best_players.sort_values(by='total_score')[::-1][:5]


# Just for fun, Let's see What are the Jersey numbers of these best players

# In[ ]:


Jersey = best_players['Jersey Number'].value_counts().reset_index(name='counts')
sn.barplot(x="index", y="counts", data=Jersey)
plt.xticks(rotation=60)


# Suprisingly, many of the bests are wearing Jersey numbers
# * 10
# * 7
# * 11
# * 9

# Much more to analyze..I should use other attributes also...Will update soon.

# In[ ]:


df_after_edit = pd.read_csv("../input/data.csv")


# In[ ]:


df_after_edit.sample(3)


# In[ ]:





# In[ ]:




