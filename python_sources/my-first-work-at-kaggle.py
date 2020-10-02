#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#-*-coding:utf-8-*-
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


data = pd.read_csv('../input/FIFA 2018 Statistics.csv')
data.head(10)


# In[ ]:


data.info()


# In[ ]:


data.corr()


# In[ ]:


f , ax = plt.subplots(figsize=(18,18))
sns.heatmap(data.corr() , annot=True , linewidths=1 , ax=ax)


# In[ ]:


data.columns


# In[ ]:


data.Saves.plot(kind='line' , color='r' , label='Saves' , linewidth=1 , alpha=0.5 , grid=True , linestyle=':' )
data.Attempts.plot(color='b' , label='Attempts' , linewidth=1 , alpha=0.5 , grid=True , linestyle='-')
plt.legend(loc='upper right') 
plt.xlabel('Saves')
plt.ylabel('Attempts')
plt.title('Saves-Attempts Relation')
plt.show()


# In[ ]:


data.plot(kind='scatter' , x='Own goals' , y='Own goal Time' , alpha=0.3 , color='blue')
plt.xlabel('Own goals')
plt.ylabel('Own goal Time')
plt.title('Own goals-Own goal Time Relation')
plt.show()


# In[ ]:


data.Attempts.plot(kind='hist' , bins=30 , figsize=(16,16))
plt.title('Histogram of the Attempts')
plt.show()


# In[ ]:


data.plot(kind='scatter' , x='Attempts' , y='Goal Scored' , alpha=0.3 ,color='red')
plt.xlabel('Goal Attempts')
plt.ylabel('Goal Scored')
plt.title('Goal Attempts and Goal Scored Relation')
plt.show()


# In[ ]:


#red = data[data.Red == 1]
#red['Team']
data['Goal Scored'].plot(kind='hist' , bins=15 , figsize=(14,14))
plt.title('Histogram of Gaol Scored')
plt.show()


# **In this step, calculate the average of passes. Find to make pass more than average pass and to make less then average pass. I want to calculate here,  the teams that pass more then average are have scored more goals or have scored less goals. If you make a lot of passes you score more goals in this cup. Is that true?**

# In[ ]:


pass_mean = np.mean(data['Passes'])
more_pass_mean = data[data['Passes'] > pass_mean]
less_pass_mean = data[data['Passes'] < pass_mean]
more_pass_mean['Goal Scored'].plot(kind='hist' , bins=15 , figsize=(14,14))
less_pass_mean['Goal Scored'].plot(kind='hist' , bins=15)
plt.title('"blue" >mean pass // "orange" <mean pass')
plt.show()


# **In this step, if goalkeeper saves more than, do the match goes penalty? We will look at the answer to this question.**

# In[ ]:


save_mean = np.mean(data['Saves'])
penalty = data[data['PSO'] == 'Yes']
print('the goalkeeper saves in the mact goes penalty')
print(penalty)
fracs=[penalty[penalty.Saves>save_mean]['Saves'].count() , penalty[penalty.Saves<save_mean]['Saves'].count()]
labels = 'more save' , 'less save'
explode = (0.05 , 0.05)
plt.pie(fracs, explode=explode , labels=labels, autopct='%1.1f%%', shadow=True)
plt.show()


# **In this step, study 1st goal scored time.**

# In[ ]:


data['1st Goal'].plot(kind='hist' , bins=50 , figsize=(14,14))
plt.axis([0,90,0,8])
plt.show()


# **And in this step , study 1st goal time interval 0-30 , 30-60 , 60-90 , no goal**

# In[ ]:


time = data['1st Goal']
zero_thirty = len([i for i in time if i>0 and i<=30])
thirty_sixty = len([i for i in time if i>30 and i<=60])
sizty_ninety = len([i for i in time if i>60 and i<=90])
no_goal = len(time) - (zero_thirty + thirty_sixty + sizty_ninety)
fracs=[zero_thirty , thirty_sixty , sizty_ninety , no_goal]
labels = '0-30' , '30-60' , '60-90' , 'no goal'
explode = (0 , 0.05 , 0 , 0.05)
plt.pie(fracs, explode=explode , labels=labels, autopct='%1.2f%%', shadow=True)
plt.show()


# **In this step, study on-target shoot and off-target shoot of the teams in cup.**

# In[ ]:


teams=[]
[teams.append(i) for i in data['Team'] if i not in teams]
on_target = [np.sum(data[data['Team'] == i]['On-Target']) for i in teams]
off_target = [np.sum(data[data['Team'] == i]['Off-Target']) for i in teams]
ind = np.arange(len(teams))
p1 = plt.bar(ind, on_target, 0.4 , yerr=np.ones(len(teams)))
p2 = plt.bar(ind, off_target, 0.4 , bottom=on_target,yerr=np.ones(len(teams)))
plt.ylabel('Shoots')
plt.title('Total shoot of teams')
plt.xticks(ind, teams , rotation=90)
plt.yticks(np.arange(0, 81, 10))
plt.legend((p1[0], p2[0]), ('On-Target', 'Off-Target'))


# **In this step , study that which teams more than average goal and whict team less than avera team **

# In[ ]:


team_total_goal = [np.sum(data[data['Team'] == i]['Goal Scored']) for i in teams]
plt.pie(team_total_goal , explode=np.ones(len(teams)) ,labels=teams, autopct='%1.2f%%', shadow=True)
plt.show()
mean_goal = sum(team_total_goal) / len(teams)
freq = [len([i for i in team_total_goal if i>mean_goal]) , 
        len([i for i in team_total_goal if i<mean_goal])]
label = 'more than average goal' , 'less than average goal'
plt.pie(freq , explode = [0.1 , 0] , labels=label, autopct='%1.2f%%', shadow=True)
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




