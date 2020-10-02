#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns 
from collections import Counter

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


premier_league=pd.read_csv('../input/epl-results-19932018/EPL_Set.csv')


# In[ ]:


premier_league.tail()


# In[ ]:


premier_league.info()


# In[ ]:


# arsenal hometeam matches
filter_arsenal_home=premier_league['HomeTeam']=='Arsenal'
arsenal_home=premier_league[filter_arsenal_home]
arsenal_home


# In[ ]:


# list of seasons
season_list=list(arsenal_home['Season'].unique())
season_list


# In[ ]:


# finding number of win
win=[]
for i in season_list:
    x=arsenal_home[arsenal_home['Season'] == i]
    counter=0
    for j in x['FTR']:
        if(j=='H'):
            counter=counter+1
    win.append(counter)

win


# In[ ]:


data_home=pd.DataFrame({'Seasons':season_list,'Victory in Emirates':win})
data_home


# In[ ]:


plt.figure(figsize=(15,10))
sns.barplot(x=data_home['Seasons'],y=data_home['Victory in Emirates'])
plt.xticks(rotation=45)
plt.xlabel('Seasons')
plt.ylabel('# of winning in Emirates')
plt.title('Number of winning in home games')


# In[ ]:


# arsenal matches in 2017-2018
filter_arsenal_away=premier_league['AwayTeam']=='Arsenal'
filter_season=premier_league['Season']=='2017-18'
arsenal_whole=premier_league[filter_arsenal_home | filter_arsenal_away]
arsenal_2018=arsenal_whole[filter_season]
arsenal_2018


# In[ ]:


# points in home games
point1=[]
for each in arsenal_2018[arsenal_2018['HomeTeam']=='Arsenal'].loc[:,'FTR']:
    if(each=='H'):
        point1.append(3)
    elif(each == 'D'):
        point1.append(1)
    else:
        point1.append(0)
        
point1


# In[ ]:


# points in away game
point2=[]
for each in arsenal_2018[arsenal_2018['AwayTeam']=='Arsenal'].loc[:,'FTR']:
    if(each=='A'):
        point2.append(3)
    elif(each == 'D'):
        point2.append(1)
    else:
        point2.append(0)
        
        
point2
        
        


# In[ ]:


arsenal_home_2018=arsenal_2018[arsenal_2018['HomeTeam']=='Arsenal']
#arsenal_home_2018
arsenal_home_2018['Point']=point1
arsenal_home_2018


# In[ ]:


arsenal_away_2018=arsenal_2018[arsenal_2018['AwayTeam']=='Arsenal']
arsenal_away_2018['Point']=point2
arsenal_away_2018


# In[ ]:


# date is index
datetime=pd.to_datetime(arsenal_home_2018['Date'])
arsenal_home_2018['date']=datetime
arsenal_home_2018=arsenal_home_2018.set_index('date')
arsenal_home_2018


# In[ ]:


datetime2=pd.to_datetime(arsenal_away_2018['Date'])
arsenal_away_2018['date']=datetime2
arsenal_away_2018=arsenal_away_2018.set_index('date')
arsenal_away_2018


# In[ ]:


#merging by date
merge=pd.concat([arsenal_home_2018,arsenal_away_2018], axis=0)
merge.sort_values(by=['date'], inplace=True, ascending=True)
merge


# In[ ]:


point=[]
counter=0
for each in merge['Point']:
    counter=counter+each
    point.append(counter)
    
merge['points']=point
#merge
merge.drop(columns=['Point'])
weeks=[i for i in range(1,39)]
merge['week']=weeks
merge


# In[ ]:


plt.figure(figsize=(20,10))
sns.pointplot(x='week',y='points',data=merge,color='red',alpha=0.8)

plt.grid()

