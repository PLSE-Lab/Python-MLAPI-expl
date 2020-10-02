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


# ## Libraries upload 

# In[ ]:



import requests
import re
import pandas as pd
from bs4 import BeautifulSoup
from time import sleep
from selenium import webdriver
import numpy as np 


# ## Getting HTML 

# In[ ]:


driver = webdriver.Chrome('chromedriver/chromedriver')
#request url 
driver.get('https://www.goalzz.com/?region=-7&team=144')
# give it some time
sleep(5)
# retrive , download html page 
html = driver.page_source
# close 
driver.close()


# In[ ]:


len(html) 


# In[ ]:


soup = BeautifulSoup(html,'lxml')


# ## the start of screaping starting match dates 

# In[ ]:


date = soup.find_all('div',attrs={'class':'matchDate'})


# In[ ]:


match_dates = []


# In[ ]:


for i in date:
    match_dates.append(i.text)


# In[ ]:


match_dates


# In[ ]:


df = pd.DataFrame()


# In[ ]:


df['date'] = match_dates


# In[ ]:


df


# ## extracting teams 

# In[ ]:


teams = soup.find_all('a',attrs={'class':'tl'})


# In[ ]:


27 * 2


# In[ ]:


teamss = []


# In[ ]:


for i in teams:
    teamss.append(i.text)


# In[ ]:


teamss.insert(2719,'Y20')


# In[ ]:





# In[ ]:


home_teams = []
away_teams = []


# In[ ]:





# In[ ]:


for i in range(0,len(teamss)):
    if i%2:
         away_teams.append(teamss[i])
    else:
         home_teams.append(teamss[i])
     


# In[ ]:


len(home_teams) , len(away_teams)


# ## extracting results of the matches 

# In[ ]:


score = soup.find_all('td',attrs={'class':'sc'})


# In[ ]:


scores = []


# In[ ]:


for i in score: 
    scores.append(i.text)


# In[ ]:


scores


# In[ ]:


sc = pd.DataFrame()


# In[ ]:


sc['scores'] = scores


# In[ ]:


sc


# In[ ]:


sc.index


# In[ ]:


sc.loc[0][0]


# In[ ]:


w = []
u = []


# In[ ]:


len(sc.loc[5][0])


# In[ ]:


for i in range(0,1426):
    if len(sc.loc[i][0]) == 10 :
        u.append(sc.loc[i][0])
    else:
        w.append(sc.loc[i][0])
        
    


# In[ ]:


w.insert(1387,'0 : 0')


# In[ ]:


df['home_team'] = home_teams


# In[ ]:


df['away_team'] = away_teams


# In[ ]:


df


# In[ ]:


df['scores'] = w


# In[ ]:


df


# In[ ]:


df.loc[0][1]


# In[ ]:


win_team = []


# ## for loop to determine the wining team and losing team 

# In[ ]:


for i in range(0,1386):
    if df.loc[i][3][0] > df.loc[i][3][4]:
        win_team.append(df.loc[i][1])
    elif df.loc[i][3][0] < df.loc[i][3][4]:
        win_team.append(df.loc[i][2])
    else :
        win_team.append('draw')


# In[ ]:


win_team.insert(1387,0)


# In[ ]:


df['waining_team'] = win_team


# In[ ]:


df


# ## extracting the competition of the match 

# In[ ]:


driver = webdriver.Chrome('chromedriver/chromedriver')
#request url 
driver.get('https://www.goalzz.com/?region=-7&team=144')
xx = []
for i in range(0,1386): 
       xx.append(driver.find_elements_by_xpath(f'//*[@id="matchesTable"]/tr[{i}]/td[1]/font/a'))


# In[ ]:


comp = []


# In[ ]:


for i in xx[1:] : 
       comp.append(i[0].text)
    


# In[ ]:


comp


# In[ ]:


df = df[df['date'] != 'N/A']


# In[ ]:


df['competiton'] = comp


# In[ ]:


df


# In[ ]:


df = df.rename(columns={'waining_team' : 'wining_team'})


# In[ ]:


df = df[['home_team' , 'scores' , 'away_team' , 'date' , 'wining_team' , 'competiton']]


# In[ ]:


df


# In[ ]:


stage = soup.find_all('td',attrs={'id':'jm10x5'})


# In[ ]:


st = []


#  ## extracting the stage of the match with a for loop

# In[ ]:


for i in range (1,1387) : 
    stage = soup.find_all('td',attrs={'id':'jm{}x5'.format(i)})
    st.append(stage[0].text)


# In[ ]:


st


# In[ ]:


df['stage'] = st[:1363]


# In[ ]:


df


# ## cleaning 
# 

# In[ ]:


dff = df 


# In[ ]:


dff.stage[dff['stage'] == ''] = np.nan


# In[ ]:


dff.isnull().sum()


# In[ ]:


dff


# In[ ]:


year = [] 


# In[ ]:


dff.loc[0][3][:4]


# In[ ]:


for i in range (0,1362) : 
      if len(dff.loc[i][3]) <= 4 : 
            dff.loc[i][3] = '{}/1/1'.format(dff.loc[i][3])


# In[ ]:


dff


# In[ ]:





# In[ ]:


dff['date'] = pd.to_datetime(dff['date'])


# In[ ]:


dff


# In[ ]:





# In[ ]:


dff['home_team'] = dff['home_team'].str.replace(' ','_')


# In[ ]:


dff['home_team']


# In[ ]:


dff['away_team'] = dff['away_team'].str.replace('-','')


# In[ ]:


dff['away_team'] = dff['away_team'].str.replace(' ','_')


# In[ ]:


dff


# In[ ]:


dff.to_csv('Al_Hilal Scores Archive.csv')


# In[ ]:


dff.info()


# In[ ]:





# In[ ]:




