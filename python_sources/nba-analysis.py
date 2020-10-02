#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# # The Data present is from 2004-2020

# In[ ]:


game_details = pd.read_csv('/kaggle/input/nba-games/games_details.csv')
game_details.columns


# In[ ]:


round(100*(game_details.isnull().sum()/len(game_details.index)),2)


# In[ ]:


game_details.drop(['GAME_ID','TEAM_ID','PLAYER_ID','START_POSITION','COMMENT','TEAM_ABBREVIATION'],axis = 1,inplace= True)


# In[ ]:


game_details['FTL'] = game_details['FTA'] - game_details['FTM']


# In[ ]:


game_details = game_details.dropna()


# In[ ]:


game_details.shape


# In[ ]:


game_details.info()


# In[ ]:


game_details['MIN'] = game_details['MIN'].str.strip(':').str[0:2]


# In[ ]:


df = game_details.copy()


# In[ ]:


top_activities = df.groupby(by='PLAYER_NAME')['PTS'].sum().sort_values(ascending =False).head(20).reset_index()
plt.figure(figsize=(15,10))
plt.xlabel('POINTS',fontsize=15)
plt.ylabel('PLAYER_NAME',fontsize=15)
plt.title('Top 20 Players in the NBA League',fontsize = 20)
ax = sns.barplot(x=top_activities['PTS'],y = top_activities['PLAYER_NAME'])
for i ,(value,name) in enumerate (zip(top_activities['PTS'],top_activities['PLAYER_NAME'])):
    ax.text(value, i-.05,f'{value:,.0f}',size = 10,ha='left',va='center')
ax.set(xlabel='POINTS',ylabel='PLAYER_NAME')


# ### Just a quick reminder to Kobe fans, the data is from 2004-2020

# ## Let us take a look at some stats of the BEST PLAYER in the league
# 
# ### 1. LeBron James

# In[ ]:


player = df.groupby(['PLAYER_NAME'])
Bron = player.get_group('LeBron James')

plt.figure(figsize=(10,8))
plt.xlabel('POINTS',fontsize = 10)
sns.countplot(Bron['PTS'])
#sns.countplot(Bron['REB'])

plt.xticks(rotation = 90)


# - This graph cleary shows the dominance KING JAMES had over the NBA from 2004-2020

# ## Lets see LeBron Laker VS LeBron Miami

# In[ ]:


Bron_Team = Bron.groupby('TEAM_CITY')
Bron_LA = Bron_Team.get_group('Los Angeles')
Bron_MIAMI = Bron_Team.get_group('Miami')

plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.xlabel('POINTS',fontsize = 10)
plt.title("LeBron as a Laker",fontsize = 12)
sns.countplot(Bron_LA['PTS'])
#sns.countplot(Bron_LA['REB'])
plt.xticks(rotation = 90)

plt.subplot(1,2,2)
#plt.figure(figsize=(8,6))
plt.title("LeBron in Miami",fontsize = 12)
plt.xlabel('POINTS',fontsize = 10)
sns.countplot(Bron_MIAMI['PTS'])
#sns.countplot(Bron_LA['REB'])
plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()


# In[ ]:


plt.figure(figsize=(12,6))

plt.subplot(1,2,1)
plt.xlabel('REBOUNDS',fontsize = 10)
plt.title("LeBron as a Laker",fontsize = 12)
sns.countplot(Bron_LA['REB'])
#sns.countplot(Bron_LA['REB'])
plt.xticks(rotation = 90)

plt.subplot(1,2,2)
#plt.figure(figsize=(8,6))
plt.title("LeBron in Miami",fontsize = 12)
plt.xlabel('REBOUNDS',fontsize = 10)
sns.countplot(Bron_MIAMI['REB'])
#sns.countplot(Bron_LA['REB'])
plt.xticks(rotation = 90)

plt.tight_layout()

plt.show()


# ### We can clearly see that LBJ had a better career at MIAMI
# ### It is quite profound on Lebron's performance as Miami Heat 
# 
# - Note he was in Miami for 4 years.
# - He has been a Laker only for 2 years now.
# 

# In[ ]:


top_3Pointers = df.groupby(by='PLAYER_NAME')['FG3M'].sum().sort_values(ascending =False).head(20).reset_index()
plt.figure(figsize=(15,10))
plt.xlabel('POINTS',fontsize=15)
plt.ylabel('PLAYER_NAME',fontsize=15)
plt.title('Top 20 - 3 point shooters in the NBA League',fontsize = 20)
ax = sns.barplot(x=top_3Pointers['FG3M'],y = top_3Pointers['PLAYER_NAME'])
for i ,(value,name) in enumerate (zip(top_3Pointers['FG3M'],top_3Pointers['PLAYER_NAME'])):
    ax.text(value, i-.05,f'{value:,.0f}',size = 10,ha='left',va='center')
ax.set(xlabel='POINTS',ylabel='PLAYER_NAME')


# ### This graph clearly shows how Stephen Curry has dominated the 3 Pointer Division by far
# 
# - Note Steph Curry made his debut in the **NBA in 2009** and the data we are comparing him with is from 2004. 

# In[ ]:


df['FTL'] = df['FTA'] - df['FTM']
top_Shaqtin = df.groupby(by='PLAYER_NAME')['FTL'].sum().sort_values(ascending =False).head(20).reset_index()


# In[ ]:


plt.figure(figsize=(15,10))
plt.xlabel('POINTS',fontsize=15)
plt.ylabel('PLAYER_NAME',fontsize=15)
plt.title('Top 20 Players in the NBA League with BAD Free Throws Stats',fontsize = 20)
ax = sns.barplot(x=top_Shaqtin['FTL'],y = top_Shaqtin['PLAYER_NAME'])
for i ,(value,name) in enumerate (zip(top_Shaqtin['FTL'],top_Shaqtin['PLAYER_NAME'])):
    ax.text(value, i-.05,f'{value:,.0f}',size = 10,ha='left',va='center')
ax.set(xlabel='Free-Throw losses',ylabel='PLAYER_NAME')


# ### We can see that these players have had the highest Free Throws L's
# 
# - The current Lakers have the top 2 (not so good at Free throw) players.
# - We can see that the Big guys are not that efficient in Free Throws.
# 

# In[ ]:


top_Blocks = df.groupby(by='PLAYER_NAME')['BLK'].sum().sort_values(ascending =False).head(20).reset_index()
plt.figure(figsize=(15,10))
plt.xlabel('BLOCKS',fontsize=15)
plt.ylabel('PLAYER_NAME',fontsize=15)
plt.title('Top 20 Players in the NBA League with highest blocks',fontsize = 20)
ax = sns.barplot(x=top_Blocks['BLK'],y = top_Blocks['PLAYER_NAME'])
for i ,(value,name) in enumerate (zip(top_Blocks['BLK'],top_Blocks['PLAYER_NAME'])):
    ax.text(value, i-.05,f'{value:,.0f}',size = 10,ha='left',va='center')
ax.set(xlabel='Blocks',ylabel='PLAYER_NAME')


# ### The above players have highest Blocks done in the NBA.
# - Dwight Howard leads the chart of both the blocks and Free Throws L's, this clearly shows his role in L.A. Lakers right now.
