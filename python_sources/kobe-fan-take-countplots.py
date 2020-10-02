#!/usr/bin/env python
# coding: utf-8

# ![](https://heavyeditorial.files.wordpress.com/2020/01/gettyimages-1164646804-1-e1580072467173.jpg?quality=65&strip=all&w=1024)

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


# In[ ]:


data = pd.read_csv('/kaggle/input/nba-all-star-game-20002016/NBA All Stars 2000-2016 - Sheet1.csv')
#games = pd.read_excel('/kaggle/input/nba-all-star-game-20002016/NBA All Star Games (1).xlsx')


# In[ ]:


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# In[ ]:


data


# ## Who has the most All Star games - top 10

# In[ ]:


top_10 =pd.DataFrame(data.Player.value_counts()[:10]).reset_index(drop = False).rename(columns = {'Player' : 'Times', 'index' : 'Player'})
top_10


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data, order = data.Player.value_counts().iloc[:20].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize = 30)
#ax.set_yticklabels(ax.get_yticklabels(), ha = 'center', fontsize = 15)
plt.tight_layout()


# ## Rest in Peace Kobe and Gigi !! You da real MVP, an absolute Legend of the sports.

# ## Lets see which position makes the cut most of the times

# In[ ]:


data.Pos.value_counts()


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Pos',data=data, order = data.Pos.value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)
plt.tight_layout()


# ### Guards make the cut a lot.

# ## Lets analyze some of the popular Players

# In[ ]:


kobe = data[data.Player == 'Kobe Bryant']
kobe


# Well, not much detail here. Lets see Heights

# ## Height

# In[ ]:


height = data.HT.unique().tolist()
height


# In[ ]:


print('The maximum height of an all star player in NBA is' ,max(height))
print('The min height of an all star player in NBA is' ,min(height))


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='HT',data=data, order = data.HT.value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)
plt.tight_layout()


# ##  WOOH! Makes sense I didnt pursue my Basketball career. <br> Anyways I think there is a relation between Guards and 6-11.

# ##  Lets see the guards

# In[ ]:


guard = data[data.Pos == 'G']
guard


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=guard, order = guard.Player.value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# In[ ]:


data.Pos.unique()


# ## SG

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'SG'], order = data['Player'][data.Pos == 'SG'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## SF

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'SF'], order = data['Player'][data.Pos == 'SF'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## F

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'F'], order = data['Player'][data.Pos == 'F'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## PF

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'PF'], order = data['Player'][data.Pos == 'PF'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## C

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'C'], order = data['Player'][data.Pos == 'C'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## GF

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'GF'], order = data['Player'][data.Pos == 'GF'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## PG

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'PG'], order = data['Player'][data.Pos == 'PG'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## FC

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=data[data.Pos == 'FC'], order = data['Player'][data.Pos == 'FC'].value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# In[ ]:


data


# ## Lets see which team has produced the most All Stars. Any guesses? Methinks Spurs or Lakers or Celtics lets see

# In[ ]:


team = pd.DataFrame(data.Team.value_counts()).reset_index().rename(columns = {'Team' : 'Times', 'index' : 'Team'})
team


# # HEAT YALL!!!

# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Team',data=data, order = data.Team.value_counts().iloc[:20].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right", fontsize = 30)
#ax.set_yticklabels(ax.get_yticklabels(), ha = 'center', fontsize = 15)
plt.tight_layout()


# ## Lets see the weights

# In[ ]:


data.WT.unique()


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='WT',data=data, order = data.WT.value_counts().index)
ax.set_xticklabels(ax.get_xticklabels(), fontsize = 30)
plt.tight_layout()


# Lets see the players with more than 200 pounds weight

# In[ ]:


weight_200 = data[data.WT > 200]
weight_200


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=weight_200, order = weight_200.Player.value_counts().iloc[:20].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# Lets see players that weigh less than 200

# In[ ]:


weight_not_200 = data[data.WT < 200]


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=weight_not_200, order = weight_not_200.Player.value_counts().iloc[:20].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# # Lets See the NBA Draft Status

# In[ ]:


data['NBA Draft Status'].value_counts()


# ### 1996 13th pick guy do really well and who is that guy

# In[ ]:


data['Player'][data['NBA Draft Status'] == '1996 Rnd 1 Pick 13'].loc[38]


# # WE LOVE YOU KOBE

# ## Lets check out the foreigners

# In[ ]:


foren = data[data.Nationality != 'United States']
foren


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Player',data=foren, order = foren.Player.value_counts().iloc[:20].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# In[ ]:


plt.figure(figsize = (50,20))
sns.set(style="darkgrid")
ax = sns.countplot(x='Nationality',data=foren, order = foren.Nationality.value_counts().iloc[:20].index)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 40, fontsize = 30)
plt.tight_layout()


# ## Thank you

# In[ ]:




