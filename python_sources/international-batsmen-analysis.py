#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


players = pd.read_csv('../input/international-cricket-players-data/personal_male.csv',parse_dates=True)


# In[ ]:


players.head()


# In[ ]:


players.info()


# **Players born and playing for different countries : 
# West-Indies cricketers are born in diff countries the most**

# In[ ]:


diff_c =players[players['nationalTeam'] != players['country']].groupby('nationalTeam',as_index=False).count().sort_values('name',ascending=False)


# **However we have to Ignore west-Indies because they are made of different Island nations**

# In[ ]:


plt.figure(figsize=(15,15))
sns.heatmap(players.pivot_table(index='nationalTeam',columns='country',values='name',aggfunc='size'),annot=True)


# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(x='name',y='nationalTeam',data=diff_c,orient='h')


# **Players born in India but playing in different countries-Count visual**

# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(players[(players['nationalTeam'] != players['country'])&(players['country']=='India')]['nationalTeam'],orient='v')


# In[ ]:


India=players[players['nationalTeam']=='India']
India['born']=pd.DatetimeIndex(India['dob']).year


# In[ ]:


plt.figure(figsize=(20,6))
sns.countplot(x='born',data=India[India['born']>1980],hue='battingStyle')


# In[ ]:


India.head()


# In[ ]:


India.groupby('battingStyle').count()['name'].plot(kind='bar')


# In[ ]:


plt.figure(figsize=(20,6))
India.groupby('bowlingStyle').count()['name'].plot(kind='bar')


# In[ ]:


plt.figure(figsize=(20,6))
g=sns.FacetGrid(India,col='battingStyle')
g.map(sns.countplot,x='born',data=India[India['born']>1980])


# In[ ]:


lefties=players.pivot_table(index='nationalTeam',columns='battingStyle',values='name',aggfunc='size')
lefties['leftie%']=lefties['Left-hand bat']/(lefties['Left-hand bat']+lefties['Right-hand bat'])*100


# **Leftie% across the national teams**

# In[ ]:


plt.figure(figsize=(10,8))
sns.barplot(y=lefties.index,x=lefties['leftie%'],data=lefties[lefties['Left-hand bat']>20],orient='h')


# In[ ]:


def get_len(x):
    return len(x.split(','))


# In[ ]:


players['team_count']=players['teams'].apply(get_len)


# In[ ]:


players.groupby('nationalTeam').mean()


# **Watchout for Right-hand players bowling Left-handed. That is rare!**

# In[ ]:


plt.figure(figsize=(6,20))
sns.heatmap(India.pivot_table(index='bowlingStyle',columns='battingStyle',values='name',aggfunc='size'),annot=True,cmap='viridis')


# **England has created twice as many cricketers as India has**

# In[ ]:


plt.figure(figsize=(10,8))
sns.countplot(y='nationalTeam',data=players,orient='h')


# In[ ]:




