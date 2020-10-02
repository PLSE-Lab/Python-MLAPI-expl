#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_column',None)
pd.set_option('display.max_row',None)

from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 11.7,8.27


# In[ ]:


# Import data
df = pd.read_excel('../input/league-of-legends-world-championship-2019/2019-summer-match-data-OraclesElixir-2019-11-10.xlsx')
df.head()


# In[ ]:


df.info()


# In[ ]:


df.shape


# In[ ]:


teamdf = df.loc[df['position']=='Team',:]
teamdf.head()


# In[ ]:


teamdf.shape


# In[ ]:


playersdf = df.loc[df['position']!='Team',:]
playersdf.head()


# In[ ]:


playersdf.shape


# # **Ban & Pick**
# 
#  Each team has five ban opportunities. Each team has three bans, followed by three picks, two bans, and two champions. So ban1, ban2, and ban3 don't use it in our team, but I think that when the opponent takes it, a nasty champion comes out.
#  
# 

# In[ ]:


bans3 = teamdf[['ban1', 'ban2', 'ban3']].melt()
bans3.head()


# In[ ]:


bans3['value'].value_counts()[:10].plot(kind='barh', figsize=(8, 6))


# What are the most loved picks?

# In[ ]:


playersdf['champion'].value_counts()[:10].plot(kind='barh', figsize=(8, 6))


# In addition to the Carry AD (Kai'sa and Xayah) and Grab supporter (Nautilus and Thresh), the lee sin continues to be loved in the jungle. and swappable Ryze, Gragas, Akali. What position did the chosen champions stand in?

# In[ ]:


playersdf.groupby(['champion','position'])['position'].count()


# In[ ]:


pd.crosstab(playersdf.champion,playersdf.position).T.style.background_gradient(cmap='summer_r')


# In fact, in League of Legends Worlds 2019, swappable champions are loved and banned.
# 
# Akali Mid: 24, Top: 11
# 
# Ryze mid: 29, top: 7
# 
# Gragas the Jungle: 40, and the Support: 9
# 
# 
# Qiyana jungle: 11, mid: 10, top: 1
# 
# Renekton Mid: 5, Top: 28
# 
# Syndra AD: 8, Mid: 21
# 
# etc
# 
# In other words, picking a swappable champion (who doesn't know where the champion stands on the line) also puts a high priority on the line. Also for winning objects.

# In[ ]:


teamdf.groupby(['team','result'])['result'].count()


# In[ ]:


pd.crosstab(teamdf.team,teamdf.result).T.style.background_gradient(cmap='summer_r')


#  # LCK
#  
# LCK played Damwon, Griffin and T1 in the LoL Worlds 2019. Let's open the LCK data further.

# In[ ]:


lckdf = df.loc[df['team'].isin(['Damwon Gaming', 'Griffin','SK Telecom T1']) ,:]
lckdf.head()


# In[ ]:


lck_team_df = lckdf[lckdf['player']=='Team']


# In[ ]:


lck_players_df = lckdf[lckdf['player']!='Team']


# In[ ]:


pd.crosstab([lck_team_df.team,lck_team_df.result],lck_team_df.side,margins=True).style.background_gradient(cmap='summer_r')


# The reason for the relatively large number of games in Damwon is that the game has been played since Play In.

# In[ ]:


sns.factorplot('side','result',hue = 'team',data = lck_team_df)
plt.show()


# Unlike the T1 and Griffin, Damwon won a lot when he started with the Blues.

# In[ ]:


lck_players_df['kda'] = (lck_players_df['k'] + lck_players_df['a'])/lck_players_df['d']


# If d is 0, kda is an inf value, so kda is replaced with the sum of k and a.

# In[ ]:


lck_players_df['kda']=lck_players_df['kda'].replace(np.inf,(lck_players_df['k'] + lck_players_df['a']))


# In[ ]:


lck_players_df


# * Who has the highest average kda?

# In[ ]:


lck_players_df.groupby('player')['kda'].mean().nlargest(5)


# Lehends, who was a supporter for Griffin, was the first. Next came AD Viper from the same team, Damwon's mid showmaker, T1's AD Teddy, and Damwon's AD Nuclear.

# * So who was the highest kda player and the champion used at that time?

# In[ ]:


lck_players_df.groupby(['player','champion'])['kda'].max().nlargest(5)


# T1's mid Faker's Tristana, Griffin's supporter Lehends's Rakan, and Griffin's AD viper Xayah were KDA 17, Damwon's Beryl's Yuumi, and Damwon's Jungle Canyon Taliyah, KDA 16. Followed.
