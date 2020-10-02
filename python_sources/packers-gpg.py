#!/usr/bin/env python
# coding: utf-8

# # What Happened to the Pack?
# 
# What happened to the Green Bay Packers last year? They started out, a whopping 6-0 and ended up losing the
# division to a lackluster Minnesota Vikings team that is pretty much carried by one player.

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("../input/nflplaybyplay2015.csv")
import warnings
warnings.filterwarnings('ignore')


# The first thing that I want to do is figure out how to extract all the plays that involve the Packers.

# In[ ]:


df.columns.values


# So we want to extract all of the plays in which the packers were either the 'posteam' or the 'DefensiveTeam'.

# In[ ]:


gb_o = df[df['posteam']=='GB']
gb_d = df[df['DefensiveTeam']=='GB']


# In[ ]:


game = {}
game_ids = gb_o.GameID.unique()
for index,id in enumerate(game_ids):
    game[index+1] = id


# In[ ]:


game[1]


# In[ ]:


game_s = pd.Series(game) 


# In[ ]:


game_1 = gb_o[gb_o['GameID']==game[1]]    


# In[ ]:


completions = game_1[game_1['PassOutcome'] == 'Complete']
completions['Yards.Gained']
plt.plot(completions['Yards.Gained'])


# In[ ]:


plt.hist(completions['Yards.Gained'])


# So while it's fun throwing some graphs and making ourselves feel intellectual, this is the time when we must ask ourselves what 
# the heck we are trying to figure out. What's the question? What happened to the packers? 

# In[ ]:


pass_yd = []
rush_yd = []
game_ = []
for i in range(16):
    game_.append(i+1)
    g = gb_o[gb_o['GameID']==game[i+1]] 
    completions = g[g['PassOutcome'] == 'Complete']
    rushes = g[g['RushAttempt'] == 1]
    pass_yd.append(completions['Yards.Gained'].sum())
    rush_yd.append(rushes['Yards.Gained'].sum())


# In[ ]:


plt.plot(game_, pass_yd, label='pass')
plt.plot(game_, rush_yd, label='rush')
plt.legend()


# ### Week 7: Packers vs. Broncos
# 
# This was the week that ended up being the turning point of the season. Let's take a closer look...

# In[ ]:


gb_dn = gb_o[gb_o['GameID']==game[7]] 


# In[ ]:


# The final score
gb_dn[['PosTeamScore', 'DefTeamScore']].tail()


# In[ ]:




