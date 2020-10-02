#!/usr/bin/env python
# coding: utf-8

# ### Introduction
# 
# Before beginning any analysis project, it's best to explore the data and see what's going on. This helps formulate ideas about features or measures that you might want to use in your chosen method.
# 
# In this notebook I'll do some quick looks at the data and get an idea of the ranges of variables and the activity of users. For recommendation systems, having a good set of users that 'rate' a lot of items is best to have, so we'll examine that as well.

# In[ ]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import re


# In[ ]:


df = pd.read_csv("../input/steam-200k.csv", header=None, index_col=None,
                 names=['UserID', 'Game', 'Action', 'Hours', 'Other'])
df.head()


# In[ ]:


# for now, ignore the 'purchase' part
pdf = df[df.Action=='play']
pdf.shape


# In[ ]:


pdf.Hours.describe()


# Someone has a lot of hours into a game! It's also interesting to see that 50% of players don't spend more than 4.5 hours in a game. 

# In[ ]:


pdf[pdf.Hours == pdf.Hours.max()].UserID


# In[ ]:


# Apparently that's all this user has played
pdf[pdf.UserID==73017395]


# In[ ]:


# See the top 15 that games take up the most time
pdf[pdf.Hours > 1000].Game.value_counts(normalize=True)[:15]


# In[ ]:


# Look at the top 20 games by total playing time in hours:
hour_sums = pdf.groupby("Game").Hours.aggregate('sum').sort_values(ascending=False)
hour_sums[:20]


# In[ ]:


# Let's put that in terms of years playing
(hour_sums/(24*365))[:20]


# In[ ]:


fig = plt.figure(figsize=(9, 5))
ax = ((hour_sums/(24*365))[:20]).plot(kind='barh')
ax.set_xlabel("Cumulative years spent playing");


# Now let's take a look at the user counts and see the distribution of how many games users play

# In[ ]:


# This counts the number of unique user IDs and runs basic stats on them
pdf.UserID.value_counts().describe()


# In[ ]:


# Let's look at the same for all the data (purchased AND played)
df.UserID.value_counts().describe()


# For recommendations, then, we see that most users don't play very many games, but they have purchased quite a lot. I would bet that some of this has to do with bundle offers. 
# 
# Either way, most users only appear play a couple/few games. 
# 
# Let's import networkx and analyze which games are played together. The network might be too large for good viz in the notebook. If that's true, we'll look at some centrality measures to see which games are the bridges to others.

# In[ ]:


import networkx as nx
from itertools import combinations


# In[ ]:


# We'll start by looking at played games only
G = nx.Graph()
for user_id in pdf.UserID.unique():
    games = pdf[pdf.UserID==user_id].Game.tolist()
    if len(games) > 1:
        for g1, g2 in combinations(sorted(games), 2):
            if G.has_edge(g1, g2):
                G[g1][g2]['weight'] += 1
            else:
                G.add_edge(g1, g2, weight=1)


# In[ ]:





# In[ ]:




