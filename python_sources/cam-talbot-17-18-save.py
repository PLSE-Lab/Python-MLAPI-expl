#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import json
import pandas as pd
import matplotlib.pyplot as plt

# Load the data set into pandas
goalies = json.loads(open('../input/goalies.json', 'r').read())['data']
df = pd.DataFrame(goalies)

# Filter to goalie we're interested in, and pull out the relevant data
df = df[df['playerName'] == 'Cam Talbot']
df = df[['gameDate', 'timeOnIce', 'shotsAgainst', 'saves', 'savePctg']]

df.set_index('gameDate')
df.sort_values('gameDate')


# **Cam Talbot 2017/2018 Season**
# 
# While the popular belief is that Cam Talbot had a *bad* season, I wanted to see for myself just how his season was overall.

# In[ ]:


print(df.describe())


# In[ ]:


df['savePctg'].hist(bins=20)
plt.show()


# In[ ]:


df['savePctg'].rolling(5, min_periods=5).mean().plot()
plt.show()


# I believe the data reveals that his numbers were still quite good for the majority of the season, and 3 major dips in his season actually skewed his final numbers.
