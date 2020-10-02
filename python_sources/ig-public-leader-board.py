#!/usr/bin/env python
# coding: utf-8

# **This Exclude Only 0.97369** (But include their second best)
# 
# **I will try to update as frequently as i can**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


df = pd.read_csv("../input/ig-public-lb/publicleaderboarddata/instant-gratification-publicleaderboard.csv")
df.sort_values(by = 'Score', ascending= False).head()

df = df[~((0.973689 < df["Score"] ) & ( df["Score"] < 0.973699 ))]
#df = df[~((0.973574 < df["Score"] ) & ( df["Score"] < 0.973589 ))]


# In[ ]:


df.tail()


# In[ ]:


P = df[["TeamId","TeamName","Score"]].groupby("TeamId").max()#.sort_values(by = 'Score', ascending= False)


# In[ ]:


Sorted = P.sort_values(ascending= False, by = "Score").head(120).reset_index()


# In[ ]:


Sorted.head(60)


# In[ ]:


Sorted.tail(60)

