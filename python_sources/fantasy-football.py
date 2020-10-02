#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
df = pd.read_csv("../input/nflplaybyplay2015.csv")
import warnings
warnings.filterwarnings('ignore')


### SCORING CONSTANTS ###
# Passing
pyd = float(1/25) # 1 pt per 25 pass yard
ptd = 4. # 4 pt per td pass
pi = -2 # -2 pt per pick

# Rushing/Receiving
ryd = float(1/10)
rtd = 6.
f = -2. # fumble

class fant_score(game, player):
    def pass_score():
        


# In[ ]:


df.columns.values


# In[ ]:




