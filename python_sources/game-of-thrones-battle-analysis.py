#!/usr/bin/env python
# coding: utf-8

# **Hey, I am trying to perform attacker vs defender analysis for GoT battles. Let's start!**
# ------------------------------------------------------------------------
# 
# First, let's import the packages we'll be using in this kernel.

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Let's take a look at the data.

# In[ ]:


battles = pd.read_csv("../input/battles.csv")
battles.head()


# Let me choose the required fields only

# In[ ]:


battles = battles[['name','year','attacker_king','defender_king','attacker_outcome']]
battles.head()


# Producing the battle outcome in integer format so that it will be easier for me later

# In[ ]:


battles['outcome'] = (battles['attacker_outcome'] == 'win')*1
battles.head()


# Plotting number of battles the attackers were involved in a bar graph format

# In[ ]:


attack = pd.DataFrame(battles.groupby("attacker_king").size().sort_values())
attack = attack.rename(columns = {0:'Battle'})
attack.plot(kind='bar')


# Plotting number of battles the defenders were involved in a bar graph format

# In[ ]:


defend = pd.DataFrame(battles.groupby("defender_king").size().sort_values())
defend = defend.rename(columns = {0:'Battle'})
defend.plot(kind='bar')


# Pivoting and comparing attacker vs defender battles

# In[ ]:


pvt = battles.pivot_table(index='attacker_king',columns='defender_king',aggfunc='sum',values='outcome')
pvt


# The heatmap shows a good picture of the battles. Looks like Joffrey's the meanest attacker in the history of Game of Thrones. 9 battles won against the Starks alone! Sweet!

# In[ ]:


sns.heatmap(pvt,annot=True)
sns.plt.suptitle('attacker win')


# Thank you for reading!
