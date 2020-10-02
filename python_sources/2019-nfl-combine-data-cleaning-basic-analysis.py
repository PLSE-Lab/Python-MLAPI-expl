#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols


# In[ ]:


# load data file
combine = pd.read_csv('../input/2019-nfl-scouting-combine/2019_nfl_combine_results.csv')


# Let's inspect first few rows to get a feel for how the data looks

# In[ ]:


print(combine[0:3])


# Let's rename some of the variable names for easier access later on

# In[ ]:


combine.rename(columns = {"Hand Size (in)": "hand_size", "Weight (lbs)":"weight", 'Height (in)':'height', 
                         "Arm Length (in)":"arm_length",'Vert Leap (in)':'vert','Broad Jump (in)':'broad', '40 Yard':'dash'}, inplace = True)
print(combine[0:3])


# Let's see all positions that are represented so we can group them for easier analysis

# In[ ]:


combine.POS.value_counts()


# In[ ]:


#Assign posititons to offense, defense, or special 
combine.loc[(combine['POS'] == "C") |(combine['POS'] == "FB") |(combine['POS'] == "OG") |(combine['POS'] == "OL") |
            (combine['POS'] == "OT") |(combine['POS'] == "QB") |(combine['POS'] == "RB") |(combine['POS'] == "TE") |
            (combine['POS'] == "WR") , "Side"] = "Offense"
combine.loc[(combine['POS'] == "CB") |(combine['POS'] == "DE") |(combine['POS'] == "DT") |(combine['POS'] == "EDG") |
            (combine['POS'] == "LB") |(combine['POS'] == "S"), "Side"] = "Defense"
combine.loc[(combine['POS'] == "K") |(combine['POS'] == "P") |(combine['POS'] == "LS"), "Side"] = "Special"


# In[ ]:


#Assign posititons to posistion zone
combine.loc[(combine['POS'] == "C") |(combine['POS'] == "OG") |(combine['POS'] == "OL") |
            (combine['POS'] == "OT") |(combine['POS'] == "DT") |(combine['POS'] == "DE") |(combine['POS'] == "EDG") 
            |(combine['POS'] == "TE") , "Zone"] = "Line"
combine.loc[(combine['POS'] == "LB") |(combine['POS'] == "RB")|(combine['POS'] == "FB"), "Zone"] = "Mid"
combine.loc[(combine['POS'] == "QB"), "Zone"] = "QB"
combine.loc[(combine['POS'] == "CB") |(combine['POS'] == "S")|(combine['POS'] == "WR"), "Zone"] = "Skilled"
combine.loc[(combine['POS'] == "K") |(combine['POS'] == "P") |(combine['POS'] == "LS"), "Zone"] = "Special"

print(combine[0:5])


# Which zone is the most represented?

# In[ ]:


combine.Zone.value_counts()


# Does offense or defense has more players in the combine?

# In[ ]:


combine.Side.value_counts()


# How many players came from each college?

# In[ ]:


combine.College.value_counts()


# Now that the data is cleaned, it is ready for statistical analysis: ANOVA on position zone and hand size, and multiple regression to predict factors impacting 40 yard dash times. The analysis will be performed with R in a following notebook.
