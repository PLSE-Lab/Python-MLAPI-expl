#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# We need pandas
# Pandas are cute
import pandas as pd


# In[ ]:


########################### Read base submission
############################################################################
submission = pd.read_csv('../input/m5-three-shades-of-dark-darker-magic/submission_v1.csv')


# In[ ]:


########################### Day 1
############################################################################

# Current number of teams       3,215 (day 2 of May)
# Number of teams 2 days before 3,186 
teams_now = 3215
teams_before = 3186
submission['F1'] *= teams_now/teams_before
submission['F1'].sum()


# In[ ]:


########################### Day 2
############################################################################

# In case you'll get a heart attack 
# because of all the MAGIC in this kernel -
# here is South Korean emergency number - 119
magic = submission['F2'].sum()
korea_emergency = 119
submission['F2'] *= magic / (magic + korea_emergency)
submission['F2'].sum()


# In[ ]:


########################### Day 3
############################################################################

# 1913 - The Woman suffrage parade of 1913 takes place in Washington, D.C.
# 1923 - Ankara replaces Istanbul (Constantinople), as the capital of Turkey.
submission['F3'] *= 1913/1923
submission['F3'].sum()


# In[ ]:


########################### Day 4
############################################################################

# I love statistics.
# When you dig in data you can 
# find VERY interesting "facts".
# Did you know that the mean 
# number of sexual partners 
# per person is 1.0073
sex_constant = 1.0073
submission['F4'] /= sex_constant
submission['F4'].sum()


# In[ ]:


########################### Day 5
############################################################################

# It's Friday  
# Friday night is a mix (colors)
# of girls #ffb0e8 and boys #330000
# Resulted mix HEX color code is #995874
color_of_the_night = 995874

# There are millions "Stories"
# We need to make it Mean, really mean
color_of_the_night /= 1000000
submission['F5'] *= color_of_the_night
submission['F5'].sum()


# In[ ]:


########################### Day 6
############################################################################

# Fridays not always pass without "consequences"
# BE CAREFUL - DRINK MODERATELY
# https://www.ncbi.nlm.nih.gov/pubmed/25701909
# Traumatic brain injury and cognition.
# PMID: 25701909 DOI: 10.1016/B978-0-444-63521-1.00037-6

# 1.00037-6 - it can't be just a coincidence
injury = 1.000376
submission['F6'] *= injury
submission['F6'].sum()


# In[ ]:


########################### Day 7
############################################################################

# Let's make it aaaall random
# We will need NumPy

# We will divide the COMPLETELY random number
# from range >1000000 by 1000000
# to have some float number from 0 to 1 

import numpy as np
np.random.seed(198505)
correction = np.random.randint(1000000)/1000000
submission['F7'] *= correction
submission['F7'].sum()


# In[ ]:


########################### Day 8
############################################################################
submission['F8'] *= (100-1)/100 + (100-12)/10000
submission['F8'].sum()


# In[ ]:


########################### Day 11
############################################################################

# Let's not change anything here
# I have a feeling that we should 
# keep origical prediction
submission['F11'] = submission['F11']
submission['F11'].sum()


# In[ ]:


########################### Days 9-19
############################################################################

# Just 1% of his fortune is equivalent to the whole health budget for Ethiopia
for i in range(9,20):
    if i!=11:
        submission['F'+str(i)] *= 1.01 
        print(submission['F'+str(i)].sum())


# In[ ]:


########################### Days 20-28
############################################################################

# Of all the people earning A$100,000 a year under the age of 50, 2% are women. Just 2% are women.
for i in range(20,29):
    submission['F'+str(i)] *= 1.02
    print(submission['F'+str(i)].sum())


# In[ ]:


########################### Export
############################################################################
submission.to_csv('submission.csv', index=False)

