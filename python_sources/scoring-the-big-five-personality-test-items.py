#!/usr/bin/env python
# coding: utf-8

# # Scoring the Big Five Personality Test items
# 
# The Big 5 personality inventory contains 5 factors. Like most personality scales, the Big 5 has a mix of items that positively and negatively load onto these personality factors. For example, the factor Extraversion describes someone who is outgoing, energetic, talkative, and enjoys human interaction. The first Extraversion item [`EXT1`] is "I am the life of the party.", a positively-keyed item; whereas the second item [`EXT2`] is "I don't talk a lot.", a negatively-keyed item.
# 
# To find out which items are positively or negatively keyed, we can look at the scale documentation on the IPIP website: https://ipip.ori.org/newBigFive5broadKey.htm
# 
# ## Reverse-coding
# 
# Before analyzing the data from a personality test, a psychologist will generally "reverse-code" the items that are negatively-keyed. This results in a dataset where the item values all have a common direction and interpretetion (i.e., a higher value corresponds with more of that trait). Mathematically, it allows you to then compute sums and averages for each of the factors. For example, after scoring the test items, we could compute an individual's average for Extraversion items to get their Extraversion score.
# 
# This version of the Big 5 scale asks individuals to rate their level of agreement from 1 to 5, where 1 is strong disagreement and 5 is strong agreement. Reverse-coding is as simple as subtracting 6 from every reverse-keyed item.
# 
# The code below will accomplish this task.

# In[ ]:


import numpy as np
import pandas as pd
df = pd.read_csv('/kaggle/input/big-five-personality-test/IPIP-FFM-data-8Nov2018/data-final.csv', sep='\t')


# In[ ]:


df.head()


# List the positively- and negatively-keyed items.

# In[ ]:


positively_keyed = ['EXT1', 'EXT3', 'EXT5', 'EXT7', 'EXT9',
                    'EST1', 'EST3', 'EST5', 'EST6', 'EST7', 
                    'EST8', 'EST9', 'EST10',
                    'AGR2', 'AGR4', 'AGR6', 'AGR8', 'AGR9', 'AGR10',
                    'CSN1', 'CSN3', 'CSN5', 'CSN7', 'CSN9', 'CSN10', 
                    'OPN1', 'OPN3', 'OPN5', 'OPN7', 'OPN8', 'OPN9', 
                    'OPN10']

negatively_keyed = ['EXT2', 'EXT4', 'EXT6', 'EXT8', 'EXT10',
                    'EST2', 'EST4',
                    'AGR1', 'AGR3', 'AGR5', 'AGR7', 
                    'CSN2', 'CSN4', 'CSN6', 'CSN8', 
                    'OPN2', 'OPN4', 'OPN6']


# Subtract 6 from each negatively-keyed item.

# In[ ]:


df.loc[:, negatively_keyed] = 6 - df.loc[:, negatively_keyed]


# In[ ]:


df.head()

