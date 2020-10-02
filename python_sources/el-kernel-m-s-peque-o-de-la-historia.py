#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd # Y de casualidad

pokemon = pd.read_csv('../input/pokemon.csv')
battles = pd.read_csv('../input/test.csv')

battles = battles     .merge(pokemon.rename(lambda x: "f_%s" % x, axis="columns"), left_on="First_pokemon", right_on="f_#")     .merge(pokemon.rename(lambda x: "s_%s" % x, axis="columns"), left_on="Second_pokemon", right_on="s_#") 
sampleSubmission = pd.read_csv('../input/sampleSubmission.csv')
sampleSubmission['Winner'] = battles.apply (lambda row: 0 if row["f_Speed"] > row["s_Speed"] else 1, axis=1)
sampleSubmission.to_csv('my_sub.csv', index=False)
sampleSubmission.head()

