#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns

pokemon = pd.read_csv("../input/Pokemon.csv", index_col=0)
pokemon.head(3)


# In[ ]:


g = sns.FacetGrid(pokemon, row="Legendary")
g.map(sns.kdeplot, "Attack")


# In[ ]:


g = sns.FacetGrid(pokemon, row="Generation", col="Legendary")
g.map(sns.kdeplot, "Attack")


# In[ ]:


sns.pairplot(pokemon[['HP', 'Attack', 'Defense']])

