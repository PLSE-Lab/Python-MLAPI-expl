#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head(3)


# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar()


# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(figsize=(12,6))


# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12,6),
    color='mediumvioletred'
)


# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12,6),
    color='mediumvioletred',
    fontsize=16
)


# In[ ]:


reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12,6),
    color='mediumvioletred',
    fontsize=16,
    title='Rankings Given by Wine Magazine'
)


# In[ ]:


ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12,6),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title('Rankings Given by Wine Magazine', fontsize=20)


# In[ ]:


ax = reviews['points'].value_counts().sort_index().plot.bar(
    figsize=(12,6),
    color='mediumvioletred',
    fontsize=16
)
ax.set_title('Rankings Given by Wine Magazine', fontsize=20)
sns.despine(bottom=True, left=True)


# In[ ]:


pokemon = pd.read_csv("../input/pokemon/Pokemon.csv")
pokemon.head(3)


# In[ ]:


pokemon.plot.scatter(x='Attack', y='Defense', 
                     figsize=(12,6),
                     title='Pokemon by Attack and Defense'
                    )


# In[ ]:


ax = pokemon['Total'].plot.hist(
    figsize=(12,6),
    fontsize=14,
    bins=50,
    color='gray'
)
ax.set_title('Pokemon by Stat Total', fontsize=20)


# In[ ]:


ax = pokemon['Type 1'].value_counts().plot.bar(
    figsize=(12,6),
    fontsize=14
)
ax.set_title('Pokemon by Primary Type', fontsize=20)
sns.despine(bottom=True, left=True)

