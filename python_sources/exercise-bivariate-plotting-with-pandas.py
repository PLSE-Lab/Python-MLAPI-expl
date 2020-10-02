#!/usr/bin/env python
# coding: utf-8

# **Scatter plot**

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head()


# In[ ]:


reviews[reviews['price'] < 100].sample(100).plot.scatter(x='price', y='points')


# In[ ]:


reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')


# **Hexplot**

# In[ ]:


reviews[reviews['price'] < 100].plot.hexbin(x='price', y='points', gridsize=15)


# (note: the x-axis is points, but is missing from the chart due to a bug)

# **Stacked plots**

# In[ ]:


wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv",index_col=0)
wine_counts


# In[ ]:


wine_counts.plot.bar(stacked=True)


# In[ ]:


wine_counts.plot.area()


# Stacked plots are visually very pretty. However, they have two major **limitations**.
# 
# The first limitation is that the second variable in a stacked plot must be a variable with a very limited number of possible values (probably an ordinal categorical, as here). Five different types of wine is a good number because it keeps the result interpretable; eight is sometimes mentioned as a suggested upper bound. Many dataset fields will not fit this critereon naturally, so you have to "make do", as here, by selecting a group of interest.
# 
# The second limitation is one of interpretability. As easy as they are to make, and as pretty as they look, stacked plots make it really hard to distinguish concrete values. For example, looking at the plots above, can you tell which wine got a score of 87 more often: Red Blends (in purple), Pinot Noir (in red), or Chardonnay (in green)? It's actually really hard to tell!

# **Bivariate line chart**

# In[ ]:


wine_counts.plot.line()


# In[ ]:


pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head()


# In[ ]:


pokemon.plot.scatter(x='Attack', y='Defense')


# In[ ]:


pokemon.plot.hexbin(x='Attack', y='Defense', gridsize=15)


# In[ ]:


pokemon_stats_legendary = pokemon.groupby(['Legendary', 'Generation']).mean()[['Attack', 'Defense']]


# In[ ]:


pokemon_stats_legendary.plot.bar(stacked=True)


# In[ ]:


pokemon_stats_by_generation = pokemon.groupby('Generation').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']]


# In[ ]:


pokemon_stats_by_generation.plot.line()

