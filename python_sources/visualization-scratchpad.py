#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


reviews = pd.read_csv("../input/wine-reviews/winemag-data_first150k.csv", index_col=0)
reviews.head()


# In[ ]:


reviews['province'].value_counts().head(10).plot.bar()


# In[ ]:


# what percent of the total is Californian vintage?
(reviews['province'].value_counts().head(10)/len(reviews)).plot.bar()


# In[ ]:


#orginal categories
reviews["points"].value_counts().sort_index().plot.bar()


# In[ ]:


#line chart
reviews["points"].value_counts().sort_index().plot.line()


# In[ ]:


reviews['points'].value_counts().sort_index().plot.area()


# In[ ]:


##Histogram for interval variable - price
reviews[reviews['price']<200]['price'].plot.hist()


# In[ ]:


#skewed data
reviews['price'].plot.hist()


# In[ ]:


reviews['points'].plot.hist()


# In[ ]:


reviews['province'].value_counts().head(10).plot.pie()


# In[ ]:


reviews['province'].value_counts().head(10).plot.pie()

# Unsquish the pie.
import matplotlib.pyplot as plt
plt.gca().set_aspect('equal')


# In[ ]:


###BiVariate Plotting
reviews[reviews['price'] <100].sample(100).plot.scatter(x='price', y='points')


# In[ ]:


reviews[reviews['price'] < 100].plot.scatter(x='price', y='points')


# In[ ]:


##Hex plot
reviews[reviews['price']<100].plot.hexbin(x='price', y='points', gridsize=15)


# ### Stacked Plots

# In[ ]:


wine_counts = pd.read_csv("../input/most-common-wine-scores/top-five-wine-score-counts.csv", index_col=0)
wine_counts.head()


# In[ ]:


wine_counts.plot.bar(stacked=True)


# In[ ]:


wine_counts.plot.area()


# In[ ]:


wine_counts.plot.line()


# In[ ]:


pd.set_option('max_columns', None)
pokemon = pd.read_csv("../input/pokemon/Pokemon.csv", index_col=0)
pokemon.head()


# In[ ]:


pokemon['Type 1'].value_counts().plot.bar()


# In[ ]:


pokemon['HP'].value_counts().sort_index().plot.line()


# In[ ]:


pokemon.plot.scatter(x="Attack", y="Defense")


# In[ ]:


pokemon.plot.hexbin(x="Attack", y="Defense", gridsize=15)


# In[ ]:


pokemon_stats_legendary = pokemon.groupby(['Legendary', 'Generation']).mean()[['Attack', 'Defense']]
pokemon_stats_legendary 


# In[ ]:


pokemon_stats_legendary.plot.bar(stacked=True)


# In[ ]:


pokemon_stats_by_generation  = pokemon.groupby('Generation').mean()[['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed' ]]
pokemon_stats_by_generation 


# In[ ]:


pokemon_stats_by_generation.plot.line()


# In[ ]:




