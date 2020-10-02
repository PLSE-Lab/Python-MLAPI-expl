#!/usr/bin/env python
# coding: utf-8

# This is an editable version of a kernel containing all exercises on Data Visualization course on Kaggle (using Pandas and Seaborn).  
# You can find the course itself on this [link:](https://www.kaggle.com/learn/data-visualization)  
# You can unhide the fields to see the source code for each plot, or fork the kernel to try it yourself

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
pokemon = pd.read_csv("../input/pokemon/pokemon.csv")
pokemon.head()


# **I. Univariate plotting with Pandas (exercises)**

# 1. The frequency of Pokemon by type (using a bar chart):  
#   
# *Hint: the column name is 'type1'*

# In[ ]:


pokemon.type1.value_counts().plot.bar()


# 2. The frequency of hit points for a whole set (using a line chart):  
#   
# *Hint: the column name is 'hp'*

# In[ ]:


pokemon.hp.value_counts().sort_index().plot.line()


# 3. The frequency of Pokemon by weight (using a histogram):  
#   
# *Hint: the column name is 'weight_kg'*

# In[ ]:


pokemon.weight_kg.plot.hist()


# **II. Bivariate plotting with Pandas**

# 1. Defense level - Attack level relation for each pokemon (using a scatter plot):  
#   
# *Hint: the column names are 'defense' and 'attack'*

# In[ ]:


pokemon.plot.scatter(x='defense', y='attack')


# 2. Defense-Attack relation (using a Hex plot):  
#   
# *Hint: the column names are 'defense' and 'attack'*

# In[ ]:


pokemon.plot.hexbin(x='defense', y='attack', gridsize=15)


# 3. For each combination of the following Pokemon qualities:  
#     * Legendary (2 possible values), and 
#     * Generation (7 possible values)  
#   
# show average Attack and Defense level (using a Stacked bar chart)  
#   
# *Hint: the column names are 'defense', 'attack', 'is_legendary' and 'generation'*

# In[ ]:


pokemon_stats_legendary = pokemon.groupby(['is_legendary', 'generation']).mean()[['attack', 'defense']]


# In[ ]:


pokemon_stats_legendary.plot.bar(stacked=True)


# 4. For each of the following properties: 
#     * hit points, 
#     * attack level, 
#     * defense level, 
#     * special attack level, 
#     * special defense level, 
#     * speed
#   
# build a chart showing how frequency of such property depends on the Pokemon generation (using a Bivariate Line Chart)  
#   
# *Hint: the column names are 'hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed' and 'generation'*

# In[ ]:


pokemon_stats_by_generation = pokemon.groupby('generation').mean()[['hp', 'attack', 'defense', 'sp_attack', 'sp_defense', 'speed']]


# In[ ]:


pokemon_stats_by_generation.plot.line()


# **III. Styling the plots**

# 1. Increase the size of a scattered plot (defense vs. attack for each Pokemon)

# In[ ]:


pokemon.plot.scatter(x='defense', 
                     y='attack', 
                     figsize=(12,6))


# 2. Increase the size of a histogram representing 'Total strength' density over the whole set of Pokemons, change bucket color, add plot title, increase font for both title and axis, and set the number of bins  
#   
# *Hint: the column name is 'base_total'*

# In[ ]:


my_plot = pokemon.base_total.plot.hist(figsize=(12,6),
                             color='gray',
                             fontsize=16,
                             bins=50)

my_plot.set_title('Density by base total', fontsize=20)


# 3. Increase the size of a bar chart (representing Pokemon frequency by Primary type), add title, increase font for both title and axis, and delete plot borders  
#   
# *Hint: the column name is 'type1'*

# In[ ]:


my_plot = pokemon.type1.value_counts().plot.bar(figsize=(12,6),
                                                fontsize=14)

my_plot.set_title('Pokemon by primary type', fontsize=20)
sns.despine(bottom=True,
            left=True)


# III. Subplots

# 1. Create a 2x1 grid of size 8x8

# In[ ]:


fig, axarr = plt.subplots(2, 1, figsize=(8,8))


# 2. Visualize defense frequency and attack frequency (using histograms) on 2x1 grid and create titles for both

# In[ ]:


fig, axarr = plt.subplots(2, 1, figsize=(8,8))
pokemon.defense.plot.hist(ax=axarr[0], bins=40)
axarr[0].set_title('Defense frequency', fontsize=14)
pokemon.attack.plot.hist(ax=axarr[1], bins=40)
axarr[1].set_title('Attack frequency', fontsize=14)


# **IV. Plotting in Seaborn**

# 1. Represent the frequency of each Pokemon generation (using a Seaborn barchart)

# In[ ]:


sns.countplot(pokemon.generation)


# 2. Represent density of Hit Points in a Pokemon set (using a Seaburn histogram with KDE line)

# In[ ]:


sns.distplot(pokemon.hp)


# 3. Create a joint plot (scatter and individual histograms) representing Attack-Defense level relation for the whole dataset

# In[ ]:


sns.jointplot(x=pokemon.attack, y=pokemon.defense)


# 4. Same as before, but should include a hex plot instead of a scatter plot

# In[ ]:


sns.jointplot(x=pokemon.attack, y=pokemon.defense, kind='hex')


# 5. Relation of Hit Points-Attack level in a dataset (using a Seaborn 2D kde plot)

# In[ ]:


sns.kdeplot(pokemon.hp, pokemon.attack, shade=True)


# 6. Show attack frequency for different types of Legendary property (using a Seaborn box plot and violin plot)  
#   
# *Hint: the column names are 'attack' and 'is_legendary'*

# In[ ]:


sns.boxplot(x=pokemon.is_legendary, y=pokemon.attack)


# In[ ]:


sns.violinplot(x=pokemon.is_legendary, y=pokemon.attack)


# **V. Faceting in Seaborn**

# 1. Build separate KDE plots for Attack frequency (for each of the legendary levels)

# In[ ]:


grid = sns.FacetGrid(pokemon, row='is_legendary')
grid.map(sns.kdeplot, 'attack')


# 2. Same as before, but should have separate plots for each of the legendary levels (horizontal grid) and  generation (vertical grid)

# In[ ]:


grid = sns.FacetGrid(pokemon, col='is_legendary', row='generation')
grid.map(sns.kdeplot, 'attack')


# 3. Compare in a pair-like manner parameters: Hit Points, Attack, Defense (using a Seaborn Pairplot)

# In[ ]:


sns.pairplot(pokemon[['hp', 'attack', 'defense']])


# **VI. Multivariate plotting**

# 1. Build Attack - Defense scatter plot for the whole dataset using different markers for different legendary levels (in terms of color and shape)

# In[ ]:


sns.lmplot(x='attack', y='defense', hue='is_legendary', fit_reg=False, data=pokemon, markers = ['x', 'o'])


# 2. Show total score statistics for each generation (using in a boxplot), where different color stand for different legendary level

# In[ ]:


sns.boxplot(x='generation', y='base_total', hue='is_legendary', data=pokemon)


# 3. Create a Seaborn heatmap representing correlation between parameters: Hit Points, Attack, Special Attack, Defense, Special Defense, Speed

# In[ ]:


sns.heatmap(pokemon.loc[:,['hp', 'attack', 'sp_attack', 'defense', 'sp_defense', 'speed']].corr(), annot=True)


# 4. Use parallel coordinates to represent each value of several qualities (attack, special attack, defense, special defense) where different types of Pokemon stand for a different line color (here, we consider only Fighting and Psychic types)

# In[ ]:


from pandas.plotting import parallel_coordinates
df = pokemon[pokemon.type1.isin(['fighting', 'psychic'])].loc[:, ['type1', 'attack', 'sp_attack', 'defense', 'sp_defense']]
parallel_coordinates(df, 'type1', colors=['violet', 'blue'])

