#!/usr/bin/env python
# coding: utf-8

# ![485bf0a6ed5ff1306f699452e22bada1.jpg](attachment:485bf0a6ed5ff1306f699452e22bada1.jpg)

# ## *IMPORT LIBRARIES FOR EDA,DATA VISUALIZATION*

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')


# ![](https://www.google.com/url?sa=i&url=https%3A%2F%2Ffactsd.com%2Fpokemon-facts%2F&psig=AOvVaw1e0P1iCEOwztarkcY6p0xL&ust=1593067924328000&source=images&cd=vfe&ved=0CAIQjRxqFwoTCLC2tdrumeoCFQAAAAAdAAAAABAM)

# ## *READING POKEMON DATASET*

# In[ ]:


df = pd.read_csv('../input/pokemon/PokemonData.csv')


# In[ ]:


df.head()


# ## *Scatter Plot*

# In[ ]:


plt.figure(figsize=(16,12))
sns.lmplot(x='Attack',y='Defense',data=df) ## seaborn and matplotlib


# In[ ]:


sns.lmplot(x='Attack',y='Defense',data=df,fit_reg=False,hue='Generation')


# ## *Box Plot*

# In[ ]:


plt.figure(figsize=(16,12))
sns.boxplot(data=df)


# In[ ]:


plt.figure(figsize=(16,12))
stats_df=df.drop(['Generation','Legendary'],axis=1)
sns.boxplot(data=stats_df)


# ## *VIOLIN PLOT*

# In[ ]:


plt.figure(figsize=(16,12))
sns.set_style('whitegrid')
sns.violinplot(x='Type1',y='Attack',data=df)


# In[ ]:


pkmn_type_colors = ['#78C850',  # Grass
                    '#F08030',  # Fire
                    '#6890F0',  # Water
                    '#A8B820',  # Bug
                    '#A8A878',  # Normal
                    '#A040A0',  # Poison
                    '#F8D030',  # Electric
                    '#E0C068',  # Ground
                    '#EE99AC',  # Fairy
                    '#C03028',  # Fighting
                    '#F85888',  # Psychic
                    '#B8A038',  # Rock
                    '#705898',  # Ghost
                    '#98D8D8',  # Ice
                    '#7038F8',  # Dragon
                   ]


# In[ ]:


plt.figure(figsize=(16,12))
sns.violinplot(x='Type1',y='Attack',data=df,palette=pkmn_type_colors)


# In[ ]:


df.head(50)


# ## *Swarm Plot*

# In[ ]:


plt.figure(figsize=(16,16))
sns.swarmplot(x='Type1',y='Attack',data=df,palette=pkmn_type_colors)


# ## *Overlaying Plots*

# In[ ]:


# Set figure size with matplotlib
plt.figure(figsize=(16,16))
 
# Create plot
sns.violinplot(x='Type1',
               y='Attack', 
               data=df, 
               inner=None, # Remove the bars inside the violins
               palette=pkmn_type_colors)
 
sns.swarmplot(x='Type1', 
              y='Attack', 
              data=df, 
              color='k', # Make points black
              alpha=0.7) # and slightly transparent
 
# Set title with matplotlib
plt.title('Attack by Type')


# In[ ]:


df.head(6)


# ## *Melting all Using Melt Function in Pandas*

# In[ ]:


# Melt DataFrame
melted_df = pd.melt(stats_df, 
                    id_vars=["Name", "Type1", "Type2","HP"], # Variables to keep
                    var_name="Stat") # Name of melted variable


# In[ ]:


melted_df.head(30)


# ## *Shape comparsion of two data frame*

# In[ ]:


print(df.shape) #Original Data Frame
print(melted_df.shape) #Melted Data Frame


# ## *Making a swarm plot using melted_df*

# In[ ]:


plt.figure(figsize=(16,16))
sns.swarmplot(x='Stat',y='value',data=melted_df,hue='Type1')


# In[ ]:


df.head(10)


# In[ ]:


plt.figure(figsize=(16,16)) # 1. Sizing the plot

sns.swarmplot(x='Stat', 
              y='value', 
              data=melted_df, 
              hue='Type1', 
              dodge=True, # 2. Separate points by hue
              palette=pkmn_type_colors) # 3. Use Pokemon palette

plt.legend(bbox_to_anchor=(1, 1), loc=2) # 4. setting the box to the right


# ## *HeatMap Ploting*

# ### *HeatMap helps you to visulize matrix like data!!*

# In[ ]:


plt.figure(figsize=(16,16))

# calculate correlation

corr = df.corr()

# Plotting heatmap using Pokedex dataset!!!
sns.heatmap(corr)


# ## ***Histogram***

# In[ ]:


# Distribution Plot

plt.figure(figsize=(16,12)) # Figure Size

sns.distplot(df.Attack)


#  ## ***Bar Plot***

# In[ ]:


# Bar plots help you visualize the distributions of categorical variables.

# count plot

plt.figure(figsize=(16,9))

sns.countplot(x='Type1',data=df, palette=pkmn_type_colors)

# changing rotations of labels

plt.xticks(rotation=-45)


# In[ ]:


df.head(7)


# ## ***Density Plot***

# ### ***Density plots display the distribution between two variables!!!***

# In[ ]:


plt.figure(figsize=(16,14))
sns.kdeplot(df.Attack, df.Defense)


# ## ****Joint Distribution Plot****

# ### ***Joint distribution plots combine information from scatter plots and histograms to give you detailed information for bi-variate distributions!!!***

# In[ ]:


plt.figure(figsize=(16,14))
sns.jointplot(x='Attack', y='Defense', data=df)


# # Pls Upvote!!
