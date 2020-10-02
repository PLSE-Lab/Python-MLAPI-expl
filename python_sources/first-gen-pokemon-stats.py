#!/usr/bin/env python
# coding: utf-8

# # Pokemon Stats
# 
# > "I want to be the very best!" - Source Unknown

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# # Let's Normalize Pokemon to Fit 1st Generation Types

# In[ ]:


pokemon = pd.read_csv("../input/Pokemon.csv")
pokemon = pokemon[pokemon.Generation == 1].drop_duplicates('#')
type1 = pokemon['Type 1'].unique()
pk_type1 = pokemon.groupby('Type 1').count()['#']

pokemon['Type 1'] = pokemon['Type 1'].str    .replace('Ice', 'Water')    .replace('Fairy', 'Normal')    .replace('Dragon', 'Normal')    .replace('Bug', 'Grass')    .replace('Toxic', 'Grass')


# In[ ]:


types = pokemon['Type 1']

colors = [
    'yellowgreen',
    'gold',
    'lightskyblue',
    'lightcoral',
    'yellow',
    'lightgreen',
    'silver',
    'white',
    'pink'
]

explode = np.arange(len(types.unique())) * 0.01

types.value_counts().plot.pie(
    explode=explode,
    colors=colors,
    title="Percentage of Different Types of Pokemon",
    autopct='%1.1f%%',
    shadow=True,
    startangle=90,
    figsize=(8,8)
)
plt.tight_layout()


# # Let's Look at How Attack and Defense is Distributed

# In[ ]:


sns.jointplot(x="Attack", y="Defense", data=pokemon);


# # Now Special Attack and Defense

# In[ ]:


sns.jointplot(x="Sp. Atk", y="Sp. Def", data=pokemon);


# # What is the Spread in Stats

# In[ ]:


sns.boxplot(data=pokemon.drop(['#', 'Total', 'Generation', 'Legendary'], 1), orient='h')


# # Wow, Stats are Pretty Balanced
# 
# ## Closer Look

# In[ ]:


normalized = pd.melt(pokemon.drop(['Generation', 'Legendary', '#', 'Total'],1), id_vars=["Name", "Type 1", "Type 2"], var_name="Stat")

plt.figure(figsize=(12,10))
plt.ylim(0, 275)
sns.swarmplot(
    x="Stat",
    y="value",
    data=normalized,
    hue="Type 1",
    split=True,
    size=7
)
plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.);


# In[ ]:


def plot_variation(pokemon, stat):
    hp_data = pokemon[['Name','Type 1', stat]]
    hp_data = hp_data.pivot_table(values=stat, index=['Name'], columns=['Type 1'])
    f, ax = plt.subplots(figsize=(18, 6))
    sns.violinplot(data=hp_data, palette="Set3", bw=.2, cut=1, linewidth=1)
    ax.set(ylim=(0, 200))
    ax.set_title("{} of Different Types of Pokemon".format(stat))
    sns.despine(left=True, bottom=True)


# In[ ]:


plot_variation(pokemon, 'HP')


# In[ ]:


plot_variation(pokemon, 'Attack')


# In[ ]:


plot_variation(pokemon, 'Defense')


# In[ ]:


plot_variation(pokemon, 'Speed')


# In[ ]:


plot_variation(pokemon, 'Sp. Atk')


# In[ ]:


plot_variation(pokemon, 'Sp. Def')


# # Let's do some PCA

# [Original analysis](https://www.kaggle.com/strakul5/d/abcsds/pokemon/principal-component-analysis-of-pokemon-data), but fitted to 151 poekmon

# In[ ]:


df = pokemon
cols = ['HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']
pokemon['id'] = pokemon['#']


# In[ ]:


scaler = StandardScaler().fit(pokemon[cols])
df_scaled = scaler.transform(pokemon[cols])


# # Enough components to explain 80% of the variance

# In[ ]:


pca = PCA(n_components=0.8)
pca.fit(df_scaled)

pcscores = pd.DataFrame(pca.transform(df_scaled))
pcscores.columns = ['PC'+str(i+1) for i in range(len(pcscores.columns))]

loadings = pd.DataFrame(pca.components_, columns=cols)
loadings.index = ['PC'+str(i+1) for i in range(len(pcscores.columns))]


# In[ ]:


sns.heatmap(pd.DataFrame(pca.components_**2, columns=cols).transpose(), annot=True, linewidths=0.5);


# The darkest shades in the plot above indicate which parameters are the most important.
# 
# For example, the loading factors for `2` show that HP is the most dominant parameter.
# That is, Pokemon with high HP will have high absolute values of `2`.

# In[ ]:


labels = set(df['Type 1'])
df['type'] = df['Type 1']
lab_dict = dict()
for i, elem in enumerate(labels):
    lab_dict[elem] = i
df = df.replace({'type' : lab_dict})

pc_types = pcscores.copy()
pc_types['Type'] = df['Type 1']

# Biplots
def make_plot(pcscores, loadings, xval=0, yval=1, max_arrow=0.2, alpha=0.4):
    n = loadings.shape[1]
    scalex = 1.0 / (pcscores.iloc[:, xval].max() - pcscores.iloc[:, xval].min())  # Rescaling to be from -1 to +1
    scaley = 1.0 / (pcscores.iloc[:, yval].max() - pcscores.iloc[:, yval].min())

    pcscores.iloc[:, xval] = pcscores.iloc[:, xval] * scalex
    pcscores.iloc[:, yval] = pcscores.iloc[:, yval] * scaley

    g = sns.lmplot(x='PC{}'.format(xval + 1), y='PC{}'.format(yval + 1), hue='Type', data=pcscores,
                   fit_reg=False, size=6, palette='muted')

    for i in range(n):
        # Only plot the longer ones
        length = np.sqrt(loadings.iloc[xval, i] ** 2 + loadings.iloc[yval, i] ** 2)
        if length < max_arrow:
            continue

        plt.arrow(0, 0, loadings.iloc[xval, i], loadings.iloc[yval, i], color='k', alpha=0.9)
        plt.text(loadings.iloc[xval, i] * 1.15, loadings.iloc[yval, i] * 1.15,
                 loadings.columns.tolist()[i], color='k', ha='center', va='center')

    g.set(ylim=(-1, 1))
    g.set(xlim=(-1, 1))


# In[ ]:


make_plot(pc_types, loadings, 2, 3, max_arrow=0.3)


# In[ ]:


make_plot(pc_types, loadings, 1, 2, max_arrow=0.3)


# # Top Pokemon by HP

# In[ ]:


df.sort_values(by='HP', ascending=False).head(n=25)


# In[ ]:


sns.pairplot(pc_types, hue='Type');

